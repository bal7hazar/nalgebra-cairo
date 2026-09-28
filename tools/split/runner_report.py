#!/usr/bin/env python3
"""Tables of a GitHub-runner measurement of the split (WP 9-NS1b, `.github/workflows/split-measure.yml`).

Merges the `measure.py --json` files of the workflow's jobs (one per crate marginal, one per
declared closure) with the plan (`mapplan.py`) and the prototype's library lines (`files.py`, the
counting rules of `scripts/consumer_cost.py`), judges the gates of `consumer_cost.toml` and
writes a Markdown report (the tables of docs/SPLIT.md §3.1 and §6.2) and a JSON summary:

    python3 tools/split/runner_report.py PLAN.json PROTO RESULTS_DIR --md OUT.md --json OUT.json

Marginal = median(consumer of the crate) - median(consumer of its direct dependencies); closure =
median(consumer of the closure's crates) - median(consumer with no dependency); the spread is the
interquartile range [q1, q3] of the per-round differences (interleaved builds, one runner).
"""
import argparse
import glob
import json
import os
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from files import files  # noqa: E402


def gates():
    with open(os.path.join(ROOT, "consumer_cost.toml"), "rb") as f:
        return tomllib.load(f)["gates"]


def spread(q, scale=1.0):
    if not q or q[0] is None:
        return "–"
    return f"{q[0] / scale:.1f} – {q[2] / scale:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("plan")
    ap.add_argument("proto")
    ap.add_argument("results")
    ap.add_argument("--md", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    g = gates()
    crates, closures, rounds = {}, {}, set()
    for path in sorted(glob.glob(os.path.join(a.results, "**", "*.json"), recursive=True)):
        d = json.load(open(path))
        if not isinstance(d, dict) or not isinstance(d.get("crates"), dict) or "baseline" not in d:
            continue
        crates.update(d["crates"])
        closures.update(d["closures"])
        rounds.add(len(d["baseline"]["samples"]["wall"]))
    rows, fails = [], []
    for c in plan["crates"] + ["glam_bridge"]:
        cdir = os.path.join(a.proto, "crates", "facade" if c == "facade" else c)
        if not os.path.isdir(cdir):
            continue
        lines = sum(files(cdir).values())
        pkg = "nalgebra" if c == "facade" else ("nalgebra_glam (bridge)" if c == "glam_bridge"
                                                  else "nalgebra_" + c)
        r = crates.get(c)
        row = {"crate": c, "package": pkg, "lines": lines,
               "deps": [d.lstrip("@") for d in (r["deps"] if r else plan["deps"].get(c, []))
                        if d not in ("@simba",)]}
        if r:
            row.update(marginal_s=r["marginal_s"], marginal_cpu=r["marginal_cpu"],
                       marginal_gb=r["marginal_gb"], marginal_s_quartiles=r["marginal_s_quartiles"],
                       full_s=r["full"]["seconds"], samples=len(r["full"]["samples"]["wall"]))
        verdict = []
        if c != "facade":
            if lines > g["max_lines"]:
                verdict.append("lines")
            if r and r["marginal_s"] > g["max_seconds"]:
                verdict.append("seconds")
            if r and r["marginal_gb"] > g["max_gb"]:
                verdict.append("GB")
        row["verdict"] = "fail: " + ", ".join(verdict) if verdict else ("pass" if r else "not measured")
        if verdict:
            fails.append(f"{pkg}: {', '.join(verdict)}")
        rows.append(row)
    crows = []
    for name, spec in plan.get("closures", {}).items():
        r = closures.get(name)
        row = {"closure": name, "members": spec}
        if r:
            ok = r["added_s"] < g["closure_seconds"] and r["added_gb"] < g["closure_gb"]
            row.update(added_s=r["added_s"], added_cpu=r["added_cpu"], added_gb=r["added_gb"],
                       added_s_quartiles=r["added_s_quartiles"], verdict="pass" if ok else "FAIL")
            if not ok and not name.startswith("glam_0"):
                fails.append(f"closure {name}")
        else:
            row["verdict"] = "not measured"
        crows.append(row)
    out = [f"### Split measurement ({len(rows)} crates, rounds per consumer: "
           f"{', '.join(str(x) for x in sorted(rounds)) or '–'})", "",
           "| crate | lines | direct deps (besides simba) | marginal s (median) | spread s (IQR) "
           "| CPU s | GB | verdict |", "|---|---:|---|---:|---:|---:|---:|---|"]
    for r in rows:
        if "marginal_s" in r:
            m = (f"{r['marginal_s']:.1f} | {spread(r['marginal_s_quartiles'])} | {r['marginal_cpu']:.1f} | "
                 f"{r['marginal_gb']:.2f}")
        else:
            m = "– | – | – | –"
        out.append(f"| `{r['package']}` | {r['lines']:,} | {', '.join(r['deps']) or '–'} | {m} | {r['verdict']} |")
    out += ["", "| closure | crates | added s (median) | spread s (IQR) | CPU s | GB | verdict |",
            "|---|---|---:|---:|---:|---:|---|"]
    for r in crows:
        if "added_s" in r:
            m = (f"{r['added_s']:.1f} | {spread(r['added_s_quartiles'])} | {r['added_cpu']:.1f} | "
                 f"{r['added_gb']:.2f}")
        else:
            m = "– | – | – | –"
        out.append(f"| {r['closure']} | {', '.join(r['members'])} | {m} | {r['verdict']} |")
    out += ["", f"Gates: {g['max_lines']:,} lines, marginal ≤ {g['max_seconds']} s / {g['max_gb']} GB, "
            f"declared closure < {g['closure_seconds']} s / {g['closure_gb']} GB. "
            + ("**Failing: " + "; ".join(fails) + "**" if fails else "Every measured row passes.")]
    with open(a.md, "w") as f:
        f.write("\n".join(out) + "\n")
    with open(a.json, "w") as f:
        json.dump({"crates": rows, "closures": crows, "fails": fails}, f, indent=1)
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
