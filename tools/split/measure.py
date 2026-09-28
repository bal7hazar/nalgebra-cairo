#!/usr/bin/env python3
"""Marginal and closure cost of the crates of a prototype workspace (WP 9-NS1 helper).

The package granularity gates, as the programme session defined them (2026-09-28):

  * gate 2, MARGINAL cost of crate X = cost(empty consumer of X) - cost(empty consumer of the DIRECT
    dependencies of X together), at most 5 s and 1 GB;
  * gate 3, a DECLARED closure = cost(empty consumer of the closure's crates) - cost(empty
    consumer with no dependency), at most 15 s and 3 GB.

Every cost is a cold `scarb build` of a trivial consumer (`scripts/consumer_cost.py`'s `measure`,
`SCARB_INCREMENTAL=false`, fresh `target/`). Results are cached in a JSON file keyed by the set of
dependencies, so an interrupted run resumes and a dependency set shared by two crates is built once.
Run it under the shared build lock so that the wall time excludes lock waits:

    flock ~/orchestrator/heavy-build.lock python3 tools/split/measure.py PROTO_DIR \\
        --cache results.json [--crate X ...] [--closure NAME=a,b ...] [--repeat N] \\
        [--interleave] [--json OUT.json]

`--interleave` (the GitHub-runner protocol, `.github/workflows/split-measure.yml`): every
dependency set still missing from the cache is built once per ROUND, `--repeat` rounds, the order
rotated each round, so a slow period of the machine hits every variant alike; a marginal's spread
is then the spread of the per-round differences (crate minus its direct dependencies, closure minus
baseline). `--json` writes the rows (medians, quartiles of the per-round differences, samples).
"""
import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))

PREFIX = "nalgebra_"
EXTERNAL = {"simba": 'simba = "0.2.0"', "glam": 'glam = "0.4.0"', "fixed": 'fixed = "0.4.0"',
            "glam_core": 'glam_core = "0.4.1"', "glam_int": 'glam_int = "0.4.1"'}


def cold_build(label, manifest_text, repeat):
    """Median wall seconds, CPU seconds (user + sys) and peak RSS (GB) of cold consumer builds."""
    env = dict(os.environ, SCARB_INCREMENTAL="false")
    walls, cpus, rsss = [], [], []
    with tempfile.TemporaryDirectory(prefix="split_cost_") as tmp:
        os.makedirs(os.path.join(tmp, "src"))
        with open(os.path.join(tmp, "Scarb.toml"), "w") as f:
            f.write(manifest_text)
        with open(os.path.join(tmp, "src", "lib.cairo"), "w") as f:
            f.write("pub fn consumer_cost_answer() -> u32 {\n    42\n}\n")
        subprocess.run(["scarb", "fetch"], cwd=tmp, env=env, capture_output=True, text=True)
        for k in range(repeat):
            shutil.rmtree(os.path.join(tmp, "target"), ignore_errors=True)
            tfile = os.path.join(tmp, "time.out")
            t0 = time.monotonic()
            p = subprocess.run(["/usr/bin/time", "-v", "-o", tfile, "scarb", "build"], cwd=tmp,
                               env=env, capture_output=True, text=True)
            wall = time.monotonic() - t0
            if p.returncode != 0:
                sys.exit(f"error: build of {label} failed:\n{p.stdout[-3000:]}{p.stderr[-2000:]}")
            t = open(tfile).read()
            user = float(re.search(r"User time \(seconds\): ([\d.]+)", t).group(1))
            syst = float(re.search(r"System time \(seconds\): ([\d.]+)", t).group(1))
            rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1)) * 1024
            print(f"  {label} [{k + 1}/{repeat}]: {wall:.1f} s wall, {user + syst:.1f} s CPU, "
                  f"{rss / 1e9:.2f} GB", file=sys.stderr)
            walls.append(wall)
            cpus.append(user + syst)
            rsss.append(rss)
    return statistics.median(walls), statistics.median(cpus), statistics.median(rsss) / 1e9, {
        "wall": walls, "cpu": cpus, "rss_bytes": rsss}


def prepare(tmp, label, manifest_text):
    """A consumer directory for `manifest_text`, its dependencies fetched."""
    d = os.path.join(tmp, re.sub(r"\W+", "_", label)[:80] or "baseline")
    os.makedirs(os.path.join(d, "src"))
    with open(os.path.join(d, "Scarb.toml"), "w") as f:
        f.write(manifest_text)
    with open(os.path.join(d, "src", "lib.cairo"), "w") as f:
        f.write("pub fn consumer_cost_answer() -> u32 {\n    42\n}\n")
    p = subprocess.run(["scarb", "fetch"], cwd=d, env=dict(os.environ, SCARB_INCREMENTAL="false"),
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"error: fetch of {label} failed:\n{p.stdout[-3000:]}{p.stderr[-2000:]}")
    return d


def one_build(label, d):
    """(wall s, CPU s, peak RSS bytes) of one cold build of the consumer in `d`."""
    shutil.rmtree(os.path.join(d, "target"), ignore_errors=True)
    tfile = os.path.join(d, "time.out")
    t0 = time.monotonic()
    p = subprocess.run(["/usr/bin/time", "-v", "-o", tfile, "scarb", "build"], cwd=d,
                       env=dict(os.environ, SCARB_INCREMENTAL="false"), capture_output=True, text=True)
    wall = time.monotonic() - t0
    if p.returncode != 0:
        sys.exit(f"error: build of {label} failed:\n{p.stdout[-3000:]}{p.stderr[-2000:]}")
    t = open(tfile).read()
    user = float(re.search(r"User time \(seconds\): ([\d.]+)", t).group(1))
    syst = float(re.search(r"System time \(seconds\): ([\d.]+)", t).group(1))
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", t).group(1)) * 1024
    return wall, user + syst, rss


def interleaved(jobs, repeat):
    """{label: cache entry} of cold builds of every consumer of `jobs` ({label: manifest text}),
    one build of each per round, `repeat` rounds, the order rotated by one each round."""
    labels = sorted(jobs)
    samples = {k: {"wall": [], "cpu": [], "rss_bytes": []} for k in labels}
    with tempfile.TemporaryDirectory(prefix="split_cost_") as tmp:
        dirs = {k: prepare(tmp, k, jobs[k]) for k in labels}
        for r in range(repeat):
            for k in labels[r % len(labels):] + labels[: r % len(labels)]:
                wall, cpu, rss = one_build(k, dirs[k])
                print(f"  round {r + 1}/{repeat} {k}: {wall:.1f} s wall, {cpu:.1f} s CPU, "
                      f"{rss / 1e9:.2f} GB", file=sys.stderr, flush=True)
                samples[k]["wall"].append(wall)
                samples[k]["cpu"].append(cpu)
                samples[k]["rss_bytes"].append(rss)
    return {k: {"seconds": statistics.median(v["wall"]), "cpu": statistics.median(v["cpu"]),
                "gb": statistics.median(v["rss_bytes"]) / 1e9, "samples": v, "interleaved": True}
            for k, v in samples.items()}


def quartiles(xs):
    """(q1, median, q3) of a list (inclusive method; a single value repeated for one sample)."""
    if len(xs) < 2:
        return (xs[0], xs[0], xs[0]) if xs else (None, None, None)
    q = statistics.quantiles(xs, n=4, method="inclusive")
    return q[0], statistics.median(xs), q[2]


def paired(a, b, field="wall"):
    """Quartiles of the per-round differences a - b (interleaved samples), None otherwise."""
    sa, sb = a["samples"][field], b["samples"][field]
    if not (a.get("interleaved") and b.get("interleaved")) or len(sa) != len(sb):
        return None
    return quartiles([x - y for x, y in zip(sa, sb)])


def direct_deps(proto, crate):
    text = open(os.path.join(proto, "crates", crate, "Scarb.toml")).read()
    deps = re.search(r"^\[dependencies\]\n(.*?)(?=^\[|\Z)", text, re.S | re.M).group(1)
    out = []
    for line in deps.splitlines():
        m = re.match(r"\s*(\w+)\s*=", line)
        if m:
            name = m.group(1)
            out.append(name[len(PREFIX):] if name.startswith(PREFIX) else "@" + name)
    return sorted(out)


def manifest(proto, members):
    lines = ['[package]', 'name = "cost_consumer"', 'version = "0.1.0"', 'edition = "2024_07"',
             "", "[dependencies]"]
    for m in sorted(members):
        if m.startswith("@"):
            lines.append(EXTERNAL[m[1:]])
        elif m.startswith("/"):
            name = re.search(r'^name = "(\w+)"', open(os.path.join(m, "Scarb.toml")).read(), re.M).group(1)
            lines.append(f'{name} = {{ path = "{m}" }}')
        else:
            d = os.path.join(proto, "crates", m)
            name = re.search(r'^name = "(\w+)"', open(os.path.join(d, "Scarb.toml")).read(), re.M).group(1)
            lines.append(f'{name} = {{ path = "{d}" }}')
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proto")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--crate", action="append", default=[])
    ap.add_argument("--closure", action="append", default=[], metavar="NAME=a,b")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--report", action="store_true", help="print the table from the cache only")
    ap.add_argument("--interleave", action="store_true",
                    help="build the missing dependency sets round by round (see the docstring)")
    ap.add_argument("--json", help="write the rows (medians, spreads, samples) to this file")
    args = ap.parse_args()
    proto = os.path.abspath(args.proto)
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}
    closures = [(spec.split("=", 1)[0], [m for m in spec.split("=", 1)[1].split(",") if m])
                for spec in args.closure]
    if args.interleave and not args.report:
        wanted = [[]] + [m for c in args.crate for m in ([c], direct_deps(proto, c))] + \
            [ms for _n, ms in closures]
        jobs = {}
        for ms in wanted:
            key = ",".join(sorted(ms)) or "(baseline)"
            if key not in cache:
                jobs[key] = manifest(proto, ms)
        if jobs:
            print(f"measuring {len(jobs)} consumers x {args.repeat} rounds, interleaved",
                  file=sys.stderr)
            cache.update(interleaved(jobs, args.repeat))
            with open(args.cache, "w") as f:
                json.dump(cache, f, indent=1, sort_keys=True)

    def cost(members):
        key = ",".join(sorted(members)) or "(baseline)"
        if key not in cache:
            if args.report:
                return None
            print(f"measuring {key}", file=sys.stderr)
            s, cpu, gb, samples = cold_build(key, manifest(proto, members), args.repeat)
            cache[key] = {"seconds": s, "cpu": cpu, "gb": gb, "samples": samples}
            with open(args.cache, "w") as f:
                json.dump(cache, f, indent=1, sort_keys=True)
        return cache[key]

    base = cost([])
    rows = []
    out = {"baseline": base, "crates": {}, "closures": {}}
    for c in args.crate:
        deps = direct_deps(proto, c)
        full, dep = cost([c]), cost(deps)
        if full is None or dep is None:
            continue
        out["crates"][c] = {"deps": deps, "full": full, "direct_deps": dep,
                            "marginal_s": full["seconds"] - dep["seconds"],
                            "marginal_cpu": full["cpu"] - dep["cpu"], "marginal_gb": full["gb"] - dep["gb"],
                            "marginal_s_quartiles": paired(full, dep),
                            "marginal_gb_quartiles": paired(full, dep, "rss_bytes")}
        rows.append((c, ", ".join(d.lstrip("@") for d in deps),
                     full["seconds"] - dep["seconds"], full["cpu"] - dep["cpu"], full["gb"] - dep["gb"],
                     full["seconds"] - base["seconds"], full["cpu"] - base["cpu"], full["gb"] - base["gb"]))
    print("| crate | direct deps | marginal s (wall) | marginal s (CPU) | marginal GB "
          "| closure s (wall) | closure s (CPU) | closure GB |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]:.1f} | {r[3]:.1f} | {r[4]:.2f} | {r[5]:.1f} | {r[6]:.1f} | {r[7]:.2f} |")
    if args.closure:
        print("\n| closure | crates | added s (wall) | added s (CPU) | added GB |")
        print("|---|---|---:|---:|---:|")
        for name, ms in closures:
            r = cost(ms)
            if r is None:
                continue
            out["closures"][name] = {"members": ms, "full": r, "added_s": r["seconds"] - base["seconds"],
                                     "added_cpu": r["cpu"] - base["cpu"], "added_gb": r["gb"] - base["gb"],
                                     "added_s_quartiles": paired(r, base),
                                     "added_gb_quartiles": paired(r, base, "rss_bytes")}
            print(f"| {name} | {', '.join(ms)} | {r['seconds'] - base['seconds']:.1f} | "
                  f"{r['cpu'] - base['cpu']:.1f} | {r['gb'] - base['gb']:.2f} |")
    print(f"\nbaseline (no dependency): {base['seconds']:.1f} s wall, {base['cpu']:.1f} s CPU, "
          f"{base['gb']:.2f} GB")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
