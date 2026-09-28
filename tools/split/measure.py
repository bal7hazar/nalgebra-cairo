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
        --cache results.json [--crate X ...] [--closure NAME=a,b ...] [--repeat N]
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
EXTERNAL = {"simba": 'simba = "0.2.0"', "glam": 'glam = "0.4.0"', "fixed": 'fixed = "0.4.0"'}


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
            lines.append(f'{PREFIX}{m} = {{ path = "{os.path.join(proto, "crates", m)}" }}')
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proto")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--crate", action="append", default=[])
    ap.add_argument("--closure", action="append", default=[], metavar="NAME=a,b")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--report", action="store_true", help="print the table from the cache only")
    args = ap.parse_args()
    proto = os.path.abspath(args.proto)
    cache = json.load(open(args.cache)) if os.path.exists(args.cache) else {}

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
    for c in args.crate:
        deps = direct_deps(proto, c)
        full, dep = cost([c]), cost(deps)
        if full is None or dep is None:
            continue
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
        for spec in args.closure:
            name, members = spec.split("=", 1)
            ms = [m for m in members.split(",") if m]
            r = cost(ms)
            if r is None:
                continue
            print(f"| {name} | {', '.join(ms)} | {r['seconds'] - base['seconds']:.1f} | "
                  f"{r['cpu'] - base['cpu']:.1f} | {r['gb'] - base['gb']:.2f} |")
    print(f"\nbaseline (no dependency): {base['seconds']:.1f} s wall, {base['cpu']:.1f} s CPU, "
          f"{base['gb']:.2f} GB")


if __name__ == "__main__":
    main()
