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
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import consumer_cost as cc  # noqa: E402

PREFIX = "nalgebra_"
EXTERNAL = {"simba": 'simba = "0.2.0"', "glam": 'glam = "0.4.0"', "fixed": 'fixed = "0.4.0"'}


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
            name = os.path.basename(m.rstrip("/"))
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
            s, rss, samples = cc.measure(key, manifest(proto, members), args.repeat, "scarb")
            cache[key] = {"seconds": s, "gb": rss / cc.GB, "samples": samples}
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
        rows.append((c, ", ".join(d.lstrip("@") for d in deps), full["seconds"] - dep["seconds"],
                     full["gb"] - dep["gb"], full["seconds"] - base["seconds"], full["gb"] - base["gb"]))
    print("| crate | direct deps | marginal s | marginal GB | closure s | closure GB |")
    print("|---|---|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]:.1f} | {r[3]:.2f} | {r[4]:.1f} | {r[5]:.2f} |")
    if args.closure:
        print("\n| closure | crates | added s | added GB |")
        print("|---|---|---:|---:|")
        for spec in args.closure:
            name, members = spec.split("=", 1)
            ms = [m for m in members.split(",") if m]
            r = cost(ms)
            if r is None:
                continue
            print(f"| {name} | {', '.join(ms)} | {r['seconds'] - base['seconds']:.1f} | "
                  f"{r['gb'] - base['gb']:.2f} |")
    print(f"\nbaseline (no dependency): {base['seconds']:.1f} s, {base['gb']:.2f} GB")


if __name__ == "__main__":
    main()
