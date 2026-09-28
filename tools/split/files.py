#!/usr/bin/env python3
"""Library lines per file of a Cairo crate (the counting rules of scripts/consumer_cost.py).

Walks the `mod x;` tree from `src/lib.cairo` exactly like `consumer_cost.py` (test-only files and
items excluded) and prints `lines<TAB>path` per file, or JSON with `--json`. WP 9-NS1 helper.

    python3 tools/split/files.py crates/nalgebra [--json out.json]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import consumer_cost as cc  # noqa: E402


def files(crate_dir):
    src = os.path.join(crate_dir, "src")
    root = os.path.join(src, "lib.cairo")
    out = {}

    def walk(path):
        raw, _nonblank, mods, _ex, _ = cc.analyse_file(path)
        out[os.path.relpath(path, src)] = raw
        for name in mods:
            child = cc.module_file(path, name)
            if child is not None:
                walk(child)

    walk(root)
    return out


def main():
    crate = sys.argv[1]
    res = files(crate)
    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(res, f, indent=1, sort_keys=True)
    else:
        for p, n in sorted(res.items()):
            print(f"{n}\t{p}")
        print(f"{sum(res.values())}\tTOTAL")


if __name__ == "__main__":
    main()
