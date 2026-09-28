#!/usr/bin/env python3
"""Zero-step proof for a package-split move (WP 9-NS1, used by NS2..n).

Compares two gas snapshot directories (`gas/*.json`, the format of `scripts/gas_report.py`:
{module path: {group: {variant: gas}}}) after normalising the module paths: the package segment
of `nalgebra` and of every `nalgebra_<crate>` sub-crate is dropped, so
`nalgebra::base::matrix3::tests` (before) and `nalgebra_core::base::matrix3::tests` (after) are
the same benchmark. Test packages (`nalgebra_tests_*`, `nalgebra_shapes_tests_*`) keep their
names and are compared as they are.

Exit status 1 if any benchmark changes value, disappears or appears (a move adds or removes
none: it only moves code).

    git worktree add /tmp/base origin/main   # or: git show main:gas/x.json > old/x.json ...
    python3 tools/split/gas_compare.py /tmp/base/gas gas
"""
import glob
import json
import os
import re
import sys

KEEP = re.compile(r"^nalgebra_(tests|shapes_tests|testing|glam)")


def normalise(module):
    head, _, rest = module.partition("::")
    if (head == "nalgebra" or head.startswith("nalgebra_")) and not KEEP.match(head):
        return "<nalgebra>::" + rest
    return module


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        for module, groups in json.load(open(f)).items():
            for group, variants in groups.items():
                for variant, gas in variants.items():
                    key = f"{normalise(module)}::{group}__{variant}"
                    if key in out and out[key] != gas:
                        sys.exit(f"error: {key} appears twice with different values in {d}")
                    out[key] = gas
    return out


def main():
    old, new = load(sys.argv[1]), load(sys.argv[2])
    changed = [(k, old[k], new[k]) for k in sorted(old.keys() & new.keys()) if old[k] != new[k]]
    gone = sorted(old.keys() - new.keys())
    added = sorted(new.keys() - old.keys())
    for k, a, b in changed:
        print(f"CHANGED {k}: {a} -> {b}")
    for k in gone:
        print(f"MISSING {k}")
    for k in added:
        print(f"ADDED   {k}")
    print(f"{len(old)} benchmarks before, {len(new)} after: {len(changed)} changed, "
          f"{len(gone)} missing, {len(added)} added")
    sys.exit(1 if changed or gone or added else 0)


if __name__ == "__main__":
    main()
