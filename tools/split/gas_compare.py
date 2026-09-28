#!/usr/bin/env python3
"""Zero-step proof of a package-split move (docs/SPLIT.md §5; WP 9-NS1, productised in 9-NS2).

Compares two sets of gas snapshots (`gas/*.json`, the format of `scripts/gas_report.py`:
{module path: {group: {variant: gas}}}) after normalising the module paths: the package segment
of the LIBRARY packages (`nalgebra`, every package of the crate map `tools/split/crates.toml`, and
any `nalgebra_<name>` that is not a test / tooling package) is dropped, so
`nalgebra::base::matrix3::tests` (before the move) and `nalgebra_core::base::matrix3::tests`
(after) are the same benchmark. Test packages (`nalgebra_tests_*`, `nalgebra_shapes_tests_*`,
`nalgebra_testing`) and `nalgebra_glam` keep their names. The snapshot FILES do not matter
(`gas_report.py` names them after the CI shard, which a move renames): every benchmark of every
file is compared.

Exit status 1 if any benchmark changes value, disappears or appears: a move PR only moves code,
so it adds, removes and changes nothing.

    python3 tools/split/gas_compare.py --base origin/main --head gas/   # a git ref: its gas/
    python3 tools/split/gas_compare.py --base /tmp/old/gas --head gas/  # a directory
    python3 tools/split/gas_compare.py OLD_DIR NEW_DIR                  # NS1's form, still accepted

A move PR regenerates the snapshots of the packages it touches (`snforge test -p <pkg> |
python3 scripts/gas_report.py --update gas/`, one package after the other), then runs the first
form against its base branch and pastes the summary line in the PR (docs/ORCHESTRATOR.md).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
KEEP = re.compile(r"^nalgebra_(tests|shapes_tests|testing|glam)")


def library_packages():
    """The library packages of the crate map (none if the map cannot be read)."""
    try:
        sys.path.insert(0, HERE)
        import cratemap

        return set(cratemap.load().packages())
    except Exception:  # noqa: BLE001 (the comparison stays usable outside the repository)
        return set()


LIBRARY = library_packages() | {"nalgebra"}


def normalise(module):
    head, _, rest = module.partition("::")
    if head in LIBRARY or (head.startswith("nalgebra_") and not KEEP.match(head)):
        return "<nalgebra>::" + rest
    return module


def add(out, source, data):
    for module, groups in data.items():
        for group, variants in groups.items():
            for variant, gas in variants.items():
                key = f"{normalise(module)}::{group}__{variant}"
                if key in out and out[key] != gas:
                    sys.exit(f"error: {key} appears twice with different values in {source}")
                out[key] = gas


def load_dir(d):
    out = {}
    files = sorted(glob.glob(os.path.join(d, "*.json")))
    if not files:
        sys.exit(f"error: no gas snapshot (*.json) in {d}")
    for f in files:
        with open(f) as fh:
            add(out, d, json.load(fh))
    return out


def load_ref(ref, gas_dir="gas"):
    """The snapshots of a git ref (`git show REF:gas/x.json`), without a checkout."""
    try:
        names = subprocess.run(["git", "ls-tree", "--name-only", f"{ref}:{gas_dir}"], cwd=ROOT,
                               check=True, capture_output=True, text=True).stdout.split()
    except subprocess.CalledProcessError as e:
        sys.exit(f"error: {ref!r} is neither a directory nor a git ref with {gas_dir}/: {e.stderr.strip()}")
    out = {}
    for n in sorted(x for x in names if x.endswith(".json")):
        text = subprocess.run(["git", "show", f"{ref}:{gas_dir}/{n}"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout
        add(out, f"{ref}:{gas_dir}/{n}", json.loads(text))
    if not out:
        sys.exit(f"error: no gas snapshot in {ref}:{gas_dir}/")
    return out


def load(spec):
    return load_dir(spec) if os.path.isdir(spec) else load_ref(spec)


def compare(old, new):
    """(changed [(key, old, new)], missing [key], added [key])."""
    changed = [(k, old[k], new[k]) for k in sorted(old.keys() & new.keys()) if old[k] != new[k]]
    return changed, sorted(old.keys() - new.keys()), sorted(new.keys() - old.keys())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", help="git ref (its gas/) or snapshot directory before the move")
    ap.add_argument("--head", help="snapshot directory after the move (default: gas/)")
    ap.add_argument("dirs", nargs="*", help="OLD NEW (NS1's form)")
    a = ap.parse_args(argv)
    if a.dirs:
        if len(a.dirs) != 2 or a.base or a.head:
            ap.error("give either --base/--head or two directories")
        base, head = a.dirs
    else:
        if not a.base:
            ap.error("--base is required")
        base, head = a.base, a.head or os.path.join(ROOT, "gas")
    old, new = load(base), load(head)
    changed, gone, added = compare(old, new)
    for k, x, y in changed:
        print(f"CHANGED {k}: {x} -> {y}")
    for k in gone:
        print(f"MISSING {k}")
    for k in added:
        print(f"ADDED   {k}")
    print(f"{len(old)} benchmarks before, {len(new)} after: {len(changed)} changed, "
          f"{len(gone)} missing, {len(added)} added")
    return 1 if changed or gone or added else 0


if __name__ == "__main__":
    sys.exit(main())
