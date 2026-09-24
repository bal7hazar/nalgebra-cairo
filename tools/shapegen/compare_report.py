#!/usr/bin/env python3
"""`snforge test` output of the staging comparison (`compare.sh`) -> the Markdown of `COMPARE.md`:
one row per compared operation, raw `l2_gas` of the generated and of the hand-written side."""

import re
import sys
from collections import defaultdict

LINE = re.compile(r"\[PASS\]\s+\S+::(bench_(\w+?)__(generated|handwritten)|test_(\w+)_bit_identical)"
                  r"\s+\(.*l2_gas:\s*~?(\d+)\)")


def main() -> int:
    gas: dict[str, dict[str, int]] = defaultdict(dict)
    identical = set()
    for line in sys.stdin:
        m = LINE.search(line)
        if not m:
            continue
        if m.group(4):
            identical.add(m.group(4))
        else:
            gas[m.group(2)][m.group(3)] = int(m.group(5))
    rows, equal = [], 0
    for group in sorted(gas):
        g, h = gas[group].get("generated"), gas[group].get("handwritten")
        same = g is not None and g == h
        equal += same
        bits = "yes" if group in identical else "**no**"
        rows.append(f"| `{group}` | {g} | {h} | {'=' if same else '≠'} | {bits} |")
    ref = sys.argv[1] if len(sys.argv) > 1 else "?"
    print("# Staging comparison: generated vs hand-written library shapes (WP 8.1b-1)\n")
    print(f"`tools/shapegen/compare.sh` against `crates/nalgebra` at `{ref}` (hand-written "
          f"`Vector2/3/4`, `Matrix2/3/4`). Raw `l2_gas` of `bench_<group>__generated` / "
          f"`__handwritten` (same black-boxed inputs); `bits`: `test_<group>_bit_identical` "
          f"passed (equal `Serde` images on three inputs).\n")
    print(f"{equal} / {len(rows)} groups equal, {len(identical)} bit-identity tests passed.\n")
    print("| group | generated | hand-written | gas | bits |")
    print("|---|---:|---:|:---:|:---:|")
    print("\n".join(rows))
    return 0 if equal == len(rows) and len(identical) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
