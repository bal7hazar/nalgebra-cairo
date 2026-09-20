#!/usr/bin/env python3
"""Cross-representation summary (net Sierra gas) from `snforge test -p scalar` output on stdin.

`scripts/gas_report.py` ranks variants inside a group (`<op>_<storage>`); this script pivots the
same numbers into one table: operation x representation.
"""
import re
import sys
from collections import defaultdict

LINE = re.compile(r"\[PASS\]\s+\S*::bench_(\w+?)__(\w+)\s+\(.*l2_gas:\s*~?(\d+)\)")
STORES = ["sm64", "i64", "i128", "felt", "i32", "sm32"]
COLS = [
    ("sm64", "signmag", "A SignMag64"),
    ("i64", "native", "B I64Q32"),
    ("i64", "bounded", "G BI64"),
    ("i64", "bounded_bi", "G (bi add)"),
    ("i64", "bounded_fused", "G fused"),
    ("i128", "native", "C I128Q64"),
    ("felt", "felt", "D FeltQ32"),
    ("felt", "felt_lazy", "D lazy"),
    ("felt", "felt_soft", "D soft"),
    ("felt", "felt_plain", "D plain"),
    ("i32", "native", "E I32Q16"),
    ("sm32", "signmag", "E' SignMag32"),
]
OPS = ["add", "sub", "neg", "abs", "mul", "div", "lt", "eq", "from_int", "to_int", "sqrt", "dot3",
       "cross3", "lerp", "mul_add", "length3", "add4", "mul4"]

gas = defaultdict(dict)
for line in sys.stdin:
    m = LINE.search(line)
    if m:
        gas[m.group(1)][m.group(2)] = int(m.group(3))

rows = []
for op in OPS:
    cases = []
    for store in STORES:
        for variant in gas.get(f"{op}_{store}", {}):
            if variant != "baseline":
                case = variant.rsplit("_", 1)[1]
                if case not in cases:
                    cases.append(case)
    order = ["pp", "pn", "np", "nn", "p", "n", "mix"]
    for case in sorted(cases, key=order.index):
        row = [f"`{op}` {case}"]
        for store, key, _ in COLS:
            group = gas.get(f"{op}_{store}", {})
            v = group.get(f"{key}_{case}")
            row.append(str(v - group["baseline"]) if v is not None else "")
        rows.append(row)

# Marginal cost of a chained op: (op4 - op1) / 3.
for op in ("add", "mul"):
    row = [f"`{op}` marginal (pn)"]
    for store, key, _ in COLS:
        g1, g4 = gas.get(f"{op}_{store}", {}), gas.get(f"{op}4_{store}", {})
        v1, v4 = g1.get(f"{key}_pn"), g4.get(f"{key}_pn")
        ok = v1 is not None and v4 is not None
        row.append(str(round(((v4 - g4["baseline"]) - (v1 - g1["baseline"])) / 3)) if ok else "")
    rows.append(row)

print("| op | " + " | ".join(c[2] for c in COLS) + " |")
print("|---|" + "---:|" * len(COLS))
for row in rows:
    print("| " + " | ".join(row) + " |")
