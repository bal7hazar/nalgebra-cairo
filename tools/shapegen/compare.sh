#!/usr/bin/env bash
# Staging comparison of the generated library shapes against the hand-written ones (DESIGN.md
# §2.2, `compare.py`): generates tools/shapegen/.tmp-compare, runs its tests and benches, and
# writes the table of `COMPARE.md` (every row must read `=`). Exit 1 on a failed test or a gas
# difference. Meaningful BEFORE the generated files replace the hand-written ones.
set -euo pipefail
cd "$(dirname "$0")/../.."
pkg=tools/shapegen/.tmp-compare
rm -rf "$pkg"
timeout 300 python3 tools/shapegen/shapegen.py --compare "$pkg"
log=$(mktemp)
(cd "$pkg" && snforge test) > "$log" 2>&1 || { tail -n 60 "$log"; exit 1; }
timeout 300 python3 tools/shapegen/compare_report.py "$(git rev-parse --short HEAD)" < "$log" \
    > tools/shapegen/COMPARE.md
tail -n 3 "$log"
rm -rf "$pkg" "$log"
grep -q "| ≠ |" tools/shapegen/COMPARE.md && { echo "gas differs, see COMPARE.md" >&2; exit 1; }
echo "compare: every row equal (tools/shapegen/COMPARE.md)"
