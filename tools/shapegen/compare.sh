#!/usr/bin/env bash
# Staging comparison of the generated library shapes against the hand-written ones (DESIGN.md
# §2.2, `compare.py`): generates tools/shapegen/.tmp-compare, runs its tests and benches, and
# writes the table of `COMPARE.md` (every row must read `=` or `<`). Exit 1 on a failed test or a
# gas increase. Meaningful BEFORE the generated files replace the hand-written ones, or against the
# commit that still had them: `compare.sh <ref>` compares with the crates of `<ref>` (`git archive`).
set -euo pipefail
cd "$(dirname "$0")/../.."
pkg=tools/shapegen/.tmp-compare
ref=${1:-HEAD}
root=.
rm -rf "$pkg"
if [[ $# -gt 0 ]]; then
    root=$(mktemp -d)
    git archive "$ref" crates Scarb.toml | tar -x -C "$root"
fi
timeout 300 python3 tools/shapegen/shapegen.py --compare "$pkg" --compare-root "$root"
log=$(mktemp)
(cd "$pkg" && snforge test) > "$log" 2>&1 || { tail -n 60 "$log"; exit 1; }
timeout 300 python3 tools/shapegen/compare_report.py "$(git rev-parse --short "$ref")" < "$log" \
    > tools/shapegen/COMPARE.md
tail -n 3 "$log"
rm -rf "$pkg" "$log"
if [[ $root != . ]]; then rm -rf "$root"; fi
grep -q "| > |" tools/shapegen/COMPARE.md && { echo "gas increases, see COMPARE.md" >&2; exit 1; }
echo "compare: no row dearer (tools/shapegen/COMPARE.md)"
