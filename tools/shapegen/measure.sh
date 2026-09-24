#!/usr/bin/env bash
# Compile budget of the generated shapes (tools/shapegen/DESIGN.md, "CI impact").
#
#   tools/shapegen/measure.sh <shapes> <label> [shapegen options]
#
# Generates a throw-away package tools/shapegen/.tmp-measure-<label> (`--no-compare`: no
# dependency on crates/nalgebra), then measures, holding the machine-wide build lock FIRST so
# that the waiting time is not counted, wall time and peak RSS of:
#   1. `scarb build`          (the library alone),
#   2. `scarb build --test`   (library + generated tests, the unit snforge compiles),
#   3. `snforge test`         (compilation to CASM + execution).
# Prints one Markdown table row.
set -euo pipefail
cd "$(dirname "$0")/../.."
shapes=$1
label=$2
shift 2
LOCK="${HEAVY_BUILD_LOCK:-$HOME/orchestrator/heavy-build.lock}"
pkg="tools/shapegen/.tmp-measure-$label"
rm -rf "$pkg"
timeout 300 python3 tools/shapegen/shapegen.py --out "$pkg" --shapes "$shapes" --no-compare "$@" >/dev/null
lib_lines=$(cat "$pkg"/src/*.cairo | wc -l)
test_lines=$( (cat "$pkg"/src/tests/*.cairo "$pkg"/tests/*.cairo 2>/dev/null || true) | wc -l)
tests=$( (grep -h "^#\[test\]" "$pkg"/src/tests/*.cairo "$pkg"/tests/*.cairo 2>/dev/null || true) | wc -l)
measure() {  # prints "seconds MB"
    (cd "$pkg" && flock "$LOCK" env HEAVY_BUILD_LOCK_HELD=1 /usr/bin/time -f "%e %U %S %M" "$@" \
        >/tmp/shapegen-measure.log 2>/tmp/shapegen-measure.time) || {
        cat /tmp/shapegen-measure.log /tmp/shapegen-measure.time >&2; exit 1; }
    tail -n 1 /tmp/shapegen-measure.time | awk '{ printf "%s s (cpu %.1f s), %.0f MB", $1, $2 + $3, $4 / 1024 }'
}
rm -rf "$pkg/target"
lib=$(measure scarb build)
rm -rf "$pkg/target"
testbuild=$(measure scarb build --test)
run=$(measure snforge test)
passed=$(grep -o "Tests: [0-9]* passed, [0-9]* failed" /tmp/shapegen-measure.log || echo "?")
echo "| $label | $shapes | $lib_lines | $test_lines | $tests | $lib | $testbuild | $run | $passed |"
rm -rf "$pkg"
