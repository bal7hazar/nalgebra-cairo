#!/usr/bin/env bash
# Short local check to run before every push (the `.githooks/pre-push` hook runs it); catches what CI
# catches cheaply. `scripts/check.sh` stays the full local equivalent of CI.
#
# Usage: scripts/prepush.sh [BASE]    BASE: the ref the push is compared against (default: origin/main).
#
# Always run (seconds, no lock):
#   scarb fmt --check                              formatting (this and the three checks below run side by side)
#   python3 scripts/consumer_cost.py --self-test   the self-tests of the scripts (no scarb, no network);
#   python3 scripts/packages_table.py --self-test  these three are the only scripts under scripts/ or
#   python3 scripts/release.py --self-test         tools/ that have one
#   python3 scripts/api_parity.py --check          the generated artefacts that need no build, as scripts/check.sh
#   python3 tools/shapegen/shapegen.py --check     runs them
#   python3 tools/linalggen/generate.py --check
# The Cairo compile steps run only when an input changed between the merge base of BASE and the working tree:
#   scarb lint --deny-warnings, scarb build       when crates/**, tests/**, any Scarb.toml, Scarb.lock or
#       .tool-versions changed: of the TOUCHED packages only (each changed path is mapped to its package with
#       `scarb metadata`); a change to the root Scarb.toml, Scarb.lock or .tool-versions means the whole workspace.
#   snforge test -p <package> | python3 scripts/gas_report.py --check gas/ --partial    the gas snapshot check of
#       the touched packages, as one CI shard does, when crates/**, tests/**, gas/**, a Scarb.toml, Scarb.lock,
#       .tool-versions or scripts/gas_report.py changed. A change of gas/** or scripts/gas_report.py alone touches
#       no package: the whole workspace is tested and checked against the whole snapshot, as scripts/check.sh does.
#   (The `nalgebra` package is tested with the Scarb features CI gives it: --no-default-features --features eigen,svd,qr.)
#
# The heavy-build lock. On the shared VPS the scarb/snforge shims serialise every compile through one lock
# file ($HEAVY_BUILD_LOCK, default ~/orchestrator/heavy-build.lock; it prevents running out of memory) and
# let a nested call run without re-locking when an ancestor process already holds that file. The Cairo steps
# are therefore run as ONE block under `flock -E 75 -w 90 <that same lock file>`: this takes the real lock,
# waiting at most 90 s for its turn. Inside it the scarb/snforge SHIMS are still called, with
# HEAVY_BUILD_LOCK_HELD=1 exported (the lock really is held), which they honour as a pass-through: they
# never wait on the lock a second time. It never bypasses the lock, and it only gives up waiting: it never
# signals or kills the process holding the lock, nor a compile that runs (the lock file is touched only through
# `flock -w`). If the lock is not obtained in 90 s the whole Cairo block is skipped, with the single line
# `heavy lock busy: Cairo compile left to CI`, and the push is not blocked (CI runs them). Without the lock
# (no lock directory or no `flock`, as on the Mac) or when a caller already holds it (HEAVY_BUILD_LOCK_HELD),
# the Cairo block runs directly, with no wait. The time the lock was waited for and the time the block itself
# took are printed separately.
#
# Git: every git command here is a read of the repository being pushed (rev-parse, merge-base, diff); none
# writes, so none needs a sanitised environment.
# Exits non-zero on the first failure, with a one-line message naming the step.
set -euo pipefail
cd "$(dirname "$0")/.."
# The Cairo compiler is not deterministic on several threads: Sierra, and so the gas snapshot, are only
# reproducible single-threaded.
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"

start=$SECONDS
step="start"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "prepush: FAILED at step: ${step} ($((SECONDS - start))s)" >&2; fi' EXIT

run() { step="$1"; shift; local t=$SECONDS; echo "prepush: $step"; "$@"; echo "prepush: $step: $((SECONDS - t))s"; }

# The Cairo block: run inside the lock (inner mode, re-executed by flock) or directly.
# PREPUSH_PKGS: the touched packages, space separated, or "*" for the whole workspace.
cairo_block() {
    local t=$SECONDS pkg
    local -a pflags=()
    if [[ "$PREPUSH_DO_BUILD" == 1 ]]; then
        if [[ "$PREPUSH_PKGS" != "*" ]]; then
            for pkg in $PREPUSH_PKGS; do pflags+=(-p "$pkg"); done
        fi
        if [[ "$PREPUSH_PKGS" == "*" || -n "$PREPUSH_PKGS" ]]; then
            run "scarb lint --deny-warnings ${pflags[*]}" scarb lint --deny-warnings "${pflags[@]}"
            run "scarb build ${pflags[*]}" scarb build "${pflags[@]}"
        fi
    fi
    if [[ "$PREPUSH_DO_GAS" == 1 ]]; then
        local t2=$SECONDS output="" out2
        if [[ "$PREPUSH_PKGS" == "*" || -z "$PREPUSH_PKGS" ]]; then
            step="snforge test --workspace"
            echo "prepush: $step, then the gas snapshot check"
            output=$(snforge test --workspace) || { echo "$output"; exit 1; }
            echo "$output" | tail -n 1
            echo "prepush: $step: $((SECONDS - t2))s"
            step="gas_report.py --check gas/"
            echo "$output" | python3 scripts/gas_report.py --check gas/
        else
            for pkg in $PREPUSH_PKGS; do
                local -a extra=()
                [[ "$pkg" == "nalgebra" ]] && extra=(--no-default-features --features eigen,svd,qr)
                step="snforge test -p $pkg"
                echo "prepush: $step ${extra[*]:-}"
                out2=$(snforge test -p "$pkg" "${extra[@]}") || { echo "$out2"; exit 1; }
                echo "$out2" | tail -n 1
                output+="$out2"$'\n'
            done
            echo "prepush: snforge test of the touched packages: $((SECONDS - t2))s"
            step="gas_report.py --check gas/ --partial"
            echo "$output" | python3 scripts/gas_report.py --check gas/ --partial
        fi
    fi
    echo "prepush: Cairo block: $((SECONDS - t))s of work"
}

if [[ "${PREPUSH_INNER:-}" == 1 ]]; then
    : > "$PREPUSH_MARK" # the lock is held: tells the caller it was obtained
    export HEAVY_BUILD_LOCK_HELD=1 # the lock really is held here: the shims run nested, without re-locking
    echo "prepush: heavy lock obtained after $(($(date +%s) - PREPUSH_T0))s of waiting"
    cairo_block
    exit 0
fi

base="${1:-origin/main}"
step="resolve base $base"
git rev-parse --verify --quiet "$base^{commit}" > /dev/null || { echo "prepush: base ref '$base' not found (git fetch origin main?)" >&2; exit 1; }
merge_base=$(git merge-base "$base" HEAD)
changed=$(git diff --name-only --no-renames "$merge_base")

# The independent checks (none holds the lock, none writes) run side by side to keep the wall time down; they
# are judged in this order, and the first failing one is named.
logs=$(mktemp -d)
trap 'rm -rf "$logs"; rc=$?; if [[ $rc -ne 0 ]]; then echo "prepush: FAILED at step: ${step} ($((SECONDS - start))s)" >&2; fi' EXIT
names=("scarb fmt --check" "api_parity.py --check" "shapegen.py --check" "linalggen/generate.py --check")
cmds=("scarb fmt --check" "python3 scripts/api_parity.py --check" "python3 tools/shapegen/shapegen.py --check" "python3 tools/linalggen/generate.py --check")
pids=()
step="scripts' self-tests"
run "consumer_cost.py --self-test" python3 scripts/consumer_cost.py --self-test
run "packages_table.py --self-test" python3 scripts/packages_table.py --self-test
run "release.py --self-test" python3 scripts/release.py --self-test
for i in "${!cmds[@]}"; do
    ( t=$SECONDS; ${cmds[$i]} > "$logs/$i.log" 2>&1; rc=$?; echo "prepush: ${names[$i]}: $((SECONDS - t))s" >> "$logs/$i.log"; exit $rc ) &
    pids+=($!)
done
failed=""
for i in "${!cmds[@]}"; do
    if wait "${pids[$i]}"; then rc=0; else rc=$?; fi
    echo "prepush: ${names[$i]}"
    cat "$logs/$i.log"
    if [[ $rc -ne 0 && -z "$failed" ]]; then failed="${names[$i]}"; fi
done
if [[ -n "$failed" ]]; then step="$failed"; exit 1; fi

PREPUSH_DO_BUILD=0
PREPUSH_DO_GAS=0
if grep -Eq '^(crates/|tests/|(.*/)?Scarb\.toml$|Scarb\.lock$|\.tool-versions$)' <<< "$changed"; then PREPUSH_DO_BUILD=1; fi
if grep -Eq '^(crates/|tests/|gas/|(.*/)?Scarb\.toml$|Scarb\.lock$|\.tool-versions$|scripts/gas_report\.py$)' <<< "$changed"; then PREPUSH_DO_GAS=1; fi
export PREPUSH_DO_BUILD PREPUSH_DO_GAS

if [[ "$PREPUSH_DO_BUILD" == 0 && "$PREPUSH_DO_GAS" == 0 ]]; then
    echo "prepush: Cairo compile skipped (no Cairo source, manifest or gas input changed against $base)"
else
    # The touched packages: each changed path is mapped to the workspace package whose directory holds it;
    # "*" (the whole workspace) for a root manifest, Scarb.lock or .tool-versions.
    step="map the changed paths to packages (scarb metadata)"
    PREPUSH_PKGS=$(PREPUSH_CHANGED="$changed" scarb metadata --format-version 1 --no-deps | python3 -c '
import json, os, sys
meta = json.load(sys.stdin)
root = meta["workspace"]["root"]
if any(f in ("Scarb.toml", "Scarb.lock", ".tool-versions") for f in os.environ["PREPUSH_CHANGED"].split("\n")):
    print("*")
    sys.exit(0)
dirs = {os.path.relpath(p["root"], root): p["name"] for p in meta["packages"]}
touched = set()
for f in os.environ["PREPUSH_CHANGED"].split("\n"):
    d = os.path.dirname(f)
    while d:
        if d in dirs:
            touched.add(dirs[d])
            break
        d = os.path.dirname(d)
print(" ".join(sorted(touched)))
')
    export PREPUSH_PKGS
    echo "prepush: touched packages: ${PREPUSH_PKGS:-none (gas/ or gas_report.py only: the whole workspace is checked)}"
    lock="${HEAVY_BUILD_LOCK:-$HOME/orchestrator/heavy-build.lock}"
    if [[ -z "${HEAVY_BUILD_LOCK_HELD:-}" ]] && command -v flock > /dev/null && [[ -d "$(dirname "$lock")" ]]; then
        step="Cairo compile (the failing step is named above)"
        mark=$(mktemp)
        rm -f "$mark"
        # PREPUSH_MARK appears only once the lock is held: a failed step is told apart from a lock that was
        # not obtained (flock exit 75 after the 90 s) and from any other flock error.
        rc=0
        PREPUSH_INNER=1 PREPUSH_MARK="$mark" PREPUSH_T0=$(date +%s) flock -E 75 -w 90 "$lock" "$0" "$base" || rc=$?
        if [[ -e "$mark" ]]; then
            rm -f "$mark"
            [[ $rc -eq 0 ]] || exit 1 # a step failed inside the lock; it printed its own message
        elif [[ $rc -eq 75 ]]; then
            echo "heavy lock busy: Cairo compile left to CI"
        elif [[ $rc -ne 0 ]]; then
            echo "prepush: flock failed on $lock (exit $rc)" >&2
            exit 1
        fi
    else
        cairo_block
    fi
fi

echo "prepush: ok in $((SECONDS - start))s"
