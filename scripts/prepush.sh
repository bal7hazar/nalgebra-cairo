#!/usr/bin/env bash
# Short local check to run before every push (the `.githooks/pre-push` hook runs it); catches what CI
# catches cheaply. `scripts/check.sh` stays the full local equivalent of CI.
#
# Usage: scripts/prepush.sh [BASE]    BASE: the ref the push is compared against (default: origin/main).
#
# Every check is gated on its inputs, as CI is (the changed paths are those between the merge base of BASE and
# the working tree). Prose `.md` files trigger nothing, so a push that changes only those takes seconds. A change
# of `.github/workflows/**`, `.tool-versions`, any `Scarb.toml` or `Scarb.lock` (the toolchain and manifests)
# triggers every fixed check below (the Cairo steps are gated on their own inputs, further down). Otherwise:
#   scarb fmt --check                              when a `.cairo` file changed
#   python3 scripts/consumer_cost.py --self-test   the self-tests of the scripts (no scarb, no network), each
#   python3 scripts/packages_table.py --self-test  when its own script changed (these three are the only
#   python3 scripts/release.py --self-test         scripts under scripts/ or tools/ that have one)
#   python3 scripts/api_parity.py --check          when it, docs/API_PARITY.md (a checked artefact), a non-`.md`
#                                                  file of tools/split/ or a `.cairo` file of crates/ changed
#   python3 tools/shapegen/shapegen.py --check     when a non-`.md` file of tools/shapegen/, tools/linalggen/ or
#   python3 tools/linalggen/generate.py --check    tools/split/ (generators and crate map) or a `.cairo` file of
#                                                  crates/ (the checked outputs) changed
# These checks (none holds the lock, none writes) run side by side; the first failing one, in this order, is named.
# The Cairo compile steps run only when an input changed:
#   scarb lint --deny-warnings, scarb build       when crates/**, tests/**, any Scarb.toml, Scarb.lock or
#       .tool-versions changed: of the TOUCHED packages only (each changed path is mapped to its package with
#       `scarb metadata`); a change to the root Scarb.toml, Scarb.lock or .tool-versions means the whole workspace.
#   snforge test -p <package> | python3 scripts/gas_report.py --check gas/ --partial    the gas snapshot check of
#       the touched packages, as one CI shard does, when crates/**, tests/**, gas/**, a Scarb.toml, Scarb.lock,
#       .tool-versions or scripts/gas_report.py changed. When gas/** or scripts/gas_report.py changed (alone or
#       with a package), the whole workspace is tested and checked against the whole snapshot, as scripts/check.sh
#       does: the snapshot is the thing that changed, so a partial comparison would not vouch for it.
#   A manifest or lock file outside the root workspace (benchmarks/**, tools/**: Scarb.toml, Scarb.lock) touches no
#       package of it: unless gas/** or scripts/gas_report.py changed too, there is nothing to compile, no lock is
#       waited for and no workspace test runs.
#   (The `nalgebra` package is tested with the Scarb features CI gives it: --no-default-features --features eigen,svd,qr.)
#
# The heavy-build lock. On the shared VPS the scarb/snforge shims serialise every compile through one lock
# file ($HEAVY_BUILD_LOCK, default ~/orchestrator/heavy-build.lock; it prevents running out of memory) and
# let a nested call run without re-locking when an ancestor process already holds that file. The Cairo steps
# are therefore run as ONE block under `flock -E 75 -w 90 <that same lock file>`: this takes the real lock,
# waiting at most 90 s for its turn. The wait starts FIRST, in the background, alongside the fixed checks (which
# need no lock): the lock-busy case costs about the 90 s, not the 90 s plus the checks. A waiter that obtains the
# lock runs the block at once (the lock is never held idle), its output kept and printed after the fixed checks; if
# a fixed check fails, or this script is interrupted (INT, TERM, HUP), this script sends TERM to the process
# groups of the jobs IT started (the lock waiter with its own compile, and the fixed checks) and to nothing
# else; the waiter also stops at its next step boundary. Inside the lock the
# scarb/snforge SHIMS are still called, with HEAVY_BUILD_LOCK_HELD=1 exported (the lock really is held), which
# they honour as a pass-through: they never wait on the lock a second time. It never bypasses the lock, and it
# only gives up waiting: it never signals any other holder of the lock, nor a compile that is not its own (the
# lock file is touched only through `flock -w`). If the lock is not obtained in 90 s the whole Cairo block is
# skipped, with the single line `heavy lock busy: Cairo compile left to CI`, and the push is not blocked (CI
# runs them). Without the lock (no lock directory or no `flock`, as on the Mac), or when a caller already holds
# it (HEAVY_BUILD_LOCK_HELD, or an ancestor process holding the lock file, the test of the shims), the Cairo
# block runs directly, with no wait: a caller does not wait for itself. The time the lock was waited for and
# the time the block itself took are printed separately.
#
# Bash: needs bash >= 4.4 (`wait $!` on a process substitution in the hook, empty arrays under `set -u`); an older
# bash fails at once with a clear message (the macOS system bash is 3.2: use a newer one first in PATH).
#
# Git: every git command here is a read of the repository being pushed (rev-parse, merge-base, diff); none
# writes, so none needs a sanitised environment.
# Exits non-zero on the first failure, with a one-line message naming the step.
if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4))); then
    echo "prepush: bash >= 4.4 is required (this is bash ${BASH_VERSION}); put a newer bash first in PATH." >&2
    exit 1
fi
set -euo pipefail
# The script's own absolute path, resolved once before any `cd` (it is re-run by path under the lock).
self=$(realpath "$0" 2> /dev/null || { cd "$(dirname "$0")" && echo "$(pwd -P)/$(basename "$0")"; })
cd "$(dirname "$self")/.."
# The Cairo compiler is not deterministic on several threads: Sierra, and so the gas snapshot, are only
# reproducible single-threaded.
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"

start=$SECONDS
step="start"
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "prepush: FAILED at step: ${step} ($((SECONDS - start))s)" >&2; fi' EXIT

# Inner mode only: stop at a step boundary when the caller has refused the push (a fixed check failed) or is gone.
stop_if_refused() {
    [[ "${PREPUSH_INNER:-}" == 1 ]] || return 0
    if [[ -s "$PREPUSH_GATE" && "$(< "$PREPUSH_GATE")" == stop ]] || ! kill -0 "$PREPUSH_PARENT" 2> /dev/null; then
        echo "prepush: stopped before step '$1': the push is already refused"
        exit 0
    fi
}

run() { stop_if_refused "$1"; step="$1"; shift; local t=$SECONDS; echo "prepush: $step"; "$@"; echo "prepush: $step: $((SECONDS - t))s"; }

lock="${HEAVY_BUILD_LOCK:-$HOME/orchestrator/heavy-build.lock}"

# The test of the scarb/snforge shims: does an ancestor process hold the lock file open?
ancestor_holds_lock() {
    local p=$PPID
    while [[ -n "$p" ]] && [[ "$p" -gt 1 ]] 2> /dev/null; do
        if ls -l "/proc/$p/fd" 2> /dev/null | grep -qF -- "$lock"; then return 0; fi
        p=$(awk '{print $4}' "/proc/$p/stat" 2> /dev/null) || return 1
    done
    return 1
}

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
        if [[ "$PREPUSH_PKGS" == "*" || -z "$PREPUSH_PKGS" || "$PREPUSH_GAS_FULL" == 1 ]]; then
            stop_if_refused "snforge test --workspace"
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
                stop_if_refused "snforge test -p $pkg"
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

# Inner mode: re-executed by flock, so the lock is held. It tells the caller (PREPUSH_MARK), then runs the block at
# once, alongside the fixed checks, so the lock is never held idle; its output goes to a file (PREPUSH_OUT) that the
# caller prints after the fixed checks. When the caller refuses the push (PREPUSH_GATE says stop) or is gone, it
# stops at the next step boundary; the caller also sends TERM to this job's process group on refusal or interrupt.
if [[ "${PREPUSH_INNER:-}" == 1 ]]; then
    : > "$PREPUSH_MARK"
    waited=$(($(date +%s) - PREPUSH_T0))
    exec > "$PREPUSH_OUT" 2>&1
    export HEAVY_BUILD_LOCK_HELD=1 # the lock really is held here: the shims run nested, without re-locking
    echo "prepush: heavy lock obtained after ${waited}s of waiting (the wait ran alongside the fixed checks)"
    cairo_block
    exit 0
fi

base="${1:-origin/main}"
step="resolve base $base"
git rev-parse --verify --quiet "$base^{commit}" > /dev/null || { echo "prepush: base ref '$base' not found (git fetch origin main?)" >&2; exit 1; }
merge_base=$(git merge-base "$base" HEAD)
changed=$(git diff --name-only --no-renames "$merge_base")
changed_code=$(grep -Ev '\.md$' <<< "$changed" || true) # prose `.md` triggers nothing; a checked `.md` is tested on $changed

# has RE: a changed path matches; any RE: it matches, or the toolchain / manifests / workflows changed (all trigger)
has() { grep -Eq "$1" <<< "$changed"; }
has_code() { grep -Eq "$1" <<< "$changed_code"; }
all_re='^\.github/workflows/|^\.tool-versions$|(^|/)Scarb\.lock$|(^|/)Scarb\.toml$'
everything=0
if has "$all_re"; then everything=1; fi
gen_re='^(tools/(shapegen|linalggen|split)/|crates/.*\.cairo$)'

# The fixed checks that apply (none holds the lock, none writes).
names=()
cmds=()
add() { names+=("$1"); cmds+=("$2"); }
if [[ $everything == 1 ]] || has_code '\.cairo$'; then add "scarb fmt --check" "scarb fmt --check"; fi
if [[ $everything == 1 ]] || has '^scripts/consumer_cost\.py$'; then add "consumer_cost.py --self-test" "python3 scripts/consumer_cost.py --self-test"; fi
if [[ $everything == 1 ]] || has '^scripts/packages_table\.py$'; then add "packages_table.py --self-test" "python3 scripts/packages_table.py --self-test"; fi
if [[ $everything == 1 ]] || has '^scripts/release\.py$'; then add "release.py --self-test" "python3 scripts/release.py --self-test"; fi
if [[ $everything == 1 ]] || has '^(scripts/api_parity\.py|docs/API_PARITY\.md)$' || has_code '^(tools/split/|crates/.*\.cairo$)'; then
    add "api_parity.py --check" "python3 scripts/api_parity.py --check"
fi
if [[ $everything == 1 ]] || has_code "$gen_re"; then
    add "shapegen.py --check" "python3 tools/shapegen/shapegen.py --check"
    add "linalggen/generate.py --check" "python3 tools/linalggen/generate.py --check"
fi

# The Cairo steps that apply, and the packages they concern.
PREPUSH_DO_BUILD=0
PREPUSH_DO_GAS=0
PREPUSH_GAS_FULL=0
if has_code '^(crates/|tests/|(.*/)?Scarb\.toml$|(.*/)?Scarb\.lock$|\.tool-versions$)'; then PREPUSH_DO_BUILD=1; fi
if has_code '^(crates/|tests/|gas/|(.*/)?Scarb\.toml$|(.*/)?Scarb\.lock$|\.tool-versions$|scripts/gas_report\.py$)'; then PREPUSH_DO_GAS=1; fi
if has_code '^(gas/|scripts/gas_report\.py$)'; then PREPUSH_GAS_FULL=1; fi
export PREPUSH_DO_BUILD PREPUSH_DO_GAS PREPUSH_GAS_FULL
PREPUSH_PKGS=""
if [[ "$PREPUSH_DO_BUILD" == 1 || "$PREPUSH_DO_GAS" == 1 ]]; then
    # The touched packages: each changed path is mapped to the workspace package whose directory holds it;
    # "*" (the whole workspace) for a root manifest, Scarb.lock or .tool-versions.
    step="map the changed paths to packages (scarb metadata)"
    PREPUSH_PKGS=$(scarb metadata --format-version 1 --no-deps | PREPUSH_CHANGED="$changed" python3 -c '
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
    if [[ -z "$PREPUSH_PKGS" && "$PREPUSH_GAS_FULL" == 0 ]]; then
        # no workspace package touched (e.g. a manifest of benchmarks/ or tools/, not a member): nothing to compile
        PREPUSH_DO_BUILD=0
        PREPUSH_DO_GAS=0
        export PREPUSH_DO_BUILD PREPUSH_DO_GAS
        echo "prepush: touched packages: none (no workspace package is concerned)"
    else
        echo "prepush: touched packages: ${PREPUSH_PKGS:-none (gas/ or gas_report.py only: the whole workspace is checked)}"
    fi
fi
export PREPUSH_PKGS

logs=$(mktemp -d)
gate="$logs/gate"
mark="$logs/mark"
set_gate() { echo "$1" > "$gate.tmp" && mv -f "$gate.tmp" "$gate"; }
# on any exit the lock waiter, if it holds the lock idle, is told to stop (and stops anyway when this script is gone)
# Interrupted (INT, TERM, HUP): exit through the EXIT trap below, which stops this script's own background jobs.
trap 'step=interrupted; exit 130' INT
trap 'step=interrupted; exit 143' TERM
trap 'step=interrupted; exit 129' HUP
# Each background job is its own process group (job control on while it is launched), so that the EXIT trap can
# stop the whole tree of a job THIS script started, and nothing else: `kill -TERM -- -<pgid>`.
waiter=""
pids=()
# Is process <pid> still a child of this script (a reaped id, or a recycled one, is not)? Field 4 of /proc/<pid>/stat
# is the parent pid; the command name (field 2) may hold spaces, so it is cut off at its last ")" first.
is_my_child() {
    local stat rest
    { stat=$(< "/proc/$1/stat"); } 2> /dev/null || return 1
    rest=${stat##*) } # "<state> <ppid> ..."
    rest=${rest#* }
    [[ "${rest%% *}" == "$$" ]]
}
stop_jobs() {
    local g
    for g in $waiter ${pids[@]+"${pids[@]}"}; do
        # a cleared id (job ended and waited) is skipped, and so is one that is no longer our child
        if [[ -n "$g" ]] && is_my_child "$g"; then kill -TERM -- "-$g" 2> /dev/null || true; fi
    done
    return 0
}
trap 'rc=$?; [[ -e "$gate" ]] || set_gate stop 2> /dev/null || true; stop_jobs; rm -rf "$logs"; if [[ $rc -ne 0 ]]; then echo "prepush: FAILED at step: ${step} ($((SECONDS - start))s)" >&2; fi' EXIT

# The lock wait starts now, before the fixed checks, when the Cairo block will run under the lock.
use_lock=0
if [[ "$PREPUSH_DO_BUILD" == 1 || "$PREPUSH_DO_GAS" == 1 ]]; then
    if [[ -z "${HEAVY_BUILD_LOCK_HELD:-}" ]] && ! ancestor_holds_lock && command -v flock > /dev/null && [[ -d "$(dirname "$lock")" ]]; then
        use_lock=1
        # flock exits 75 when the lock is not obtained in 90 s; its exit status lands in $logs/flock.rc. PREPUSH_MARK
        # appears only once the lock is held: a failed step is told apart from a lock that was not obtained and
        # from any other flock error.
        set -m
        (
            rc=0
            PREPUSH_INNER=1 PREPUSH_MARK="$mark" PREPUSH_GATE="$gate" PREPUSH_OUT="$logs/cairo.log" PREPUSH_PARENT=$$ \
                PREPUSH_T0=$(date +%s) flock -E 75 -w 90 "$lock" "$self" "$base" 2> "$logs/flock.err" || rc=$?
            echo "$rc" > "$logs/flock.rc.tmp" && mv -f "$logs/flock.rc.tmp" "$logs/flock.rc"
        ) > /dev/null 2>&1 < /dev/null &
        waiter=$!
        set +m
    fi
fi

# The fixed checks run side by side to keep the wall time down; they are judged in order, the first failing one is named.
if [[ ${#cmds[@]} -eq 0 ]]; then
    echo "prepush: fixed checks skipped (no changed path feeds one of them)"
else
    set -m
    for i in "${!cmds[@]}"; do
        ( t=$SECONDS; ${cmds[$i]} > "$logs/$i.log" 2>&1; rc=$?; echo "prepush: ${names[$i]}: $((SECONDS - t))s" >> "$logs/$i.log"; exit $rc ) &
        pids+=($!)
    done
    set +m
    failed=""
    for i in "${!cmds[@]}"; do
        if wait "${pids[$i]}"; then rc=0; else rc=$?; fi
        pids[$i]=""
        echo "prepush: ${names[$i]}"
        cat "$logs/$i.log"
        if [[ $rc -ne 0 && -z "$failed" ]]; then failed="${names[$i]}"; fi
    done
    if [[ -n "$failed" ]]; then step="$failed"; exit 1; fi
fi

if [[ "$PREPUSH_DO_BUILD" == 0 && "$PREPUSH_DO_GAS" == 0 ]]; then
    echo "prepush: Cairo compile skipped (no workspace Cairo source, manifest or gas input changed against $base)"
elif [[ $use_lock == 1 ]]; then
    step="Cairo compile (the failing step is named above)"
    # the block started as soon as the lock was obtained; print its output as it grows, until the waiter ends
    off=0
    flush() {
        local size
        size=$(wc -c 2> /dev/null < "$logs/cairo.log" || echo 0)
        if ((size > off)); then tail -c +$((off + 1)) "$logs/cairo.log" | head -c $((size - off)) || true; off=$size; fi
    }
    while [[ ! -e "$logs/flock.rc" ]]; do flush; sleep 0.5; done
    flush
    waiter="" # ended by itself
    if [[ -e "$mark" ]]; then
        [[ "$(< "$logs/flock.rc")" -eq 0 ]] || exit 1 # a step failed inside the lock; it printed its own message
    else
        rc=$(< "$logs/flock.rc")
        if [[ $rc -eq 75 ]]; then
            echo "heavy lock busy: Cairo compile left to CI"
        elif [[ $rc -ne 0 ]]; then
            echo "prepush: flock failed on $lock (exit $rc): $(cat "$logs/flock.err" 2> /dev/null)" >&2
            exit 1
        fi
    fi
else
    cairo_block
fi

echo "prepush: ok in $((SECONDS - start))s"
