# WP 10-PP — nalgebra-cairo: a pre-push check, its hook, and retries on tool downloads

Lot of the nalgebra track of Slingfall (orchestrator: herdr project `slingfall-nalgebra`), 2026-10-02.
Owner's request "stop pushing red CI": about 15 of 22 red CI runs across the organisation since
2026-10-01 would have been stopped by a local check of seconds to minutes (formatting, generated
artefacts not regenerated, script unit tests, compile errors); 6 were infrastructure failures (HTTP
500 or DNS while downloading scarb or snforge). simba-cairo's equivalent is merged (simba-cairo #5:
read its `scripts/prepush.sh` and `.githooks/pre-push` on simba's `main` as the model, at
`/home/claude/projects/simba-cairo`, read-only). Runs on the VPS.

## 1. Goal

1. `scripts/prepush.sh`, aim **under 2 minutes** when the heavy-build lock is free:
   - always: `scarb fmt --check`; the self-tests of the Python scripts (`scripts/consumer_cost.py
     --self-test`, `scripts/packages_table.py --self-test`, `scripts/release.py --self-test`, and any
     other script under `scripts/` or `tools/` that has a self-test: list what you found); the
     generated-artefact checks that need no build (`python3 scripts/api_parity.py --check`,
     `python3 tools/shapegen/shapegen.py --check`, `python3 tools/linalggen/generate.py --check`,
     as `scripts/check.sh` runs them);
   - only when a Cairo source or manifest changed against the push's base (`crates/**`, `tests/**`,
     any `Scarb.toml`, `Scarb.lock`, `.tool-versions`): `scarb lint --deny-warnings` and `scarb build`
     **of the touched packages only** (map each changed path to its package with `scarb metadata`;
     a change to a root manifest or `.tool-versions` means the whole workspace); then the gas
     check of the touched test packages (`snforge test -p <pkg>` piped into
     `python3 scripts/gas_report.py --check gas/` as CI does per shard — read ci.yml and
     gas_report.py for the exact form) when `crates/**`, `tests/**`, `gas/**`, manifests or
     `scripts/gas_report.py` changed;
   - `export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}` (gas is compared);
   - the header states which inputs trigger which check; exit non-zero on the first failure with a
     one-line message naming the step; print the elapsed time at the end. `scripts/check.sh` stays
     the full local equivalent of CI, unchanged.
2. **The heavy-build lock (programme rule):** the compile block runs under the host's lock held by
   the script itself: `flock -E 75 -w 90 ~/orchestrator/heavy-build.lock <command>`, with
   `HEAVY_BUILD_LOCK_HELD=1` exported inside so the scarb/snforge shims do not wait a second time
   (read the shims: `which -a scarb snforge`). Nothing is ever killed (no kill, timeout, pkill); the
   lock file is touched only through `flock -w`. On exit 75 print exactly
   `heavy lock busy: Cairo compile left to CI` and pass. Any other flock failure is an error. On a
   machine without the lock directory (the Mac), the compile always runs.
3. **Git environment (urgent programme rule):** in `prepush.sh`, the hook, and every test or
   self-test they run, any git command that is not a read of the repository being pushed runs only
   in a sanitised environment: `env -u GIT_DIR -u GIT_WORK_TREE -u GIT_INDEX_FILE -u GIT_COMMON_DIR
   -u GIT_PREFIX …` or `env -i`. That covers `git init`, `commit`, `config` in a temp dir. Check
   each self-test you call: if one runs git outside a pure read, report it and wrap it.
4. `.githooks/pre-push` (executable): as simba's: deletion lines skipped (`continue`), tags compared
   by their peeled commit (`git rev-parse --verify --quiet "$sha^{commit}"`), refuse when a pushed
   branch commit is not HEAD or tracked files differ from HEAD ("commit or stash, then push"), base =
   the remote sha of the single pushed ref when it exists locally, else `origin/main`;
   `git diff --name-only --no-renames`. **Never run `git config core.hooksPath`**: the owner sets it.
5. `AGENTS.md`, Definition of done / workflow: run `scripts/prepush.sh` (the hook does it) before
   every push; never push red; never `--no-verify`; how a fresh clone enables the hook.
6. Workflows (**the files this lot changes: `.github/workflows/ci.yml` and
   `.github/workflows/split-measure.yml`**): one retry on every `software-mansion/setup-scarb` and
   `foundry-rs/setup-snfoundry` step, as simba-cairo's `ci.yml` does (the step with an `id` and
   `continue-on-error: true`, then the same step again with `if: steps.<id>.outcome == 'failure'`;
   same action pins, same inputs, `tool-versions` included where a step has it); and in `ci.yml` the
   existing `concurrency:` block cancels only superseded pull-request runs:
   `cancel-in-progress: ${{ github.event_name == 'pull_request' }}` (group unchanged). Nothing else
   in the workflows changes; no job is skipped for docs-only PRs.

## 2. Scope: file allowlist

This brief committed verbatim as `docs/briefs/wp-10-prepush.md` (first commit); `scripts/prepush.sh`
(new); `.githooks/pre-push` (new); `AGENTS.md` (the lines above); `.github/workflows/ci.yml` and
`split-measure.yml` (retries and the concurrency line only). Nothing else.

## 3. Acceptance criteria

1. Measured times (real `time` output, VPS), each naming the head it was measured at and the lock
   wait separately: (a) only a document changed; (b) one Cairo file changed in a small package, lock
   free (or, if the shared lock is never free for 30 minutes, the same run on a private
   `HEAVY_BUILD_LOCK=$(mktemp)` file, said so); (c) the same with the lock busy (90 s, the skip line).
   In the PR body and the report.
2. The hook blocks a failing script: shown on a throwaway local commit that breaks formatting (run
   the hook directly; drop the commit; never push it).
3. CI green; the retry steps present and not triggered on a normal run.
4. Every git command of the hook, the script and the self-tests it runs listed in the report as
   "read of the pushed repo" or "sanitised env".

## 4. Verification and report

Conventional commits; push through the hook with `git -c core.hooksPath=.githooks push` (no config
written); **all fixes of one review loop in one push**. `gh pr create` per the template (body: what
each part does, the measured times, "Audit: none needed — local tooling and CI retries"). CI: at most
one GitHub call per PR every 5 minutes (`gh pr checks <n> --watch --interval 300`), or end your turn.
Never merge, never launch a review or any agent yourself. Foreground only. `REPORT.md` at the
worktree root (not committed), `## Summary` first. Prefer words over links to external issues in PR
text and commit messages.

Work autonomously, do not ask questions, do not widen the scope.

