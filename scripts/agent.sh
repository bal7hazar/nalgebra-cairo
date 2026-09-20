#!/usr/bin/env bash
# Orchestrator helper (docs/ORCHESTRATOR.md): run one work package with a local CLI in a
# dedicated git worktree, in the background, output redirected to a log file.
#
#   scripts/agent.sh <worktree> <claude|codex> <model> <brief.md> <log> [--resume "<follow-up>"]
#
# claude: `--model` is sonnet | opus | fable; codex: `<model>` may carry an effort suffix, e.g.
# `gpt-5.6:high` (default effort: high). The agent writes REPORT.md (git-ignored) at the worktree
# root; the orchestrator reads that file and the log, never the transcript.
set -euo pipefail
worktree=$1; cli=$2; model=$3; brief=$4; log=$5; shift 5
resume=""
if [[ "${1:-}" == "--resume" ]]; then resume=$2; fi
cd "$worktree"
task=$(basename "$worktree")

framing='You are a sub-agent of an orchestrator, executing ONE work package of nalgebra.cairo in the git worktree you are started in (see docs/ORCHESTRATOR.md for the process, AGENTS.md for the rules). Hard rules:
- Stay strictly inside the scope and file allowlist of the brief. Shared files (workspace Scarb.toml, lib.cairo beyond adding your own lines where the brief allows it, .github, scripts, docs, gas snapshots of other modules) belong to the orchestrator: list what you need from them in the "Escalations" section of REPORT.md instead of editing them.
- Do not ask questions: decide, document the decision in doc comments, report it. Do not widen the scope. Do not stop before the deliverables are complete.
- Work incrementally: compile early and often, keep changes small, never delete tests to make the gate pass. Watch the compile budget of test files (the first cause of CI failures).
- Efficiency: no loops in static code, every sum of products through a fused `Real` kernel, `#[inline(always)]` on small ops only; when the cheapest formulation is not obvious, bench the variants (winner in the library, losers under `#[cfg(test)]` with their benches).
- Definition of done, in the FOREGROUND (never a background command followed by the end of your turn): `./scripts/check.sh --update` green; conventional commits ending with the line `Co-Authored-By: Claude <noreply@anthropic.com>`; `git push -u origin <branch>`; `gh pr create` following .github/PULL_REQUEST_TEMPLATE.md; `gh pr checks --watch` until green (fix and push again if red); NEVER merge; write REPORT.md at the worktree root with: summary, API, gas table, deviations, deferred items, requested re-exports, escalations, PR URL. Never switch branches, never stash or reset, never touch other worktrees.'

case "$cli" in
  claude)
    if [[ -n "$resume" ]]; then
      exec claude --continue -p "$resume" --dangerously-skip-permissions > "$log" 2>&1
    fi
    exec claude -p "$(cat "$brief")" \
      --model "$model" \
      --name "$task" \
      --dangerously-skip-permissions \
      --append-system-prompt "$framing" \
      > "$log" 2>&1
    ;;
  codex)
    effort="${model##*:}"; [[ "$effort" == "$model" ]] && effort=high
    model="${model%%:*}"
    prompt="$framing

$(cat "$brief")"
    if [[ -n "$resume" ]]; then
      exec codex exec -C "$worktree" resume --last "$resume" > "$log" 2>&1
    fi
    exec codex exec -C "$worktree" -m "$model" -c "model_reasoning_effort=$effort" \
      --dangerously-bypass-approvals-and-sandbox -o "$worktree/REPORT.md" "$prompt" > "$log" 2>&1
    ;;
  *) echo "unknown cli: $cli (claude|codex)" >&2; exit 2 ;;
esac
