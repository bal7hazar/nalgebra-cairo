#!/usr/bin/env bash
# Orchestrator helper: run a scoped sub-agent with the Claude CLI in a dedicated git worktree.
#
#   scripts/agent.sh <worktree-dir> <model> <prompt-file> <log-file>
#
# The agent gets the repository rules (AGENTS.md) plus a framing system prompt that keeps it inside
# its work package; edits are auto-accepted, shell commands are limited to the build/test toolchain
# and read-only git; pushing, switching branches or touching other worktrees is impossible.
set -euo pipefail
worktree=$1; model=$2; prompt_file=$3; log=$4
cd "$worktree"

framing='You are a sub-agent of an orchestrator, executing ONE work package of nalgebra.cairo in the git worktree you are started in. Hard rules, in addition to AGENTS.md:
- Stay strictly inside the scope and file ownership stated in the task. Never edit files outside it, never edit other worktrees, never change workspace manifests, `.github`, `scripts`, `docs` or `benchmarks` unless the task says so.
- Do not ask questions: decide, document the decision in code comments/doc comments, and report it at the end. Do not stop before the deliverables are complete and the gate (`./scripts/check.sh --update`) is green.
- Work incrementally: compile early and often (Cairo errors are frequent), keep changes small, never delete tests to make the gate pass.
- Efficiency: no loops in static code, every sum of products through a fused `Real` kernel, `#[inline(always)]` on small ops only, measure alternatives with gas benches when the cheapest implementation is not obvious.
- Git: only `git add` and `git commit` on the current branch (conventional commit message ending with the line `Co-Authored-By: Claude <noreply@anthropic.com>`). Never push, never switch branches, never stash or reset.
- Finish with a concise report (API summary, gas table, design decisions with measurements, open issues) as your final message.'

exec claude -p "$(cat "$prompt_file")" \
  --model "$model" \
  --permission-mode acceptEdits \
  --append-system-prompt "$framing" \
  --allowedTools "Bash(scarb:*)" "Bash(snforge:*)" "Bash(python3:*)" "Bash(python:*)" "Bash(cargo:*)" \
    "Bash(git add:*)" "Bash(git commit:*)" "Bash(git status:*)" "Bash(git diff:*)" "Bash(git log:*)" "Bash(git show:*)" \
    "Bash(ls:*)" "Bash(cat:*)" "Bash(head:*)" "Bash(tail:*)" "Bash(wc:*)" "Bash(grep:*)" "Bash(rg:*)" "Bash(find:*)" \
    "Bash(mkdir:*)" "Bash(cp:*)" "Bash(mv:*)" "Bash(rm:*)" "Bash(diff:*)" "Bash(sort:*)" "Bash(sed:*)" "Bash(awk:*)" "Bash(echo:*)" \
    "Bash(./scripts/check.sh:*)" "Bash(scripts/check.sh:*)" "Bash(asdf:*)" \
  --disallowedTools "Bash(git push:*)" "Bash(git checkout:*)" "Bash(git switch:*)" "Bash(git stash:*)" "Bash(git reset:*)" "Bash(git rebase:*)" "Bash(gh:*)" \
  > "$log" 2>&1
