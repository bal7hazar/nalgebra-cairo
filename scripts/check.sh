#!/usr/bin/env bash
# Local equivalent of the CI gate for the library workspace. Pass `--update` to refresh the snapshot.
# Requires python3 with mpmath (`pip install mpmath`) for the generated-code freshness checks.
set -euo pipefail
cd "$(dirname "$0")/.."

scarb fmt --check
python3 tools/polygen/polygen.py --check
python3 tools/fixed_model/gen_vectors.py --check
scarb lint --deny-warnings
scarb build
output=$(snforge test --workspace) || { echo "$output"; exit 1; }
echo "$output" | tail -n 1
if [[ "${1:-}" == "--update" ]]; then
    echo "$output" | python3 scripts/gas_report.py --update gas/
else
    echo "$output" | python3 scripts/gas_report.py --check gas/
fi
