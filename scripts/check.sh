#!/usr/bin/env bash
# Local equivalent of the CI gate for the library workspace. Pass `--update` to refresh the snapshot.
set -euo pipefail
cd "$(dirname "$0")/.."

scarb fmt --check
python3 scripts/api_parity.py --check
python3 tools/shapegen/shapegen.py --check
python3 crates/nalgebra/src/linalg/generate.py --check
scarb lint --deny-warnings
scarb build
output=$(snforge test --workspace) || { echo "$output"; exit 1; }
echo "$output" | tail -n 1
if [[ "${1:-}" == "--update" ]]; then
    echo "$output" | python3 scripts/gas_report.py --update gas/
else
    echo "$output" | python3 scripts/gas_report.py --check gas/
fi
