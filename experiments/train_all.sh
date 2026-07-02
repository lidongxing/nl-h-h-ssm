#!/usr/bin/env bash
# Step 5 (skeleton): train NL-H-H-SSM on every YAML under experiments/configs/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
for cfg in experiments/configs/*.yaml; do
  echo "=== ${cfg} ==="
  python experiments/run_main.py --config "${cfg}"
done
