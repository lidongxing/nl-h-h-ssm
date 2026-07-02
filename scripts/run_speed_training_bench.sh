#!/usr/bin/env bash
# Training-step throughput (forward + backward + AdamW).
#
#   cd NL-H-H-SSM
#   export CUDA_VISIBLE_DEVICES=0
#   bash scripts/run_speed_training_bench.sh
#
# Saves: benchmarks/speed_results/speed_bench_train.json
# Does NOT overwrite forward-only speed_bench.json or Figure 7.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python benchmarks/speed_test.py \
  --json-out benchmarks/speed_results/speed_bench_train.json \
  --no-plot
echo "Saved benchmarks/speed_results/speed_bench_train.json"
