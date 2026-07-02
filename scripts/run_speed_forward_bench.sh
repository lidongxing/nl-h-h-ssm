#!/usr/bin/env bash
# Forward-only throughput + Figure 7 (for §6.2 Table / Fig. scalability).
#
#   cd NL-H-H-SSM
#   export CUDA_VISIBLE_DEVICES=0
#   bash scripts/run_speed_forward_bench.sh
#
# Saves: benchmarks/speed_results/speed_bench.json
#        assets/figure7_speed_scaling.png

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python benchmarks/speed_test.py --forward-only
echo "Saved benchmarks/speed_results/speed_bench.json and assets/figure7_speed_scaling.png"
