#!/usr/bin/env bash
# Run all three RQ3 enhancement steps in order (GPU-heavy).
#
#   cd NL-H-H-SSM
#   export CUDA_VISIBLE_DEVICES=0
#   sed -i 's/\r$//' scripts/run_rq3_enhancements_all.sh
#   EPOCHS=40 bash scripts/run_rq3_enhancements_all.sh
#
# Steps:
#   1) Component ablation (Logic + M5)
#   2) Parameter sensitivity (Logic c_base + capacity)
#   3) Training-step speed benchmark (fwd+bwd)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

EPOCHS="${EPOCHS:-40}"
export EPOCHS

echo "========== Step 1/3: Component ablation =========="
bash scripts/run_rq3_ablation.sh
python scripts/aggregate_component_ablation.py
python benchmarks/model_ablation_figure9.py --from-json benchmarks/ablation_model_components.json

echo ""
echo "========== Step 2/3: Parameter sensitivity =========="
bash scripts/run_parameter_sensitivity.sh
python scripts/aggregate_parameter_sensitivity.py
python benchmarks/parameter_sensitivity_figure10.py --from-json benchmarks/figure10_parameter_sensitivity.json

echo ""
echo "========== Step 3/3: Training-step speed benchmark =========="
bash scripts/run_speed_training_bench.sh

echo ""
echo "All RQ3 enhancements complete."
echo "  Fig.9:  assets/figure9_component_ablation.png"
echo "  Fig.10: assets/figure10_parameter_sensitivity.png"
echo "  Train:  benchmarks/speed_results/speed_bench_train.json"
