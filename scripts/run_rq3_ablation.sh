#!/usr/bin/env bash
# RQ3 component ablation: Logic + M5 x (full + wo_hyp + wo_acg + wo_ph_scan).
#
#   cd NL-H-H-SSM
#   export CUDA_VISIBLE_DEVICES=0
#   sed -i 's/\r$//' scripts/run_rq3_ablation.sh
#   EPOCHS=40 bash scripts/run_rq3_ablation.sh
#
# Outputs: results/logic_nlh_ssm.json (full, skip if exists unless RERUN_FULL=1)
#          results/logic_nlh_ssm_wo_hyp.json, ...
#          results/m5_nlh_ssm_wo_acg.json, ...
#
# After runs:
#   python scripts/aggregate_component_ablation.py
#   python benchmarks/model_ablation_figure9.py --from-json benchmarks/ablation_model_components.json

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

EPOCHS="${EPOCHS:-40}"
NLH_HP="${NLH_HP:-configs/nlh_tuned.yaml}"
DATA_DIR="${DATA_DIR:-data/processed}"
RERUN_FULL="${RERUN_FULL:-0}"
STEMS=(logic m5)
ABLATIONS=("" wo_hyp wo_acg wo_ph_scan)

run_one() {
  local stem="$1"
  local abl="$2"
  local csv="${DATA_DIR}/${stem}.csv"
  local tag="nlh_ssm"
  [[ -n "${abl}" ]] && tag="nlh_ssm_${abl}"
  local out="results/${stem}_${tag}.json"
  if [[ -z "${abl}" && "${RERUN_FULL}" != "1" && -f "${out}" ]]; then
    echo "Skip existing full model: ${out}"
    return 0
  fi
  echo ""
  echo "========== ${stem} | ablation=${abl:-full} | epochs=${EPOCHS} =========="
  local extra=()
  [[ -n "${abl}" ]] && extra+=(--nlh_ablation "${abl}")
  python experiments/run_exp.py \
    --model nlh_ssm \
    --dataset "${csv}" \
    --nlh_hparams_file "${NLH_HP}" \
    --epochs "${EPOCHS}" \
    "${extra[@]}"
}

for stem in "${STEMS[@]}"; do
  for abl in "${ABLATIONS[@]}"; do
    run_one "${stem}" "${abl}"
  done
done

echo ""
echo "Done. Aggregate + plot:"
echo "  python scripts/aggregate_component_ablation.py"
echo "  python benchmarks/model_ablation_figure9.py --from-json benchmarks/ablation_model_components.json"
