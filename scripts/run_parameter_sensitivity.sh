#!/usr/bin/env bash
# RQ3 parameter sensitivity on Logic (Expert-Log):
#   (1) c_base sweep -> ACD (Fig.10 left)
#   (2) capacity sweep: nlh expand {1,2,4,8} vs mamba2 d_state {16,32,64,128}
#
#   cd NL-H-H-SSM
#   export CUDA_VISIBLE_DEVICES=0
#   sed -i 's/\r$//' scripts/run_parameter_sensitivity.sh
#   EPOCHS=40 bash scripts/run_parameter_sensitivity.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

EPOCHS="${EPOCHS:-40}"
NLH_HP="${NLH_HP:-configs/nlh_tuned.yaml}"
DATA_DIR="${DATA_DIR:-data/processed}"
CSV="${DATA_DIR}/logic.csv"
OUT_DIR="benchmarks/sensitivity_results"
mkdir -p "${OUT_DIR}"
BASELINE_LR="${BASELINE_LR:-0.001}"

read -r SL ST <<<"$(python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${NLH_HP}").read_text(encoding="utf-8")) or {}
d = dict(cfg.get("defaults") or {})
d.update((cfg.get("overrides") or {}).get("logic", {}) or {})
print(int(d["seq_len"]), int(d.get("stride", d["seq_len"])))
PY
)"

echo "=== (1) c_base sweep on Logic (ACD, lower is better) ==="
for C in 0.1 0.5 1.0 1.5 2.0; do
  echo "--- c_base=${C} ---"
  python experiments/run_exp.py \
    --model nlh_ssm \
    --dataset "${CSV}" \
    --nlh_hparams_file "${NLH_HP}" \
    --epochs "${EPOCHS}" \
    --force_nlh_c_base "${C}"
  cp "results/logic_nlh_ssm.json" "${OUT_DIR}/logic_cbase_${C}.json"
done

echo ""
echo "=== (2) capacity sweep on Logic (NL-H expand vs Mamba-2 d_state) ==="
for DS in 16 32 64 128; do
  EX=$((DS / 16))
  echo "--- nlh expand=${EX} (label d_state=${DS}) ---"
  python experiments/run_exp.py \
    --model nlh_ssm \
    --dataset "${CSV}" \
    --nlh_hparams_file "${NLH_HP}" \
    --epochs "${EPOCHS}" \
    --force_nlh_expand "${EX}"
  cp "results/logic_nlh_ssm.json" "${OUT_DIR}/logic_expand_${EX}.json"

  echo "--- mamba2 d_state=${DS} ---"
  python experiments/run_exp.py \
    --model mamba2 \
    --dataset "${CSV}" \
    --seq_len "${SL}" \
    --stride "${ST}" \
    --epochs "${EPOCHS}" \
    --lr "${BASELINE_LR}" \
    --mamba_d_state "${DS}"
  cp "results/logic_mamba2.json" "${OUT_DIR}/logic_mamba2_dstate_${DS}.json"
done

echo ""
echo "Aggregate:"
echo "  python scripts/aggregate_parameter_sensitivity.py"
echo "  python benchmarks/parameter_sensitivity_figure10.py --from-json benchmarks/figure10_parameter_sensitivity.json"
