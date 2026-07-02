#!/usr/bin/env bash
# Table 6 full benchmark: 10 datasets x (4 baselines + nlh_ssm).
# Same seq_len/stride per dataset from configs/nlh_tuned.yaml (fair comparison).
#
#   cd NL-H-H-SSM
#   export CUDA_VISIBLE_DEVICES=0
#   sed -i 's/\r$//' scripts/run_table6_all.sh   # once on Linux if copied from Windows
#   EPOCHS=40 bash scripts/run_table6_all.sh
#
# Subcommands:
#   all        (default) 10 datasets x 5 models
#   nlh_only             only nlh_ssm
#   baselines_only       transformer informer mamba1 mamba2
#   rank                 python scripts/rank_table6.py
#   tex                  generate table6_harv_fragment.tex
#
# Env: EPOCHS=40  NLH_YAML=configs/nlh_tuned.yaml  DATA_DIR=data/processed
#      BASELINE_LR=0.001  MODELS="transformer informer mamba1 mamba2"

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

YAML="${NLH_YAML:-configs/nlh_tuned.yaml}"
DATA_DIR="${DATA_DIR:-data/processed}"
EPOCHS="${EPOCHS:-40}"
BASELINE_LR="${BASELINE_LR:-0.001}"
NLH_HP="${NLH_HP:-configs/nlh_tuned.yaml}"
BASELINES="${MODELS:-transformer informer mamba1 mamba2}"

# Table 6 order (must match generate_table_6.py)
STEMS=(tourism labor prison m5 wiki electricity traffic solar logic medical)

read_windows() {
  local stem="$1"
  python - <<PY
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${YAML}").read_text(encoding="utf-8")) or {}
d = dict(cfg.get("defaults") or {})
d.update((cfg.get("overrides") or {}).get("${stem}", {}) or {})
sl = d.get("seq_len")
st = d.get("stride", sl)
if sl is None:
    sys.exit(1)
print(int(sl), int(st if st is not None else sl))
PY
}

run_one() {
  local model="$1" stem="$2" sl="$3" st="$4"
  local csv="${DATA_DIR}/${stem}.csv"
  echo ""
  echo "========== ${stem} | ${model} | seq_len=${sl} stride=${st} | epochs=${EPOCHS} =========="
  if [[ "${model}" == "nlh_ssm" ]]; then
    python experiments/run_exp.py \
      --model nlh_ssm \
      --dataset "${csv}" \
      --nlh_hparams_file "${NLH_HP}" \
      --epochs "${EPOCHS}"
  else
    python experiments/run_exp.py \
      --model "${model}" \
      --dataset "${csv}" \
      --seq_len "${sl}" \
      --stride "${st}" \
      --epochs "${EPOCHS}" \
      --lr "${BASELINE_LR}"
  fi
}

run_models() {
  local which="$1" # all | nlh | baselines
  local n_stems=${#STEMS[@]}
  local n_models=0
  [[ "${which}" == "all" || "${which}" == "baselines" ]] && n_models=$((n_models + 4))
  [[ "${which}" == "all" || "${which}" == "nlh" ]] && n_models=$((n_models + 1))
  local total=$((n_stems * n_models))
  local i=0

  echo "[table6] ${which}: ${n_stems} datasets x ${n_models} models = ${total} runs"
  echo "[table6] windows from ${YAML}"
  echo "[table6] GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"

  for stem in "${STEMS[@]}"; do
    [[ -f "${DATA_DIR}/${stem}.csv" ]] || { echo "[skip] missing ${DATA_DIR}/${stem}.csv"; continue; }
    read -r SL ST < <(read_windows "${stem}") || {
      echo "[skip] ${stem}: no seq_len/stride in ${YAML}" >&2
      continue
    }
    if [[ "${which}" == "all" || "${which}" == "baselines" ]]; then
      for model in ${BASELINES}; do
        i=$((i + 1))
        echo "[${i}/${total}]"
        run_one "${model}" "${stem}" "${SL}" "${ST}"
      done
    fi
    if [[ "${which}" == "all" || "${which}" == "nlh" ]]; then
      i=$((i + 1))
      echo "[${i}/${total}]"
      run_one "nlh_ssm" "${stem}" "${SL}" "${ST}"
    fi
  done
}

cmd="${1:-all}"
case "${cmd}" in
  all) run_models all ;;
  nlh_only|nlh) run_models nlh ;;
  baselines_only|baselines) run_models baselines ;;
  rank) python scripts/rank_table6.py --results_dir results ;;
  tex)
    python scripts/generate_table_6.py --results_dir results --format harv \
      --impr_ref mamba2 --out table6_harv_fragment.tex
    echo "Wrote table6_harv_fragment.tex — paste into elsarticle-template-harv.tex (tab:main_results)"
    ;;
  help|-h)
    sed -n '2,20p' "$0"
    ;;
  *)
    echo "Unknown command: ${cmd}  (try: all | nlh_only | baselines_only | rank | tex | help)" >&2
    exit 1
    ;;
esac

echo ""
echo "[table6] finished. Next: EPOCHS=${EPOCHS} bash scripts/run_table6_all.sh rank"
