#!/usr/bin/env bash
# Re-train Table-6 *baselines* with the same seq_len / stride / epochs as nlh_ssm
# (from configs/nlh_tuned.yaml). Does NOT use nlh_lr / nlh_c_base — those are nlh-only.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0
#   bash scripts/run_table6_fair_baselines.sh              # all stems in yaml
#   bash scripts/run_table6_fair_baselines.sh m5 wiki    # subset
#   MODELS="transformer mamba2" bash scripts/run_table6_fair_baselines.sh m5

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

YAML="${NLH_YAML:-configs/nlh_tuned.yaml}"
DATA_DIR="${DATA_DIR:-data/processed}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-0.001}"
MODELS="${MODELS:-transformer informer mamba1 mamba2}"

if [[ ! -f "${YAML}" ]]; then
  echo "Missing ${YAML}" >&2
  exit 1
fi

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

stems=("$@")
if [[ ${#stems[@]} -eq 0 ]]; then
  mapfile -t stems < <(python - <<PY
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("${YAML}").read_text(encoding="utf-8")) or {}
for k in (cfg.get("overrides") or {}):
    print(k)
PY
)
fi

for stem in "${stems[@]}"; do
  csv="${DATA_DIR}/${stem}.csv"
  [[ -f "${csv}" ]] || { echo "[skip] missing ${csv}"; continue; }
  read -r SL ST < <(read_windows "${stem}") || { echo "[skip] no seq_len in yaml for ${stem}"; continue; }
  echo "=== ${stem}  window seq_len=${SL} stride=${ST}  epochs=${EPOCHS} ==="
  for model in ${MODELS}; do
    echo "--- ${model} ---"
    python experiments/run_exp.py \
      --model "${model}" \
      --dataset "${csv}" \
      --seq_len "${SL}" \
      --stride "${ST}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}"
  done
done

echo "Done. Check: python scripts/rank_table6.py"
