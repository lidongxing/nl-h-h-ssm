#!/usr/bin/env bash
# Table-6 campaign: per-dataset nlh_ssm tuning + training.
#
# Usage (repo root):
#   export CUDA_VISIBLE_DEVICES=0
#   bash scripts/table6_nlh_campaign.sh rank          # current standings
#   bash scripts/table6_nlh_campaign.sh tune_a        # phase A: high-priority tune
#   bash scripts/table6_nlh_campaign.sh train_a       # phase A: train from yaml
#   bash scripts/table6_nlh_campaign.sh tune_b        # phase B: harder datasets
#   bash scripts/table6_nlh_campaign.sh train_b
#   bash scripts/table6_nlh_campaign.sh tune_all     # full 10-dataset window search (slow)
#   bash scripts/run_table6_fair_baselines.sh        # baselines, same windows as nlh yaml
#
# After nlh train_a, re-run baselines: bash scripts/run_table6_fair_baselines.sh m5 wiki ...

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATA_DIR="${DATA_DIR:-data/processed}"
OUT_YAML="${OUT_YAML:-configs/nlh_tuned.yaml}"
EPOCHS_TUNE="${EPOCHS_TUNE:-15}"
EPOCHS_TRAIN="${EPOCHS_TRAIN:-40}"
MAX_WIN="${MAX_WIN:-4000}"

rank() {
  python scripts/rank_table6.py --results_dir results
}

tune_one() {
  local stem="$1"
  local extra="${2:-}"
  # shellcheck disable=SC2086
  python scripts/tune_nlh_ssm.py \
    --data_dir "${DATA_DIR}" \
    --datasets "${stem}" \
    --search_windows \
    --tune_epochs "${EPOCHS_TUNE}" \
    --max_windows "${MAX_WIN}" \
    --out "${OUT_YAML}" \
    --merge_existing \
    ${extra}
}

train_one() {
  local stem="$1"
  python experiments/run_exp.py \
    --model nlh_ssm \
    --dataset "${DATA_DIR}/${stem}.csv" \
    --nlh_hparams_file "${OUT_YAML}" \
    --epochs "${EPOCHS_TRAIN}"
}

# Tier A: already close or already #1 — best shot at more wins
TIER_A=(m5 logic medical wiki traffic solar)

tune_a() {
  echo "=== Phase A tune (search_windows): ${TIER_A[*]} ==="
  for s in "${TIER_A[@]}"; do
    echo "--- tune ${s} ---"
    tune_one "${s}"
  done
  rank
}

train_a() {
  echo "=== Phase A train (${EPOCHS_TRAIN} epochs, yaml overrides) ==="
  for s in "${TIER_A[@]}"; do
    echo "--- train ${s} ---"
    train_one "${s}"
  done
  rank
}

# Tier B: industrial / SE — harder
TIER_B=(electricity tourism labor prison)

tune_b() {
  echo "=== Phase B tune: ${TIER_B[*]} ==="
  for s in "${TIER_B[@]}"; do
    echo "--- tune ${s} ---"
    tune_one "${s}"
  done
  rank
}

train_b() {
  echo "=== Phase B train ==="
  for s in "${TIER_B[@]}"; do
    echo "--- train ${s} ---"
    train_one "${s}"
  done
  rank
}

tune_all() {
  python scripts/tune_nlh_ssm.py \
    --data_dir "${DATA_DIR}" \
    --search_windows \
    --tune_epochs "${EPOCHS_TUNE}" \
    --max_windows "${MAX_WIN}" \
    --out "${OUT_YAML}" \
    --merge_existing
  rank
}

train_all() {
  local stems=(tourism labor prison m5 wiki electricity traffic solar logic medical)
  for s in "${stems[@]}"; do
    [[ -f "${DATA_DIR}/${s}.csv" ]] || continue
    echo "--- train ${s} ---"
    train_one "${s}"
  done
  rank
}

# traffic: avoid seq_len=128 stride=1 (millions of windows). Re-tune after updating tune_nlh_ssm.py.
# Fair baseline (same window as nlh json):
#   SL=$(python -c "import json;print(json.load(open('results/wiki_nlh_ssm.json'))['hyperparams']['seq_len'])")
#   ST=$(python -c "import json;print(json.load(open('results/wiki_nlh_ssm.json'))['hyperparams']['stride'])")
#   python experiments/run_exp.py --model transformer --dataset data/processed/wiki.csv --seq_len $SL --stride $ST --epochs 30

case "${1:-help}" in
  rank) rank ;;
  tune_a) tune_a ;;
  train_a) train_a ;;
  tune_b) tune_b ;;
  train_b) train_b ;;
  tune_all) tune_all ;;
  train_all) train_all ;;
  *)
    cat <<'EOF'
Commands: rank | tune_a | train_a | tune_b | train_b | tune_all | train_all

Priority:
  A (tune_a/train_a): m5 logic medical wiki traffic solar
  B (tune_b/train_b): electricity tourism labor prison

Env: CUDA_VISIBLE_DEVICES, EPOCHS_TUNE=15, EPOCHS_TRAIN=40, MAX_WIN=4000
EOF
    ;;
esac
