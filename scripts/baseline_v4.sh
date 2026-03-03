#!/usr/bin/env bash
set -euo pipefail

# Baseline V4 defaults
DATA_DIR="${DATA_DIR:-./data}"
BASE_OUT="${BASE_OUT:-./outputs/baseline_v4}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-cold_drug}"   # cold_drug=v3, cold_drug_v2=legacy
EPOCHS="${EPOCHS:-15}"
LIMIT="${LIMIT:-50000}"
DEVICE="${DEVICE:-auto}"

TAUS=(0.5 1.5)

for TAU in "${TAUS[@]}"; do
  TAU_TAG="${TAU/./p}"
  OUT_DIR="${BASE_OUT}/la_tau_${TAU_TAG}_e${EPOCHS}_l${LIMIT}"

  echo "[Baseline V4][train] tau=${TAU} out=${OUT_DIR} strategy=${SPLIT_STRATEGY}"
  python scripts/train.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUT_DIR}" \
    --split_strategy "${SPLIT_STRATEGY}" \
    --logit_adjust_tau "${TAU}" \
    --epochs "${EPOCHS}" \
    --limit "${LIMIT}" \
    --device "${DEVICE}"

  echo "[Baseline V4][eval] tau=${TAU} out=${OUT_DIR} strategy=${SPLIT_STRATEGY}"
  python scripts/evaluate.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUT_DIR}" \
    --checkpoint "${OUT_DIR}/checkpoints/best.pt" \
    --split_strategy "${SPLIT_STRATEGY}" \
    --limit "${LIMIT}" \
    --device "${DEVICE}"
done

echo "Baseline V4 complete. Metrics files:"
echo "  ${BASE_OUT}/la_tau_0p5_e${EPOCHS}_l${LIMIT}/evaluation_metrics.json"
echo "  ${BASE_OUT}/la_tau_1p5_e${EPOCHS}_l${LIMIT}/evaluation_metrics.json"
