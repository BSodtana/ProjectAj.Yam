#!/usr/bin/env bash
set -euo pipefail

# Baseline V4 defaults
DATA_DIR="${DATA_DIR:-./data}"
BASE_OUT="${BASE_OUT:-./outputs/baseline_v4}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-cold_drug}"   # cold_drug=v3, cold_drug_v2=legacy
COLD_K="${COLD_K:-5}"
EPOCHS="${EPOCHS:-20}"
LIMIT="${LIMIT:-0}"                          # 0 => full data (do not pass --limit)
DEVICE="${DEVICE:-auto}"
DRW_RATIO="${DRW_RATIO:-0.7}"
DRW_LR_DROP="${DRW_LR_DROP:-0.2}"

# Override with env var, e.g. TAUS="0.5 1.5" or TAUS="0.5"
TAUS_STR="${TAUS:-0.5}"
read -r -a TAU_LIST <<< "${TAUS_STR}"

for TAU in "${TAU_LIST[@]}"; do
  TAU_TAG="${TAU/./p}"
  DRW_START_EPOCH="$(python3 - <<PY
import math
e = int(${EPOCHS})
r = float("${DRW_RATIO}")
print(math.floor(r * e) + 1)
PY
)"
  for ECFP in 0 1; do
    for COLD_FOLD in $(seq 0 $((COLD_K - 1))); do
      OUT_DIR="${BASE_OUT}/la_tau_${TAU_TAG}_drw${DRW_RATIO}_ecfp${ECFP}_k${COLD_K}_f${COLD_FOLD}_e${EPOCHS}_l${LIMIT}"

      TRAIN_CMD=(
        python3 scripts/train.py
        --data_dir "${DATA_DIR}"
        --output_dir "${OUT_DIR}"
        --split_strategy "${SPLIT_STRATEGY}"
        --cold_k "${COLD_K}"
        --cold_fold "${COLD_FOLD}"
        --logit_adjust_tau "${TAU}"
        --epochs "${EPOCHS}"
        --enable_drw
        --drw_start_epoch "${DRW_START_EPOCH}"
        --drw_lr_drop "${DRW_LR_DROP}"
        --class_weight_method inv_sqrt
        --class_weight_normalize sample_mean
        --class_weight_clip_min 0.25
        --class_weight_clip_max 4.0
        --class_weight_eps 1e-12
        --device "${DEVICE}"
      )
      if [ "${ECFP}" -eq 1 ]; then
        TRAIN_CMD+=(--use_ecfp_features)
      fi
      if [ "${LIMIT}" -gt 0 ]; then
        TRAIN_CMD+=(--limit "${LIMIT}")
      fi

      echo "[Baseline V4][train] tau=${TAU} out=${OUT_DIR} strategy=${SPLIT_STRATEGY} k=${COLD_K} fold=${COLD_FOLD} limit=${LIMIT} ecfp=${ECFP} drw_start_epoch=${DRW_START_EPOCH}"
      "${TRAIN_CMD[@]}"

      EVAL_CMD=(
        python3 scripts/evaluate.py
        --data_dir "${DATA_DIR}"
        --output_dir "${OUT_DIR}"
        --checkpoint "${OUT_DIR}/checkpoints/best.pt"
        --split_strategy "${SPLIT_STRATEGY}"
        --cold_k "${COLD_K}"
        --cold_fold "${COLD_FOLD}"
        --device "${DEVICE}"
      )
      if [ "${LIMIT}" -gt 0 ]; then
        EVAL_CMD+=(--limit "${LIMIT}")
      fi

      echo "[Baseline V4][eval] tau=${TAU} out=${OUT_DIR} strategy=${SPLIT_STRATEGY} k=${COLD_K} fold=${COLD_FOLD} limit=${LIMIT} ecfp=${ECFP}"
      "${EVAL_CMD[@]}"
    done
  done
done

echo "Baseline V4 complete. Metrics files:"
for TAU in "${TAU_LIST[@]}"; do
  TAU_TAG="${TAU/./p}"
  for ECFP in 0 1; do
    for COLD_FOLD in $(seq 0 $((COLD_K - 1))); do
      echo "  ${BASE_OUT}/la_tau_${TAU_TAG}_drw${DRW_RATIO}_ecfp${ECFP}_k${COLD_K}_f${COLD_FOLD}_e${EPOCHS}_l${LIMIT}/evaluation_metrics.json"
    done
  done
done
