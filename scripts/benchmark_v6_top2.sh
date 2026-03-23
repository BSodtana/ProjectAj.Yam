#!/usr/bin/env bash
set -euo pipefail

# Benchmark V6 top-2 single-fold configs.
# Default behavior is print-only so nothing executes unless you pass --run.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/tuning_baseline_v4_single_fold}"
SPLIT_CACHE_DIR="${SPLIT_CACHE_DIR:-${OUTPUT_ROOT}/split_cache}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SPLIT_SEED="${SPLIT_SEED:-42}"
COLD_K="${COLD_K:-5}"
COLD_FOLD="${COLD_FOLD:-0}"
COLD_PROTOCOL="${COLD_PROTOCOL:-s1}"
COLD_MIN_TEST_PAIRS="${COLD_MIN_TEST_PAIRS:-5000}"
COLD_MIN_TEST_LABELS="${COLD_MIN_TEST_LABELS:-45}"
COLD_MAX_RESAMPLES="${COLD_MAX_RESAMPLES:-200}"
COLD_DEDUPE_POLICY="${COLD_DEDUPE_POLICY:-keep_all}"
COLD_SELECTION_OBJECTIVE="${COLD_SELECTION_OBJECTIVE:-selected_fold}"

MODE="print"
if [[ "${1:-}" == "--run" ]]; then
  MODE="run"
elif [[ "${1:-}" == "--print-only" || -z "${1:-}" ]]; then
  MODE="print"
else
  echo "Usage: $0 [--print-only|--run]" >&2
  exit 1
fi

RUN_DIRS=(
  "${OUTPUT_ROOT}/runs/tau_0p5_lr_0p001_fold_0_drw_on_ratio_0p7_drop_0p2_start_15"
  "${OUTPUT_ROOT}/runs/tau_0p5_lr_0p001_fold_0_drw_on_ratio_0p8_drop_0p2_start_17"
)

build_eval_cmd() {
  local run_dir="$1"
  cat <<EOF
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/evaluate.py" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${run_dir}" \
  --split_cache_dir "${SPLIT_CACHE_DIR}" \
  --checkpoint "${run_dir}/checkpoints/best.pt" \
  --batch_size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --split_strategy cold_drug \
  --split_seed "${SPLIT_SEED}" \
  --cold_k "${COLD_K}" \
  --cold_fold "${COLD_FOLD}" \
  --cold_protocol "${COLD_PROTOCOL}" \
  --cold_min_test_pairs "${COLD_MIN_TEST_PAIRS}" \
  --cold_min_test_labels "${COLD_MIN_TEST_LABELS}" \
  --cold_max_resamples "${COLD_MAX_RESAMPLES}" \
  --cold_dedupe_policy "${COLD_DEDUPE_POLICY}" \
  --cold_selection_objective "${COLD_SELECTION_OBJECTIVE}" \
  --use_ecfp_features
EOF
}

for run_dir in "${RUN_DIRS[@]}"; do
  echo "[target] ${run_dir}"
  if [[ ! -f "${run_dir}/checkpoints/best.pt" ]]; then
    echo "Missing checkpoint: ${run_dir}/checkpoints/best.pt" >&2
    exit 1
  fi

  if [[ "${MODE}" == "print" ]]; then
    build_eval_cmd "${run_dir}"
    echo
    continue
  fi

  eval "$(build_eval_cmd "${run_dir}")"
  echo "[done] ${run_dir}/evaluation_metrics.json"
done
