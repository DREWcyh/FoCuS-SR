#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CKPT="${CKPT:-${1:-preset/models/focussr.pkl}}"
EXP_NAME="${EXP_NAME:-$(basename "${CKPT}" .pkl)}"
EVAL_NAME="${EVAL_NAME:-default}"
DATASET="${DATASET:-realsr}"

case "${DATASET}" in
  realsr)
    DEFAULT_INPUT="preset/testfolder/realsr/test_LR"
    DEFAULT_GT="preset/testfolder/realsr/test_HR"
    ;;
  drealsr)
    DEFAULT_INPUT="preset/testfolder/drealsr/test_LR"
    DEFAULT_GT="preset/testfolder/drealsr/test_HR"
    ;;
  div2k)
    DEFAULT_INPUT="preset/testfolder/div2k/lq"
    DEFAULT_GT="preset/testfolder/div2k/gt"
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}. Use realsr, drealsr, or div2k." >&2
    exit 1
    ;;
esac

INPUT="${INPUT:-${DEFAULT_INPUT}}"
GT="${GT:-${DEFAULT_GT}}"
BENCH_DIR="${BENCH_DIR:-experiments/${EXP_NAME}/eval/${DATASET}/${EVAL_NAME}}"
RESULT_DIR="${RESULT_DIR:-${BENCH_DIR}/sr}"
METRIC_DIR="${METRIC_DIR:-${BENCH_DIR}/metrics}"

PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-preset/models/stable-diffusion-2-1-base}"
PROCESS_SIZE="${PROCESS_SIZE:-512}"
UPSCALE="${UPSCALE:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
ALIGN_METHOD="${ALIGN_METHOD:-adain}"
LAMBDA_PIX="${LAMBDA_PIX:-1.0}"
LAMBDA_SEM="${LAMBDA_SEM:-1.0}"
RUN_INFER="${RUN_INFER:-1}"
RUN_METRICS="${RUN_METRICS:-1}"
RESUME_EXISTING="${RESUME_EXISTING:-1}"

has_pngs() {
  [[ -d "$1" ]] && find "$1" -maxdepth 1 -type f -name '*.png' -print -quit | grep -q .
}

mkdir -p "${RESULT_DIR}" "${METRIC_DIR}"

echo "[Eval:${DATASET}] checkpoint: ${CKPT}"
echo "[Eval:${DATASET}] experiment: ${EXP_NAME}"
echo "[Eval:${DATASET}] eval name:  ${EVAL_NAME}"
echo "[Eval:${DATASET}] input:      ${INPUT}"
echo "[Eval:${DATASET}] gt:         ${GT}"
echo "[Eval:${DATASET}] sr:         ${RESULT_DIR}"
echo "[Eval:${DATASET}] metrics:    ${METRIC_DIR}"

if [[ "${RUN_INFER}" == "1" ]]; then
  if [[ "${RESUME_EXISTING}" == "1" ]] && has_pngs "${RESULT_DIR}"; then
    echo "[skip] sr images already exist"
  else
    rm -rf "${RESULT_DIR}"
    mkdir -p "${RESULT_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" infer.py \
      --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
      --pretrained_path "${CKPT}" \
      --input_image "${INPUT}" \
      --output_dir "${RESULT_DIR}" \
      --upscale "${UPSCALE}" \
      --process_size "${PROCESS_SIZE}" \
      --mixed_precision "${MIXED_PRECISION}" \
      --align_method "${ALIGN_METHOD}" \
      --lambda_pix "${LAMBDA_PIX}" \
      --lambda_sem "${LAMBDA_SEM}"
  fi
fi

if [[ "${RUN_METRICS}" == "1" ]]; then
  if [[ "${RESUME_EXISTING}" == "1" && -f "${METRIC_DIR}/results.json" ]]; then
    echo "[skip] metrics already exist: ${METRIC_DIR}/results.json"
  else
    rm -rf "${METRIC_DIR}"
    mkdir -p "${METRIC_DIR}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" metrics.py \
      --inp_imgs "${RESULT_DIR}" \
      --gt_imgs "${GT}" \
      --log "${METRIC_DIR}" \
      --json "${METRIC_DIR}/results.json"
  fi
fi

echo "[done] ${METRIC_DIR}/results.json"
