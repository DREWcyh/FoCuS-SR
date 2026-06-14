#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-merged_unet_weight_semantic_probe_hq_b200}"
EXP_NAME="${EXP_NAME:-${RUN_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/${EXP_NAME}/layers}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${RESUME_CKPT:-}}"
DATASET_TXT="${DATASET_TXT:-preset/gt_hq_path.txt}"
NUM_BATCHES="${NUM_BATCHES:-200}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-preset/models/stable-diffusion-2-1-base}"
PRETRAINED_MODEL_PATH_CSD="${PRETRAINED_MODEL_PATH_CSD:-${PRETRAINED_MODEL_PATH}}"
DEG_FILE_PATH="${DEG_FILE_PATH:-params.yml}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
SEED="${SEED:-123}"

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  echo "CHECKPOINT_PATH or RESUME_CKPT is required for merged semantic probing." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" probe.py \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --probe_space merged_unet_weight \
  --merge_lora_scope pix_lora \
  --merged_weight_target semantic_lora_targets \
  --num_batches "${NUM_BATCHES}" \
  --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
  --pretrained_model_path_csd "${PRETRAINED_MODEL_PATH_CSD}" \
  --dataset_txt_paths "${DATASET_TXT}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --deg_file_path "${DEG_FILE_PATH}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --seed "${SEED}"
