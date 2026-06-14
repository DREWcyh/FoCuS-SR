#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT="${CKPT:-preset/models/focussr.pkl}"
INPUT_IMAGE="${1:-preset/test_datasets}"
OUTPUT_DIR="${2:-results/demo}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" infer.py \
  --pretrained_model_path preset/models/stable-diffusion-2-1-base \
  --pretrained_path "${CKPT}" \
  --process_size 512 \
  --upscale 4 \
  --input_image "${INPUT_IMAGE}" \
  --output_dir "${OUTPUT_DIR}" \
  --default
