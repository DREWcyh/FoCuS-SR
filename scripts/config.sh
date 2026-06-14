#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export GPU="${GPU:-0}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

export RUN_NAME="${RUN_NAME:-focus_sr_main}"
export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-preset/models/stable-diffusion-2-1-base}"
export PRETRAINED_MODEL_PATH_CSD="${PRETRAINED_MODEL_PATH_CSD:-${PRETRAINED_MODEL_PATH}}"

export GT_LSDIR_PATH="${GT_LSDIR_PATH:-preset/gt_lsdir_path.txt}"
export GT_FFHQ_PATH="${GT_FFHQ_PATH:-preset/gt_ffhq_path.txt}"
export GT_ALL_PATH="${GT_ALL_PATH:-preset/gt_all_path.txt}"
export GT_HQ_PATH="${GT_HQ_PATH:-preset/gt_hq_path.txt}"
export DATASET_TEST_FOLDER="${DATASET_TEST_FOLDER:-preset/testfolder}"

export PROBE_BATCHES="${PROBE_BATCHES:-200}"
export STAGE1_STEPS="${STAGE1_STEPS:-6000}"
export STAGE2_STEPS="${STAGE2_STEPS:-8500}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
export REPORT_TO="${REPORT_TO:-tensorboard}"

export LORA_RANK_PIX="${LORA_RANK_PIX:-4}"
export LORA_RANK_SEM="${LORA_RANK_SEM:-4}"
export RANK_VALUES="${RANK_VALUES:-8,4,2}"
export CONSISTENCY_VALUES="${CONSISTENCY_VALUES:-0.10,0.05,0.00}"
export BUDGET_TOLERANCE="${BUDGET_TOLERANCE:-0.005}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/${RUN_NAME}/train}"
export PROBE_ROOT="${PROBE_ROOT:-experiments/${RUN_NAME}/layers}"

export BASE_PIXEL_PROBE_DIR="${BASE_PIXEL_PROBE_DIR:-${PROBE_ROOT}/base_unet_weight_pixel_probe_all_data_b${PROBE_BATCHES}}"
export PIXEL_CURRICULUM_JSON="${PIXEL_CURRICULUM_JSON:-${PROBE_ROOT}/pixel_rank_curriculum_base_weight_all_data.json}"
export MERGED_SEM_PROBE_DIR="${MERGED_SEM_PROBE_DIR:-${PROBE_ROOT}/merged_unet_weight_semantic_probe_hq_b${PROBE_BATCHES}}"
export SEMANTIC_CURRICULUM_JSON="${SEMANTIC_CURRICULUM_JSON:-${PROBE_ROOT}/semantic_rank_consistency_curriculum_merged_hq.json}"

export STAGE1_DIR="${STAGE1_DIR:-${OUTPUT_ROOT}/stage1_pixel_rank_all_data_s${STAGE1_STEPS}}"
export STAGE2_DIR="${STAGE2_DIR:-${OUTPUT_ROOT}/stage2_semantic_rank_cons_hq_s${STAGE2_STEPS}}"
export STAGE1_CKPT="${STAGE1_CKPT:-${STAGE1_DIR}/checkpoints/model_${STAGE1_STEPS}.pkl}"
export STAGE2_CKPT="${STAGE2_CKPT:-${STAGE2_DIR}/checkpoints/model_${STAGE2_STEPS}.pkl}"

export CFG_CSD="${CFG_CSD:-7.5}"
export MIN_DM_STEP_RATIO="${MIN_DM_STEP_RATIO:-0.02}"
export MAX_DM_STEP_RATIO="${MAX_DM_STEP_RATIO:-0.5}"
export NULL_TEXT_RATIO="${NULL_TEXT_RATIO:-0.5}"
export ALIGN_METHOD="${ALIGN_METHOD:-adain}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}"
export EVAL_FREQ="${EVAL_FREQ:-500}"

export STAGE1_LR="${STAGE1_LR:-5e-5}"
export STAGE2_LR="${STAGE2_LR:-5e-5}"
export STAGE2_L2="${STAGE2_L2:-0.5}"
export STAGE2_LPIPS="${STAGE2_LPIPS:-5.0}"
export STAGE2_CSD="${STAGE2_CSD:-1.0}"
export STAGE2_CONSISTENCY="${STAGE2_CONSISTENCY:-0.1}"

export USE_RANK="${USE_RANK:-1}"
