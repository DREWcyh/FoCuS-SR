#!/usr/bin/env bash
set -euo pipefail

USER_RUN_NAME="${RUN_NAME:-}"
USER_OUTPUT_ROOT="${OUTPUT_ROOT:-}"
USER_PROBE_ROOT="${PROBE_ROOT:-}"
USER_DATA_CURRICULUM_DIR="${DATA_CURRICULUM_DIR:-}"
USER_HQ_CURRICULUM_TXT="${HQ_CURRICULUM_TXT:-}"
USER_STAGE1_DIR="${STAGE1_DIR:-}"
USER_STAGE2A_DIR="${STAGE2A_DIR:-}"
USER_STAGE2B_DIR="${STAGE2B_DIR:-}"
USER_STAGE1_CKPT="${STAGE1_CKPT:-}"
USER_STAGE2A_CKPT="${STAGE2A_CKPT:-}"
USER_STAGE2B_CKPT="${STAGE2B_CKPT:-}"
USER_MERGED_SEM_PROBE_DIR="${MERGED_SEM_PROBE_DIR:-}"
USER_SAFE_TO_RISK_CURRICULUM_JSON="${SAFE_TO_RISK_CURRICULUM_JSON:-}"

source "$(dirname "$0")/config.sh"

RUN_NAME="${USER_RUN_NAME:-focus_sr_main}"
OUTPUT_ROOT="${USER_OUTPUT_ROOT:-experiments/${RUN_NAME}/train}"
PROBE_ROOT="${USER_PROBE_ROOT:-experiments/${RUN_NAME}/layers}"
DATA_CURRICULUM_DIR="${USER_DATA_CURRICULUM_DIR:-experiments/${RUN_NAME}/data}"
HQ_CURRICULUM_TXT="${USER_HQ_CURRICULUM_TXT:-${DATA_CURRICULUM_DIR}/gt_hq_clip_musiq_path.txt}"

STAGE2A_STEPS="${STAGE2A_STEPS:-5500}"
STAGE2B_STEPS="${STAGE2B_STEPS:-3000}"
RUN_DATA_SELECTION="${RUN_DATA_SELECTION:-1}"
MUSIQ_THRESHOLD="${MUSIQ_THRESHOLD:-78.0}"
CLIP_CLUSTER_COUNT="${CLIP_CLUSTER_COUNT:-50}"
CLIP_MODEL="${CLIP_MODEL:-ViT-B-32}"
CLIP_PRETRAINED="${CLIP_PRETRAINED:-laion2b_s34b_b79k}"
SKIP_CLIP="${SKIP_CLIP:-0}"
DATA_SELECTION_DEVICE="${DATA_SELECTION_DEVICE:-cuda}"
DATA_SELECTION_MAX_IMAGES="${DATA_SELECTION_MAX_IMAGES:-}"
DATA_SELECTION_TARGET_COUNT="${DATA_SELECTION_TARGET_COUNT:-}"
STAGE1_PRESET_SET="${STAGE1_PRESET_SET:-5}"
if [[ -z "${STAGE1_DEG_FILE_PATHS:-}" ]]; then
  if [[ "${STAGE1_PRESET_SET}" == "3" ]]; then
    STAGE1_DEG_FILE_PATHS="src/datasets/stage1_3preset/params_stage1_mild.yml,src/datasets/stage1_3preset/params_stage1_medium.yml,src/datasets/stage1_3preset/params_stage1_heavy.yml"
    STAGE1_DEG_PRESET_PROBS="${STAGE1_DEG_PRESET_PROBS:-0.3,0.4,0.3}"
  elif [[ "${STAGE1_PRESET_SET}" == "5" ]]; then
    STAGE1_DEG_FILE_PATHS="src/datasets/stage1_5preset/params_stage1_clean_mild.yml,src/datasets/stage1_5preset/params_stage1_standard.yml,src/datasets/stage1_5preset/params_stage1_blur_aliasing.yml,src/datasets/stage1_5preset/params_stage1_noise_jpeg.yml,src/datasets/stage1_5preset/params_stage1_severe_mixed.yml"
    STAGE1_DEG_PRESET_PROBS="${STAGE1_DEG_PRESET_PROBS:-0.20,0.25,0.20,0.20,0.15}"
  else
    echo "Unsupported STAGE1_PRESET_SET=${STAGE1_PRESET_SET}; use 3 or 5, or set STAGE1_DEG_FILE_PATHS manually." >&2
    exit 1
  fi
else
  STAGE1_DEG_PRESET_PROBS="${STAGE1_DEG_PRESET_PROBS:-}"
fi

MERGED_SEM_PROBE_DIR="${USER_MERGED_SEM_PROBE_DIR:-${PROBE_ROOT}/merged_unet_weight_semantic_probe_hq_b${PROBE_BATCHES}}"
SAFE_TO_RISK_CURRICULUM_JSON="${USER_SAFE_TO_RISK_CURRICULUM_JSON:-${PROBE_ROOT}/safe_to_risk_rank_curriculum_merged_hq.json}"

STAGE1_DIR="${USER_STAGE1_DIR:-${OUTPUT_ROOT}/stage1_uniform_rank${LORA_RANK_PIX}_all_l2_s${STAGE1_STEPS}}"
STAGE2A_DIR="${USER_STAGE2A_DIR:-${OUTPUT_ROOT}/stage2a_lowconf_rank_hq_obj_s${STAGE2A_STEPS}}"
STAGE2B_DIR="${USER_STAGE2B_DIR:-${OUTPUT_ROOT}/stage2b_highconf_rank_hq_safeobj_s${STAGE2B_STEPS}}"
STAGE1_CKPT="${USER_STAGE1_CKPT:-${STAGE1_DIR}/checkpoints/model_${STAGE1_STEPS}.pkl}"
STAGE2A_CKPT="${USER_STAGE2A_CKPT:-${STAGE2A_DIR}/checkpoints/model_${STAGE2A_STEPS}.pkl}"
STAGE2B_CKPT="${USER_STAGE2B_CKPT:-${STAGE2B_DIR}/checkpoints/model_${STAGE2B_STEPS}.pkl}"

echo "== FoCuS-SR main training pipeline =="
echo "run: ${RUN_NAME}"
echo "output: ${OUTPUT_ROOT}"
echo "probe root: ${PROBE_ROOT}"
echo "data curriculum dir: ${DATA_CURRICULUM_DIR}"
echo "stage1: uniform pixel LoRA rank=${LORA_RANK_PIX}, broad degradation pair curriculum"
echo "stage1 preset set: ${STAGE1_PRESET_SET}"
echo "stage1 degradation presets: ${STAGE1_DEG_FILE_PATHS}"
echo "stage1 degradation preset probs: ${STAGE1_DEG_PRESET_PROBS}"
echo "stage2 split: conflict-only top30% -> Stage2b; no prune"
echo "semantic rank: importance-only, budget around uniform rank=${LORA_RANK_SEM}"

test -f "${GT_ALL_PATH}"
IFS=',' read -r -a STAGE1_DEG_FILES <<< "${STAGE1_DEG_FILE_PATHS}"
for deg_file in "${STAGE1_DEG_FILES[@]}"; do
  deg_file="${deg_file#"${deg_file%%[![:space:]]*}"}"
  deg_file="${deg_file%"${deg_file##*[![:space:]]}"}"
  if [[ ! -f "${deg_file}" && ! -f "src/datasets/${deg_file}" ]]; then
    echo "Missing Stage1 degradation preset: ${deg_file}" >&2
    exit 1
  fi
done

mkdir -p "${PROBE_ROOT}" "${DATA_CURRICULUM_DIR}"

if [[ "${RUN_DATA_SELECTION}" == "1" ]]; then
  echo
  echo "[0/5] Data curriculum: MUSIQ quality + CLIP balanced semantic diversity"
  data_args=(
    --input_txt "${GT_ALL_PATH}"
    --output_dir "${DATA_CURRICULUM_DIR}"
    --output_txt "${HQ_CURRICULUM_TXT}"
    --musiq_threshold "${MUSIQ_THRESHOLD}"
    --cluster_count "${CLIP_CLUSTER_COUNT}"
    --clip_model "${CLIP_MODEL}"
    --clip_pretrained "${CLIP_PRETRAINED}"
    --device "${DATA_SELECTION_DEVICE}"
  )
  if [[ -n "${DATA_SELECTION_MAX_IMAGES}" ]]; then
    data_args+=(--max_images "${DATA_SELECTION_MAX_IMAGES}")
  fi
  if [[ -n "${DATA_SELECTION_TARGET_COUNT}" ]]; then
    data_args+=(--target_count "${DATA_SELECTION_TARGET_COUNT}")
  fi
  if [[ "${SKIP_CLIP}" == "1" ]]; then
    data_args+=(--skip_clip)
  fi
  "${PYTHON_BIN}" scripts/build_data_curriculum.py "${data_args[@]}"
  HQ_TRAIN_PATH="${HQ_CURRICULUM_TXT}"
else
  echo
  echo "[0/5] Data curriculum skipped: using existing GT_HQ_PATH=${GT_HQ_PATH}"
  test -f "${GT_HQ_PATH}"
  HQ_TRAIN_PATH="${GT_HQ_PATH}"
fi
test -f "${HQ_TRAIN_PATH}"

echo
echo "[1/5] Stage1 train: all-data + broad degradation pairs + L2, train uniform-rank pixel LoRA"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m accelerate.commands.launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision "${MIXED_PRECISION}" \
  --dynamo_backend no \
  train.py \
    --pretrained_model_path="${PRETRAINED_MODEL_PATH}" \
    --pretrained_model_path_csd="${PRETRAINED_MODEL_PATH_CSD}" \
    --dataset_txt_paths="${GT_ALL_PATH}" \
    --highquality_dataset_txt_paths="${GT_ALL_PATH}" \
    --dataset_test_folder="${DATASET_TEST_FOLDER}" \
    --output_dir="${STAGE1_DIR}" \
    --learning_rate="${STAGE1_LR}" \
    --train_batch_size="${TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps="${GRAD_ACCUM}" \
    --checkpointing_steps="${CHECKPOINTING_STEPS}" \
    --eval_freq="${EVAL_FREQ}" \
    --max_train_steps="${STAGE1_STEPS}" \
    --pix_steps="${STAGE1_STEPS}" \
    --lambda_l2=1.0 \
    --lambda_lpips=0.0 \
    --lambda_csd=0.0 \
    --lambda_consistency=0.0 \
    --enable_pixel_stage_perceptual_losses False \
    --lora_rank_unet_pix="${LORA_RANK_PIX}" \
    --lora_rank_unet_sem="${LORA_RANK_SEM}" \
    --cfg_csd "${CFG_CSD}" \
    --min_dm_step_ratio="${MIN_DM_STEP_RATIO}" \
    --max_dm_step_ratio="${MAX_DM_STEP_RATIO}" \
    --null_text_ratio="${NULL_TEXT_RATIO}" \
    --align_method="${ALIGN_METHOD}" \
    --deg_file_paths="${STAGE1_DEG_FILE_PATHS}" \
    --deg_preset_probs="${STAGE1_DEG_PRESET_PROBS}" \
    --tracker_project_name "focussr" \
    --mixed_precision "${MIXED_PRECISION}" \
    --report_to "${REPORT_TO}" \
    --enable_xformers_memory_efficient_attention \
    --is_module False
test -f "${STAGE1_CKPT}"

echo
echo "[2/5] Stage2 probe: merged pixel foundation -> semantic importance/conflict"
GPU="${GPU}" \
PYTHON_BIN="${PYTHON_BIN}" \
RUN_NAME="$(basename "${MERGED_SEM_PROBE_DIR}")" \
OUTPUT_DIR="${MERGED_SEM_PROBE_DIR}" \
CHECKPOINT_PATH="${STAGE1_CKPT}" \
DATASET_TXT="${HQ_TRAIN_PATH}" \
NUM_BATCHES="${PROBE_BATCHES}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH}" \
PRETRAINED_MODEL_PATH_CSD="${PRETRAINED_MODEL_PATH_CSD}" \
MIXED_PRECISION="${MIXED_PRECISION}" \
scripts/probe_layers.sh

"${PYTHON_BIN}" scripts/build_layer_curriculum.py \
  --importance_json "${MERGED_SEM_PROBE_DIR}/layer_importance.json" \
  --checkpoint_path "${STAGE1_CKPT}" \
  --output "${SAFE_TO_RISK_CURRICULUM_JSON}" \
  --baseline_rank "${LORA_RANK_SEM}" \
  --rank_values "${RANK_VALUES}" \
  --budget_tolerance "${BUDGET_TOLERANCE}"
test -f "${SAFE_TO_RISK_CURRICULUM_JSON}"

echo
echo "[3/5] Stage2a train: low-conflict semantic modules, aggressive semantic objective"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m accelerate.commands.launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision "${MIXED_PRECISION}" \
  --dynamo_backend no \
  train.py \
    --pretrained_model_path="${PRETRAINED_MODEL_PATH}" \
    --pretrained_model_path_csd="${PRETRAINED_MODEL_PATH_CSD}" \
    --dataset_txt_paths="${GT_LSDIR_PATH}" \
    --highquality_dataset_txt_paths="${HQ_TRAIN_PATH}" \
    --dataset_test_folder="${DATASET_TEST_FOLDER}" \
    --prob=0.0 \
    --output_dir="${STAGE2A_DIR}" \
    --resume_ckpt "${STAGE1_CKPT}" \
    --reset_semantic_lora_on_resume True \
    --semantic_rank_curriculum_json "${SAFE_TO_RISK_CURRICULUM_JSON}" \
    --sequential_semantic_lora_curriculum_json "${SAFE_TO_RISK_CURRICULUM_JSON}" \
    --semantic_train_phase stage2a \
    --learning_rate="${STAGE2_LR}" \
    --train_batch_size="${TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps="${GRAD_ACCUM}" \
    --checkpointing_steps="${CHECKPOINTING_STEPS}" \
    --eval_freq="${EVAL_FREQ}" \
    --max_train_steps="${STAGE2A_STEPS}" \
    --pix_steps=0 \
    --lambda_l2=0.3 \
    --lambda_lpips=5.0 \
    --lambda_csd=1.0 \
    --lambda_consistency=0.05 \
    --consistency_aug hflip \
    --consistency_space semantic_delta \
    --lora_rank_unet_pix="${LORA_RANK_PIX}" \
    --lora_rank_unet_sem="${LORA_RANK_SEM}" \
    --cfg_csd "${CFG_CSD}" \
    --min_dm_step_ratio="${MIN_DM_STEP_RATIO}" \
    --max_dm_step_ratio="${MAX_DM_STEP_RATIO}" \
    --null_text_ratio="${NULL_TEXT_RATIO}" \
    --align_method="${ALIGN_METHOD}" \
    --deg_file_path="params.yml" \
    --tracker_project_name "focussr" \
    --mixed_precision "${MIXED_PRECISION}" \
    --report_to "${REPORT_TO}" \
    --enable_xformers_memory_efficient_attention \
    --is_module False
test -f "${STAGE2A_CKPT}"

echo
echo "[4/5] Stage2b train: high-conflict semantic modules, conservative constrained objective"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m accelerate.commands.launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision "${MIXED_PRECISION}" \
  --dynamo_backend no \
  train.py \
    --pretrained_model_path="${PRETRAINED_MODEL_PATH}" \
    --pretrained_model_path_csd="${PRETRAINED_MODEL_PATH_CSD}" \
    --dataset_txt_paths="${GT_LSDIR_PATH}" \
    --highquality_dataset_txt_paths="${HQ_TRAIN_PATH}" \
    --dataset_test_folder="${DATASET_TEST_FOLDER}" \
    --prob=0.0 \
    --output_dir="${STAGE2B_DIR}" \
    --resume_ckpt "${STAGE2A_CKPT}" \
    --semantic_rank_curriculum_json "${SAFE_TO_RISK_CURRICULUM_JSON}" \
    --sequential_semantic_lora_curriculum_json "${SAFE_TO_RISK_CURRICULUM_JSON}" \
    --semantic_train_phase stage2b \
    --learning_rate="${STAGE2_LR}" \
    --train_batch_size="${TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps="${GRAD_ACCUM}" \
    --checkpointing_steps="${CHECKPOINTING_STEPS}" \
    --eval_freq="${EVAL_FREQ}" \
    --max_train_steps="${STAGE2B_STEPS}" \
    --pix_steps=0 \
    --lambda_l2=1.0 \
    --lambda_lpips=2.0 \
    --lambda_csd=0.5 \
    --lambda_consistency=0.3 \
    --consistency_aug hflip \
    --consistency_space semantic_delta \
    --lora_rank_unet_pix="${LORA_RANK_PIX}" \
    --lora_rank_unet_sem="${LORA_RANK_SEM}" \
    --cfg_csd "${CFG_CSD}" \
    --min_dm_step_ratio="${MIN_DM_STEP_RATIO}" \
    --max_dm_step_ratio="${MAX_DM_STEP_RATIO}" \
    --null_text_ratio="${NULL_TEXT_RATIO}" \
    --align_method="${ALIGN_METHOD}" \
    --deg_file_path="params.yml" \
    --tracker_project_name "focussr" \
    --mixed_precision "${MIXED_PRECISION}" \
    --report_to "${REPORT_TO}" \
    --enable_xformers_memory_efficient_attention \
    --is_module False
test -f "${STAGE2B_CKPT}"

echo
echo "[5/5] Done."
echo "HQ train txt: ${HQ_TRAIN_PATH}"
echo "Safe-to-risk curriculum: ${SAFE_TO_RISK_CURRICULUM_JSON}"
echo "Stage1 checkpoint: ${STAGE1_CKPT}"
echo "Stage2a checkpoint: ${STAGE2A_CKPT}"
echo "Final checkpoint: ${STAGE2B_CKPT}"
echo "Eval command:"
echo "GPU=${GPU} CKPT=${STAGE2B_CKPT} EXP_NAME=${RUN_NAME} EVAL_NAME=s$((STAGE2A_STEPS + STAGE2B_STEPS)) bash scripts/eval_all.sh"
