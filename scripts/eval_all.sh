#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CKPT="${CKPT:-${1:-preset/models/focussr.pkl}}"
EXP_NAME="${EXP_NAME:-$(basename "${CKPT}" .pkl)}"
EVAL_NAME="${EVAL_NAME:-default}"
DATASETS="${DATASETS:-realsr,drealsr,div2k}"

IFS=',' read -r -a DATASET_LIST <<< "${DATASETS}"

for dataset in "${DATASET_LIST[@]}"; do
  dataset="${dataset#"${dataset%%[![:space:]]*}"}"
  dataset="${dataset%"${dataset##*[![:space:]]}"}"
  if [[ -z "${dataset}" ]]; then
    continue
  fi

  echo
  echo "== Evaluating ${dataset} =="
  DATASET="${dataset}" \
  CKPT="${CKPT}" \
  EXP_NAME="${EXP_NAME}" \
  EVAL_NAME="${EVAL_NAME}" \
  bash scripts/eval.sh
done

echo
echo "All requested evaluations finished."
