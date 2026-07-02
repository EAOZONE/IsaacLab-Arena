#!/usr/bin/env bash
# Fine-tune GR00T N1.6 on a local alex_lever GR00T-layout dataset (single node).
#
# Requires the Isaac-GR00T uv env (submodules/Isaac-GR00T). Run prepare first:
#   bash isaaclab_arena_gr00t/training/prepare_alex_lever_dataset.sh
#
# Usage (from repo root):
#   export DATASET_PATH=datasets/alex_lever_gr00t
#   export OUTPUT_DIR=~/models/alex_lever_gr00t_finetune
#   bash isaaclab_arena_gr00t/training/alex_finetune_single_gpu.sh
#
# For <=16 GB GPUs:
#   LOW_VRAM=1 bash isaaclab_arena_gr00t/training/alex_finetune_single_gpu.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
GR00T_DIR="${GR00T_DIR:-${REPO_ROOT}/submodules/Isaac-GR00T}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/datasets/alex_lever_gr00t}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/models/alex_lever_gr00t_finetune}"
MODALITY_CONFIG="${MODALITY_CONFIG:-${REPO_ROOT}/isaaclab_arena_gr00t/embodiments/alex/alex_lever_data_config.py}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.6-3B}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_STEPS="${MAX_STEPS:-15000}"
SAVE_STEPS="${SAVE_STEPS:-2500}"
LOW_VRAM="${LOW_VRAM:-0}"
USE_LORA="${USE_LORA:-0}"
LORA_RANK="${LORA_RANK:-64}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"

[[ -f "${DATASET_PATH}/meta/stats.json" ]] || {
  echo "GR00T dataset not found at ${DATASET_PATH}. Run prepare_alex_lever_dataset.sh first."
  exit 1
}

if [[ "${LOW_VRAM}" == "1" ]]; then
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
  TUNE_FLAGS=(--no-tune-llm --no-tune-visual --no-tune-projector --tune-diffusion-model)
else
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
  TUNE_FLAGS=(--tune-llm --tune-visual --tune-projector --tune-diffusion-model)
fi

if [[ "${USE_LORA}" == "1" ]]; then
  TUNE_FLAGS+=(--use-lora --lora-rank "${LORA_RANK}")
fi

mkdir -p "${OUTPUT_DIR}"
cd "${GR00T_DIR}"

LAUNCHER=(uv run python)
if (( NUM_GPUS > 1 )); then
  LAUNCHER=(uv run torchrun --nproc_per_node="${NUM_GPUS}")
fi

"${LAUNCHER[@]}" gr00t/experiment/launch_finetune.py \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --modality-config-path "${MODALITY_CONFIG}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --num-gpus "${NUM_GPUS}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit 5 \
  --base-model-path "${BASE_MODEL_PATH}" \
  "${TUNE_FLAGS[@]}" \
  --dataloader-num-workers "${DATALOADER_WORKERS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
