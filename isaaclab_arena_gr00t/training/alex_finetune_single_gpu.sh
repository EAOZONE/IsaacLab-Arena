#!/usr/bin/env bash
# Fine-tune GR00T N1.6 on an Alex ability-hands LeRobot dataset (single GPU).
#
# Prerequisites (host, outside Arena Docker):
#   1. LeRobot dataset from convert_hdf5_to_lerobot.py + alex_open_microwave_config.yaml
#   2. Isaac-GR00T uv env: https://github.com/NVIDIA/Isaac-GR00T#installation-guide
#
# Usage:
#   export DATASET_PATH=/tmp/alex_demo_generated/lerobot
#   export OUTPUT_DIR=~/models/alex_open_microwave_finetune
#   bash isaaclab_arena_gr00t/training/alex_finetune_single_gpu.sh
#
# Run from Isaac-GR00T repo root, or set ISAAC_GR00T_DIR and ARENA_REPO paths below.
#
# Low-VRAM mode (≤16 GB, enabled by default):
#   Trains only the diffusion head (not LLM / visual / projector).
#   Uses launch_finetune_low_vram.py which enables gradient_checkpointing and
#   the Adafactor optimizer.  Set LOW_VRAM=0 to use the standard launcher instead.

set -euo pipefail

ISAAC_GR00T_DIR="${ISAAC_GR00T_DIR:-$(cd "$(dirname "$0")/../../submodules/Isaac-GR00T" && pwd)}"
ARENA_REPO="${ARENA_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"

DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to your LeRobot folder (contains meta/, data/, videos/)}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/models/alex_open_microwave_finetune}"
MODALITY_CONFIG="${MODALITY_CONFIG:-${ARENA_REPO}/isaaclab_arena_gr00t/embodiments/alex/alex_data_config.py}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.6-3B}"

# Effective batch = GLOBAL_BATCH_SIZE; per-step batch = GLOBAL_BATCH_SIZE / GRAD_ACCUM_STEPS.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"
# Reduce if data-prefetch pressure causes OOM (default 1e5 loads a lot into RAM).
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-200}"

LOW_VRAM="${LOW_VRAM:-1}"

cd "${ISAAC_GR00T_DIR}"

if [[ "${LOW_VRAM}" == "1" ]]; then
  LAUNCHER="${ARENA_REPO}/isaaclab_arena_gr00t/training/launch_finetune_low_vram.py"
else
  LAUNCHER="gr00t/experiment/launch_finetune.py"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python "${LAUNCHER}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --modality-config-path "${MODALITY_CONFIG}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --num-gpus 1 \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit 5 \
  --base-model-path "${BASE_MODEL_PATH}" \
  --no-tune-llm \
  --no-tune-visual \
  --no-tune-projector \
  --tune-diffusion-model \
  --dataloader-num-workers "${DATALOADER_WORKERS}" \
  --num-shards-per-epoch "${NUM_SHARDS_PER_EPOCH}" \
  --embodiment-tag NEW_EMBODIMENT \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
