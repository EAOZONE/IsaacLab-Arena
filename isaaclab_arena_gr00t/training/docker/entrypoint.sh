#!/usr/bin/env bash
# Download dataset -> fine-tune GR00T N1.6 -> upload checkpoint to HuggingFace.
#
# Required env:
#   HF_TOKEN          HuggingFace token with write access (unless SKIP_UPLOAD=1)
# Optional env (defaults shown):
#   HF_DATASET_ID     H2Ozone/test_obs_new      dataset repo to download
#   HF_MODEL_REPO     H2Ozone/test_obs_new_gr00t   model repo to upload to
#   MODALITY_CONFIG   alex_test_obs_new_data_config.py  GR00T modality config path
#   MODALITY_TEMPLATE alex_test_obs_new_modality.json   modality.json for v3 conversion
#   ACTION_FROM_STATE_DIMS  ""      optional action dims filled from state at conversion
#   SKIP_UPLOAD       0     set 1 to train without uploading
#   UPLOAD_OPTIMIZER_STATE  0     set 1 to also upload optimizer/scheduler/rng state
#   GLOBAL_BATCH_SIZE 8
#   GRAD_ACCUM_STEPS  1
#   MAX_STEPS         30000
#   SAVE_STEPS        5000
#   NUM_GPUS          1
#   DATALOADER_WORKERS 2
#   LOW_VRAM          0     set 1 for <=16GB GPUs: diffusion head only, batch 2 + accum
#   USE_LORA          0     set 1 to use LoRA for fine-tuning
#   LORA_RANK         64    LoRA rank to use if USE_LORA=1
#   SKIP_DOWNLOAD     0     set 1 when DATASET_PATH is bind-mounted (cluster/local data)
#   OUTPUT_DIR        /checkpoints  (mount a volume here to survive restarts; training
#                                    auto-resumes from the last checkpoint it finds)
#   COLOR_JITTER_BRIGHTNESS/CONTRAST/SATURATION/HUE   0.4/0.5/0.6/0.1
#                     Photometric augmentation strength on real training frames (torchvision
#                     ColorJitter ranges; hue is capped at 0.5). Raised from the previous
#                     0.3/0.4/0.5/0.08 defaults -- those didn't close the sim2real gap for
#                     lever_eef closed-loop eval (policy stayed frozen on rendered sim images
#                     despite that jitter), so this pushes further before trying anything else.
#   RANDOM_ROTATION_ANGLE  10   Max +/- degrees of random image rotation (was unset/0 before).

set -euo pipefail

HF_DATASET_ID="${HF_DATASET_ID:-H2Ozone/test_obs_new}"
HF_MODEL_REPO="${HF_MODEL_REPO:-H2Ozone/test_obs_new_gr00t}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
# Lives under /cache so the dataset download persists with the same volume/bind
# as the HF model cache (and lands on writable storage on clusters).
DATASET_DIR_NAME="${HF_DATASET_ID//\//__}"
DATASET_PATH="${DATASET_PATH:-/cache/dataset/${DATASET_DIR_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-/checkpoints}"
ARENA_DIR=/workspace/IsaacLab-Arena
MODALITY_CONFIG="${MODALITY_CONFIG:-${ARENA_DIR}/isaaclab_arena_gr00t/embodiments/alex/alex_test_obs_new_data_config.py}"
MODALITY_TEMPLATE="${MODALITY_TEMPLATE:-${ARENA_DIR}/isaaclab_arena_gr00t/embodiments/alex/alex_test_obs_new_modality.json}"
# H2Ozone/test_obs_new has real 36-dim EEF actions (no spine); keep them untouched by default.
# Set e.g. "13:33" only for older datasets with missing command streams.
ACTION_FROM_STATE_DIMS="${ACTION_FROM_STATE_DIMS:-}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.6-3B}"

MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
NUM_GPUS="${NUM_GPUS:-1}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-16}"
LOW_VRAM="${LOW_VRAM:-0}"
USE_LORA="${USE_LORA:-0}"
LORA_RANK="${LORA_RANK:-64}"

COLOR_JITTER_BRIGHTNESS="${COLOR_JITTER_BRIGHTNESS:-0.4}"
COLOR_JITTER_CONTRAST="${COLOR_JITTER_CONTRAST:-0.5}"
COLOR_JITTER_SATURATION="${COLOR_JITTER_SATURATION:-0.6}"
COLOR_JITTER_HUE="${COLOR_JITTER_HUE:-0.1}"
RANDOM_ROTATION_ANGLE="${RANDOM_ROTATION_ANGLE:-10}"

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

cd /workspace/Isaac-GR00T

echo "=== GPU check ==="
nvidia-smi

[[ -f "${MODALITY_CONFIG}" ]] || { echo "Missing MODALITY_CONFIG: ${MODALITY_CONFIG}"; exit 1; }
[[ -f "${MODALITY_TEMPLATE}" ]] || { echo "Missing MODALITY_TEMPLATE: ${MODALITY_TEMPLATE}"; exit 1; }

# Fail fast on a bad/missing token before hours of training.
if [[ "${SKIP_UPLOAD}" != "1" ]]; then
  : "${HF_TOKEN:?Set HF_TOKEN (write access) or SKIP_UPLOAD=1}"
  echo "=== Verifying HuggingFace token and creating ${HF_MODEL_REPO} ==="
  uv run python /workspace/upload_to_hf.py --verify-only --repo-id "${HF_MODEL_REPO}"
fi

if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
  echo "=== Using local dataset at ${DATASET_PATH} (SKIP_DOWNLOAD=1) ==="
else
  echo "=== Downloading dataset ${HF_DATASET_ID} -> ${DATASET_PATH} ==="
  uv run python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${HF_DATASET_ID}", repo_type="dataset", local_dir="${DATASET_PATH}")
EOF
fi

for sub in meta data videos; do
  [[ -d "${DATASET_PATH}/${sub}" ]] || { echo "Missing ${DATASET_PATH}/${sub} — not a LeRobot dataset"; exit 1; }
done

# LeRobot v3.0 datasets (chunked parquets, meta/tasks.parquet) must be converted
# to the episode-per-file layout GR00T's loader reads.
if [[ ! -f "${DATASET_PATH}/meta/episodes.jsonl" && -f "${DATASET_PATH}/meta/tasks.parquet" ]]; then
  CONVERTED_PATH="${DATASET_PATH%/}_gr00t"
  if [[ ! -f "${CONVERTED_PATH}/meta/stats.json" ]]; then
    echo "=== Converting LeRobot v3 dataset -> ${CONVERTED_PATH} ==="
    CONVERT_ARGS=(
      --input_dir "${DATASET_PATH}"
      --output_dir "${CONVERTED_PATH}"
      --modality_template "${MODALITY_TEMPLATE}"
    )
    [[ -n "${ACTION_FROM_STATE_DIMS}" ]] && CONVERT_ARGS+=(--action_from_state_dims "${ACTION_FROM_STATE_DIMS}")
    uv run python "${ARENA_DIR}/isaaclab_arena_gr00t/lerobot/convert_lerobot_v3_to_gr00t.py" "${CONVERT_ARGS[@]}"
  fi
  DATASET_PATH="${CONVERTED_PATH}"
fi

mkdir -p "${OUTPUT_DIR}"

echo "=== Fine-tuning (output: ${OUTPUT_DIR}) ==="
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
  --color-jitter-params \
    brightness "${COLOR_JITTER_BRIGHTNESS}" \
    contrast "${COLOR_JITTER_CONTRAST}" \
    saturation "${COLOR_JITTER_SATURATION}" \
    hue "${COLOR_JITTER_HUE}" \
  --random-rotation-angle "${RANDOM_ROTATION_ANGLE}"

if [[ "${SKIP_UPLOAD}" != "1" ]]; then
  echo "=== Uploading latest checkpoint to ${HF_MODEL_REPO} ==="
  UPLOAD_ARGS=(--repo-id "${HF_MODEL_REPO}" --output-dir "${OUTPUT_DIR}")
  [[ "${UPLOAD_OPTIMIZER_STATE:-0}" == "1" ]] && UPLOAD_ARGS+=(--include-optimizer-state)
  uv run python /workspace/upload_to_hf.py "${UPLOAD_ARGS[@]}"
fi

echo "=== Done ==="
