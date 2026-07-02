#!/usr/bin/env bash
# Convert a local LeRobot v3.0 alex_lever dataset to the GR00T episode-per-file layout.
#
# Usage (from repo root):
#   bash isaaclab_arena_gr00t/training/prepare_alex_lever_dataset.sh
#
# Optional env:
#   INPUT_DIR   datasets/alex_lever          LeRobot v3 root (meta/tasks.parquet)
#   OUTPUT_DIR  datasets/alex_lever_gr00t    GR00T-layout output
#   UPLOAD_HF   0                            set 1 to push OUTPUT_DIR to HuggingFace
#   HF_DATASET_REPO  your-org/alex_lever     dataset repo (required if UPLOAD_HF=1)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INPUT_DIR="${INPUT_DIR:-${REPO_ROOT}/datasets/alex_lever}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/datasets/alex_lever_gr00t}"
MODALITY_TEMPLATE="${REPO_ROOT}/isaaclab_arena_gr00t/embodiments/alex/alex_lever_modality.json"
UPLOAD_HF="${UPLOAD_HF:-0}"
HF_DATASET_REPO="${HF_DATASET_REPO:-}"

for sub in meta data videos; do
  [[ -d "${INPUT_DIR}/${sub}" ]] || { echo "Missing ${INPUT_DIR}/${sub}"; exit 1; }
done

if [[ -f "${OUTPUT_DIR}/meta/stats.json" ]]; then
  echo "GR00T dataset already exists at ${OUTPUT_DIR} (meta/stats.json present). Skipping conversion."
else
  echo "=== Converting ${INPUT_DIR} -> ${OUTPUT_DIR} ==="
  python3 "${REPO_ROOT}/isaaclab_arena_gr00t/lerobot/convert_lerobot_v3_to_gr00t.py" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --modality_template "${MODALITY_TEMPLATE}" \
    --action_from_state_dims 13:33
fi

if [[ "${UPLOAD_HF}" == "1" ]]; then
  : "${HF_DATASET_REPO:?Set HF_DATASET_REPO when UPLOAD_HF=1}"
  : "${HF_TOKEN:?Set HF_TOKEN with write access}"
  echo "=== Uploading ${OUTPUT_DIR} -> ${HF_DATASET_REPO} ==="
  huggingface-cli upload "${HF_DATASET_REPO}" "${OUTPUT_DIR}" . --repo-type dataset
fi

echo "=== Ready for training ==="
echo "  GR00T dataset: ${OUTPUT_DIR}"
echo "  Modality config: isaaclab_arena_gr00t/embodiments/alex/alex_lever_data_config.py"
echo "  Embodiment tag: NEW_EMBODIMENT"
