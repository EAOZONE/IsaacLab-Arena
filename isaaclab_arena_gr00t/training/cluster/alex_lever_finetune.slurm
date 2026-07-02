#!/bin/bash
#SBATCH --job-name=alex-lever-gr00t
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=alex_lever_gr00t_%j.out
#SBATCH --error=alex_lever_gr00t_%j.err

# Fine-tune GR00T N1.6 on real-robot alex_lever data (H100 cluster via Apptainer).
#
# One-time setup on the cluster:
#   1. Build the training image locally and push to a registry, OR pull a pre-built tag:
#        ./isaaclab_arena_gr00t/training/docker/build.sh ghcr.io/<you>/alex-gr00t-train
#        docker push ghcr.io/<you>/alex-gr00t-train
#   2. Convert the dataset on your workstation and rsync to scratch:
#        bash isaaclab_arena_gr00t/training/prepare_alex_lever_dataset.sh
#        rsync -av datasets/alex_lever_gr00t/ $SCRATCH/alex_lever_gr00t/
#   3. Pull the SIF once:
#        apptainer pull alex-gr00t-train.sif docker://ghcr.io/<you>/alex-gr00t-train:latest
#
# Submit:
#   export HF_TOKEN=hf_xxx
#   export HF_MODEL_REPO=<you>/alex_lever_gr00t_real
#   sbatch isaaclab_arena_gr00t/training/cluster/alex_lever_finetune.slurm
#
# Knobs (override before sbatch):
#   NUM_GPUS=8
#   GLOBAL_BATCH_SIZE=64
#   MAX_STEPS=15000
#   SAVE_STEPS=2500

set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN (write access for checkpoint upload)}"
: "${SCRATCH:?Set SCRATCH to your cluster scratch path}"

NUM_GPUS="${NUM_GPUS:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-15000}"
SAVE_STEPS="${SAVE_STEPS:-2500}"
HF_MODEL_REPO="${HF_MODEL_REPO:-H2Ozone/alex_lever_gr00t}"
SIF="${SIF:-${SCRATCH}/alex-gr00t-train.sif}"
DATASET_PATH="${DATASET_PATH:-${SCRATCH}/alex_lever_gr00t}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/alex_lever_checkpoints}"
HF_CACHE="${HF_CACHE:-${SCRATCH}/alex_hf_cache}"

mkdir -p "${OUTPUT_DIR}" "${HF_CACHE}"
[[ -f "${SIF}" ]] || { echo "Missing ${SIF} — run apptainer pull first"; exit 1; }
[[ -d "${DATASET_PATH}/meta" ]] || { echo "Missing dataset at ${DATASET_PATH}"; exit 1; }

export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
export APPTAINERENV_HF_HOME="/cache/huggingface"
export APPTAINERENV_SKIP_DOWNLOAD=1
export APPTAINERENV_DATASET_PATH="/cache/dataset"
export APPTAINERENV_OUTPUT_DIR="/checkpoints"
export APPTAINERENV_HF_MODEL_REPO="${HF_MODEL_REPO}"
export APPTAINERENV_NUM_GPUS="${NUM_GPUS}"
export APPTAINERENV_GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}"
export APPTAINERENV_MAX_STEPS="${MAX_STEPS}"
export APPTAINERENV_SAVE_STEPS="${SAVE_STEPS}"
export APPTAINERENV_DATALOADER_WORKERS=16
export APPTAINERENV_LOW_VRAM=0

# Optional: enable wandb
# export APPTAINERENV_WANDB_API_KEY=...
# export APPTAINERENV_WANDB_MODE=online

apptainer run --nv \
  --bind "${DATASET_PATH}:/cache/dataset" \
  --bind "${HF_CACHE}:/cache/huggingface" \
  --bind "${OUTPUT_DIR}:/checkpoints" \
  "${SIF}"
