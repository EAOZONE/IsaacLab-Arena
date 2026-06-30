# Self-contained GR00T N1.6 fine-tuning image for the Alex open-microwave task.
#
# Everything is fetched from the internet at build/run time — no files from any
# particular machine are needed. On any computer with Docker + NVIDIA toolkit:
#
#   docker build -t alex-gr00t-train \
#     https://github.com/EAOZONE/IsaacLab-Arena.git#main:isaaclab_arena_gr00t/training/docker
#   docker run --gpus all --shm-size 16g -e HF_TOKEN=hf_xxx \
#     -v alex_hf_cache:/cache/huggingface -v "$PWD/checkpoints:/checkpoints" \
#     alex-gr00t-train
#
# On `docker run` the container downloads the LeRobot dataset and base model
# from HuggingFace, fine-tunes, and uploads the checkpoint back to HuggingFace.
# See README.md in this directory for details.

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv system-wide (not under /root) so the image also works on GPU
# clusters where Apptainer/Singularity runs the container as a non-root user.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Copy the local checkout of submodules/Isaac-GR00T to ensure
# local code changes (like the LoRA implementation) are included.
WORKDIR /workspace
COPY submodules/Isaac-GR00T /workspace/Isaac-GR00T

# flash-attn compiles from source here (no GPU needed at build time, only nvcc).
# MAX_JOBS bounds the parallel nvcc jobs so the build doesn't exhaust RAM.
WORKDIR /workspace/Isaac-GR00T
ENV MAX_JOBS=4
RUN uv sync --python 3.10 && uv pip install -e . && uv pip install peft

# At runtime `uv run` must never try to re-sync the env: the filesystem is
# read-only under Apptainer/Singularity on clusters.
ENV UV_NO_SYNC=1

# Copy the Arena code and entrypoint from their project-relative paths.
# We copy the entire alex embodiment folder to get modality configs and yamls.
WORKDIR /workspace/IsaacLab-Arena
COPY isaaclab_arena_gr00t/embodiments/alex /workspace/IsaacLab-Arena/isaaclab_arena_gr00t/embodiments/alex

COPY isaaclab_arena_gr00t/training/docker/upload_to_hf.py /workspace/upload_to_hf.py
COPY isaaclab_arena_gr00t/training/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# wandb is a hard dependency of the trainer; disable unless WANDB_API_KEY is set.
ENV WANDB_MODE=disabled
ENV HF_HOME=/cache/huggingface

VOLUME ["/checkpoints", "/cache/huggingface"]

ENTRYPOINT ["/entrypoint.sh"]
