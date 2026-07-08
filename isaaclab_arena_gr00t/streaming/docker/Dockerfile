# Self-contained GR00T inference-backend image: runs both the GR00T model server
# (submodules/Isaac-GR00T/gr00t/eval/run_gr00t_server.py) and the openpi-protocol bridge
# (gr00t_openpi_bridge_server.py) that the Java Gr00tClient/Gr00tUpdateThread talk to, so a new
# machine only needs Docker + the NVIDIA container toolkit - no local uv/conda setup.
#
#   ./build.sh
#   docker run --rm --gpus all --shm-size 2g \
#     -e GR00T_HF_REPO=H2Ozone/lever_eef_gr00t -e GR00T_CHECKPOINT_STEP=30000 \
#     -e HF_TOKEN=hf_xxx -p 8000:8000 -v gr00t_inference_hf_cache:/cache/huggingface \
#     gr00t-inference
#
# Mirrors isaaclab_arena_gr00t/training/docker/Dockerfile's base image and uv/flash-attn build
# (gr00t hard-depends on flash-attn regardless of train vs. eval use), minus the training-only
# Arena embodiment files.

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv system-wide (not under /root) so the image also works on GPU clusters where
# Apptainer/Singularity runs the container as a non-root user.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /workspace
COPY submodules/Isaac-GR00T /workspace/Isaac-GR00T

# flash-attn compiles from source here (no GPU needed at build time, only nvcc). MAX_JOBS bounds
# the parallel nvcc jobs so the build doesn't exhaust RAM.
WORKDIR /workspace/Isaac-GR00T
ENV MAX_JOBS=4
RUN uv sync --python 3.10 && uv pip install -e . && uv pip install websockets

# At runtime `uv run` must never try to re-sync the env: the filesystem is read-only under
# Apptainer/Singularity on clusters.
ENV UV_NO_SYNC=1

COPY isaaclab_arena_gr00t/streaming/gr00t_openpi_bridge_server.py /workspace/gr00t_openpi_bridge_server.py
COPY isaaclab_arena_gr00t/streaming/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV HF_HOME=/cache/huggingface
VOLUME ["/cache/huggingface"]

ENTRYPOINT ["/entrypoint.sh"]
