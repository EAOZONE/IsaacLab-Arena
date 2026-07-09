#!/usr/bin/env bash
# Runs the GR00T inference backend (model server + openpi-protocol bridge) that the Java
# Gr00tClient/Gr00tUpdateThread (in the ihmc repo-group) talk to, via the self-contained image
# built by docker/build.sh - so bringing this up on a new machine only needs Docker + the NVIDIA
# container toolkit, not a local uv/conda environment. Meant to be launched as a single subprocess
# (e.g. by RDX's Gr00tBackendProcess): sending this script SIGTERM stops the container, whose own
# entrypoint.sh trap tears down both of its Python children.
set -euo pipefail

: "${GR00T_HF_REPO:?}"
: "${GR00T_CHECKPOINT_STEP:?}"

IMAGE="${GR00T_INFERENCE_IMAGE:-ghcr.io/eaozone/gr00t-inference:latest}"
BRIDGE_PORT="${GR00T_BRIDGE_PORT:-8000}"
CONTAINER_NAME="gr00t-backend"

# Forward optional deployment tuning only when explicitly set, so an unset value does not
# override the image defaults (notably the compiled DiT action head).
OPTIONAL_TUNING_ENV=()
for ENV_NAME in GR00T_ENABLE_TORCH_COMPILE GR00T_TORCH_COMPILE_MODE GR00T_DENOISING_STEPS; do
  if [[ -n "${!ENV_NAME:-}" ]]; then
    OPTIONAL_TUNING_ENV+=(-e "$ENV_NAME")
  fi
done

# Clean up a leftover container from a previous ungraceful shutdown before starting a new one.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

# On a new machine the image won't be present locally yet - pull it instead of requiring a manual
# build.sh run. Once pulled it's cached, so this is a no-op on subsequent starts.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker pull "$IMAGE"
fi

exec docker run --rm --gpus all --shm-size 2g \
  --name "$CONTAINER_NAME" \
  -e GR00T_HF_REPO -e GR00T_CHECKPOINT_STEP -e GR00T_MODEL_PORT -e GR00T_BRIDGE_PORT -e GR00T_TASK_DESCRIPTION \
  -e HF_TOKEN \
  "${OPTIONAL_TUNING_ENV[@]}" \
  -p "${BRIDGE_PORT}:${BRIDGE_PORT}" \
  -v gr00t_inference_hf_cache:/cache/huggingface \
  "$IMAGE"
