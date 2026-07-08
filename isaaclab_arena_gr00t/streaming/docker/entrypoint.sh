#!/usr/bin/env bash
# Sequences the GR00T model server then the openpi-protocol bridge inside the container - same
# logic that used to live in the bare-metal streaming/start_gr00t_backend.sh, moved here now that
# script just does `docker run` against this image.
#
# Required env: GR00T_HF_REPO, GR00T_CHECKPOINT_STEP.
# Optional env: GR00T_MODEL_PORT (5555), GR00T_BRIDGE_PORT (8000), GR00T_TASK_DESCRIPTION ("").
set -euo pipefail

HF_REPO="${GR00T_HF_REPO:?}"
CHECKPOINT_STEP="${GR00T_CHECKPOINT_STEP:?}"
MODEL_PORT="${GR00T_MODEL_PORT:-5555}"
BRIDGE_PORT="${GR00T_BRIDGE_PORT:-8000}"
TASK_DESCRIPTION="${GR00T_TASK_DESCRIPTION:-}"

cd /workspace/Isaac-GR00T

# The image doesn't put huggingface_hub's `hf` CLI on the system PATH - it only exists inside the
# uv-managed venv here, so it must be run via `uv run` rather than bare `hf`.
MODEL_PATH="$(uv run hf download "$HF_REPO")/checkpoint-$CHECKPOINT_STEP"
echo "Resolved checkpoint: $MODEL_PATH"

uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "$MODEL_PATH" --embodiment-tag NEW_EMBODIMENT \
  --device cuda --host 127.0.0.1 --port "$MODEL_PORT" &
MODEL_PID=$!

BRIDGE_PID=""
cleanup() {
  kill "$MODEL_PID" "$BRIDGE_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Waiting for GR00T model server on 127.0.0.1:$MODEL_PORT..."
until (exec 3<>"/dev/tcp/127.0.0.1/$MODEL_PORT") 2>/dev/null; do
  if ! kill -0 "$MODEL_PID" 2>/dev/null; then
    echo "GR00T model server exited before becoming ready" >&2
    exit 1
  fi
  sleep 1
done
exec 3<&- 3>&-

uv run python /workspace/gr00t_openpi_bridge_server.py \
  --listen-host 0.0.0.0 --listen-port "$BRIDGE_PORT" \
  --gr00t-host 127.0.0.1 --gr00t-port "$MODEL_PORT" \
  --task-description "$TASK_DESCRIPTION" &
BRIDGE_PID=$!

wait -n "$MODEL_PID" "$BRIDGE_PID"
