#!/usr/bin/env bash
# Build the GR00T inference-backend image from a local checkout.
#
# Usage:
#   ./isaaclab_arena_gr00t/streaming/docker/build.sh [image-tag]
#
# First build compiles flash-attn from source and can take 30+ minutes.

set -euo pipefail

IMAGE="${1:-gr00t-inference}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Build context is the repo root: the Dockerfile COPYs submodules/Isaac-GR00T and
# isaaclab_arena_gr00t/streaming/... paths relative to it.
DOCKER_BUILDKIT=1 docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE}" "${REPO_ROOT}"
