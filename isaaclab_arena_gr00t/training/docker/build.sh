#!/usr/bin/env bash
# Build the Alex GR00T fine-tuning image from a local checkout.
#
# Usage:
#   ./isaaclab_arena_gr00t/training/docker/build.sh [image-tag]
#
# On a machine without this repo, build straight from GitHub instead:
#   docker build -t alex-gr00t-train \
#     https://github.com/EAOZONE/IsaacLab-Arena.git#main:isaaclab_arena_gr00t/training/docker
#
# First build compiles flash-attn from source and can take 30+ minutes.

set -euo pipefail

IMAGE="${1:-alex-gr00t-train}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Build context is the repo root: the Dockerfile COPYs submodules/Isaac-GR00T and
# isaaclab_arena_gr00t/... paths relative to it.
DOCKER_BUILDKIT=1 docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE}" "${REPO_ROOT}"
