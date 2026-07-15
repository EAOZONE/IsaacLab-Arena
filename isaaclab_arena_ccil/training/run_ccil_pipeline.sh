#!/usr/bin/env bash
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Driver for the CCIL-BC workflow. The convert/export steps run here (torch + h5py only);
# the CCIL training steps run in the separate Python 3.8 CCIL env (see README.md) and are
# left as explicit manual steps because they depend on the CCIL repo + its config files.
#
# Usage:
#   HDF5=/datasets/alex_microwave/demo.hdf5 \
#   CCIL_DIR=/datasets/alex_microwave/ccil \
#   POLICY_PT=/path/to/CCIL/output/policy.pt \
#   bash isaaclab_arena_ccil/training/run_ccil_pipeline.sh <stage>
#
#   LeRobot / Hugging Face (e.g. H2Ozone/test_obs_new):
#   REPO_ID=H2Ozone/test_obs_new \
#   CCIL_DIR=/datasets/test_obs_new/ccil \
#   PICKLE=/datasets/test_obs_new/ccil/test_obs_new_visual.pkl \
#   bash isaaclab_arena_ccil/training/run_ccil_pipeline.sh convert_lerobot_visual
#
#   <stage> = convert | convert_visual | convert_lerobot | convert_lerobot_visual | export
#   State-only CCIL training stages run inside the CCIL repo, see README.
#   Visual BC training runs with train_visual_bc.py, see README.

set -euo pipefail

CCIL_DIR="${CCIL_DIR:-}"
PY="${PY:-/isaac-sim/python.sh}"
HERE="$(cd "$(dirname "$0")" && pwd)"

stage="${1:-convert}"
case "${stage}" in
  convert|convert_visual|export)
    HDF5="${HDF5:?Set HDF5 to the Arena demo HDF5 path}"
    CCIL_DIR="${CCIL_DIR:-$(dirname "${HDF5}")/ccil}"
    PICKLE="${PICKLE:-${CCIL_DIR}/alex_microwave.pkl}"
    META="${META:-${CCIL_DIR}/ccil_bc_meta.json}"
    ;;
  convert_lerobot|convert_lerobot_visual)
    : "${REPO_ID:=}"
    : "${LEROBOT_DIR:=}"
    if [ -z "${REPO_ID}" ] && [ -z "${LEROBOT_DIR}" ]; then
      echo "Set REPO_ID (e.g. H2Ozone/test_obs_new) or LEROBOT_DIR for ${stage}." >&2
      exit 2
    fi
    CCIL_DIR="${CCIL_DIR:?Set CCIL_DIR to the output directory for CCIL pickles}"
    mkdir -p "${CCIL_DIR}"
    if [ "${stage}" = "convert_lerobot_visual" ]; then
      PICKLE="${PICKLE:-${CCIL_DIR}/test_obs_new_visual.pkl}"
    else
      PICKLE="${PICKLE:-${CCIL_DIR}/test_obs_new.pkl}"
    fi
    META="${META:-${CCIL_DIR}/ccil_bc_meta.json}"
    ;;
esac

case "${stage}" in
  convert)
    "${PY}" "${HERE}/../data/convert_hdf5_to_ccil.py" --hdf5_file "${HDF5}" --out_file "${PICKLE}"
    ;;
  convert_visual)
    "${PY}" "${HERE}/../data/convert_hdf5_to_ccil.py" \
      --hdf5_file "${HDF5}" \
      --out_file "${PICKLE}" \
      --image_keys zed_left_cam_rgb zed_right_cam_rgb \
      --image_size 128 128
    ;;
  convert_lerobot)
    src_args=()
    [ -n "${REPO_ID}" ] && src_args+=(--repo_id "${REPO_ID}")
    [ -n "${LEROBOT_DIR}" ] && src_args+=(--lerobot_dir "${LEROBOT_DIR}")
    "${PY}" "${HERE}/../data/convert_lerobot_to_ccil.py" \
      "${src_args[@]}" \
      --out_file "${PICKLE}"
    ;;
  convert_lerobot_visual)
    src_args=()
    [ -n "${REPO_ID}" ] && src_args+=(--repo_id "${REPO_ID}")
    [ -n "${LEROBOT_DIR}" ] && src_args+=(--lerobot_dir "${LEROBOT_DIR}")
    IMAGE_KEYS="${IMAGE_KEYS:-observation.images.cam_zed_left observation.images.cam_zed_right}"
    OUTPUT_IMAGE_KEYS="${OUTPUT_IMAGE_KEYS:-zed_left_cam_rgb zed_right_cam_rgb}"
    # shellcheck disable=SC2086
    "${PY}" "${HERE}/../data/convert_lerobot_to_ccil.py" \
      "${src_args[@]}" \
      --out_file "${PICKLE}" \
      --image_keys ${IMAGE_KEYS} \
      --output_image_keys ${OUTPUT_IMAGE_KEYS} \
      --image_size 128 128
    ;;
  export)
    POLICY_PT="${POLICY_PT:?Set POLICY_PT to the TorchScript policy.pt from CCIL train_bc_policy.py}"
    "${PY}" "${HERE}/export_bc_to_torch.py" --policy_pt "${POLICY_PT}" --pickle "${PICKLE}" --out_meta "${META}"
    ;;
  *)
    echo "Unknown stage '${stage}'. Stages: convert | convert_visual | convert_lerobot | convert_lerobot_visual | export." >&2
    echo "State-only CCIL training runs in the py3.8 CCIL env; visual BC uses train_visual_bc.py. See README.md." >&2
    exit 2
    ;;
esac
