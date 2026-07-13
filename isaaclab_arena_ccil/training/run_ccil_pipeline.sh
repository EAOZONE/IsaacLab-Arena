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
#   <stage> = convert | convert_visual | export
#   State-only CCIL training stages run inside the CCIL repo, see README.
#   Visual BC training runs with train_visual_bc.py, see README.

set -euo pipefail

HDF5="${HDF5:?Set HDF5 to the Arena demo HDF5 path}"
CCIL_DIR="${CCIL_DIR:-$(dirname "${HDF5}")/ccil}"
PICKLE="${PICKLE:-${CCIL_DIR}/alex_microwave.pkl}"
META="${META:-${CCIL_DIR}/ccil_bc_meta.json}"
PY="${PY:-/isaac-sim/python.sh}"
HERE="$(cd "$(dirname "$0")" && pwd)"

stage="${1:-convert}"
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
  export)
    POLICY_PT="${POLICY_PT:?Set POLICY_PT to the TorchScript policy.pt from CCIL train_bc_policy.py}"
    "${PY}" "${HERE}/export_bc_to_torch.py" --policy_pt "${POLICY_PT}" --pickle "${PICKLE}" --out_meta "${META}"
    ;;
  *)
    echo "Unknown stage '${stage}'. Stages: convert | convert_visual | export." >&2
    echo "State-only CCIL training runs in the py3.8 CCIL env; visual BC uses train_visual_bc.py. See README.md." >&2
    exit 2
    ;;
esac
