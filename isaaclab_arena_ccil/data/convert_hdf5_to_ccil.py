# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert an Arena teleop/Mimic HDF5 dataset to the CCIL trajectory-pickle format.

CCIL (https://github.com/personalrobotics/CCIL) consumes a pickle holding a list of
trajectory dicts, each ``{"observations": (T, obs_dim), "actions": (T, act_dim)}`` as
2D float arrays (observations and actions are the *same* length T; CCIL slices the
last transition off internally).

For Alex_V2 open-microwave we use the same state/action fields as the GR00T LeRobot
pipeline so the action space is identical:

* observations := ``obs/robot_joint_pos``      (49 dof, ``alex_49dof_joint_space.yaml``)
* actions      := ``processed_actions``        (34 dof, ``alex_34dof_action_joint_space.yaml``)

``processed_actions`` are exactly what ``env.step`` consumed, so a policy trained on
them is directly env-applicable with no joint remapping.

This script depends only on ``h5py`` + ``numpy`` and runs in the Arena container.

Usage::

    /isaac-sim/python.sh isaaclab_arena_ccil/data/convert_hdf5_to_ccil.py \\
        --hdf5_file /datasets/alex_microwave/demo.hdf5 \\
        --out_file /datasets/alex_microwave/ccil/alex_microwave.pkl
"""

from __future__ import annotations

import argparse
import h5py
import numpy as np
import os
import pickle

# Default HDF5 fields, matching the GR00T LeRobot config for Alex open-microwave.
DEFAULT_STATE_KEY = "robot_joint_pos"
DEFAULT_ACTION_KEY = "processed_actions"


def _read_trajectory(demo: h5py.Group, state_key: str, action_key: str) -> dict[str, np.ndarray]:
    """Extract one trajectory's observation/action arrays from an HDF5 episode group."""
    assert "obs" in demo, "episode group has no 'obs' subgroup"
    assert state_key in demo["obs"], f"obs/{state_key} missing (have: {list(demo['obs'].keys())})"
    assert action_key in demo, f"{action_key} missing (have: {list(demo.keys())})"

    observations = np.asarray(demo["obs"][state_key][()], dtype=np.float32)
    actions = np.asarray(demo[action_key][()], dtype=np.float32)

    assert observations.ndim == 2, f"observations must be 2D, got {observations.shape}"
    assert actions.ndim == 2, f"actions must be 2D, got {actions.shape}"
    assert observations.shape[0] == actions.shape[0], (
        f"observations ({observations.shape[0]}) and actions ({actions.shape[0]}) must share length T"
    )
    return {"observations": observations, "actions": actions}


def convert(hdf5_file: str, out_file: str, state_key: str, action_key: str) -> list[dict[str, np.ndarray]]:
    """Convert an HDF5 dataset to a list of CCIL trajectory dicts and pickle it."""
    assert os.path.exists(hdf5_file), f"{hdf5_file} does not exist"

    trajectories: list[dict[str, np.ndarray]] = []
    with h5py.File(hdf5_file, "r") as f:
        assert "data" in f, "HDF5 has no top-level 'data' group"
        data = f["data"]
        # Sort numerically (demo_0, demo_1, ... demo_10) for stable, human-friendly ordering.
        demo_ids = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]) if k.split("_")[-1].isdigit() else k)
        assert demo_ids, "HDF5 'data' group has no episodes"
        for demo_id in demo_ids:
            trajectories.append(_read_trajectory(data[demo_id], state_key, action_key))

    obs_dim = trajectories[0]["observations"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]
    total_steps = sum(t["observations"].shape[0] for t in trajectories)
    for i, t in enumerate(trajectories):
        assert t["observations"].shape[1] == obs_dim, f"episode {i} obs_dim {t['observations'].shape[1]} != {obs_dim}"
        assert t["actions"].shape[1] == act_dim, f"episode {i} act_dim {t['actions'].shape[1]} != {act_dim}"
        assert np.isfinite(t["observations"]).all(), f"episode {i} has non-finite observations"
        assert np.isfinite(t["actions"]).all(), f"episode {i} has non-finite actions"

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"Wrote {len(trajectories)} trajectories ({total_steps} steps) to {out_file}")
    print(f"  observations: (T, {obs_dim})   actions: (T, {act_dim})")
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Arena HDF5 to CCIL trajectory pickle.")
    parser.add_argument("--hdf5_file", required=True, help="Input Arena HDF5 dataset.")
    parser.add_argument("--out_file", required=True, help="Output CCIL pickle path.")
    parser.add_argument("--state_key", default=DEFAULT_STATE_KEY, help="obs/<state_key> used as observations.")
    parser.add_argument("--action_key", default=DEFAULT_ACTION_KEY, help="Top-level action field used as actions.")
    args = parser.parse_args()
    convert(args.hdf5_file, args.out_file, args.state_key, args.action_key)


if __name__ == "__main__":
    main()
