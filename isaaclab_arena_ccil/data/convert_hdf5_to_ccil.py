# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert an Arena teleop/Mimic HDF5 dataset to the CCIL trajectory-pickle format.

CCIL (https://github.com/personalrobotics/CCIL) consumes a pickle holding a list of
trajectory dicts, each ``{"observations": (T, obs_dim), "actions": (T, act_dim)}`` as
2D float arrays (observations and actions are the *same* length T; CCIL slices the
last transition off internally). For visuomotor BC, pass ``--image_keys`` to also
include ``"images": {key: uint8 (T, 3, H, W)}``.

For Alex ability-hands open-microwave the action space is the *raw* Pink IK action —
end-effector target poses plus finger joints — rather than the post-IK joint targets:

* observations := ``obs/robot_joint_pos``  (49 dof, ``alex_49dof_joint_space.yaml``)
* actions      := ``actions``              (34 dim: left EE pose [pos 3 + quat 4],
                                            right EE pose [pos 3 + quat 4], then 20
                                            ability-hand finger joints)

``actions`` is the raw action ``env.step`` *received* (the EE targets that Pink IK / the
real-robot IK streamer then resolves to a whole-body joint solution). A policy trained on
this stream must be applied through an IK-in-the-loop embodiment (``alex_ability_hands``),
**not** the direct joint-position embodiment ``alex_ability_hands_joint_pos`` — the latter
expects ``processed_actions`` (post-IK joint targets). The finger block (last 20 dims) is
identical in both streams; only the first 14 dims change (EE poses vs. arm/wrist joints).

The state-only path depends only on ``h5py`` + ``numpy``. The optional image path also
uses ``torch`` for resize and runs in the Arena container.

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
import torch
import torch.nn.functional as F

# Default HDF5 fields. ``actions`` is the raw Pink IK action (EE target poses + finger
# joints); pass ``--action_key processed_actions`` to train on post-IK joint targets instead.
DEFAULT_STATE_KEY = "robot_joint_pos"
DEFAULT_ACTION_KEY = "actions"
DEFAULT_IMAGE_GROUP = "camera_obs"
DEFAULT_IMAGE_SIZE = (128, 128)


def _resize_rgb_frames(frames: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Convert NHWC RGB frames to resized uint8 NCHW frames."""
    assert frames.ndim == 4, f"image frames must be 4D NHWC, got {frames.shape}"
    assert frames.shape[-1] == 3, f"image frames must have 3 RGB channels, got {frames.shape}"

    tensor = torch.as_tensor(frames)
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
    else:
        tensor = tensor.float()
        if tensor.numel() > 0 and float(tensor.max()) <= 1.0:
            tensor = tensor * 255.0
    tensor = tensor.permute(0, 3, 1, 2)
    if tuple(tensor.shape[-2:]) != image_size:
        tensor = F.interpolate(tensor, size=image_size, mode="bilinear", align_corners=False)
    tensor = tensor.clamp(0, 255).round().to(torch.uint8)
    return tensor.cpu().numpy()


def _read_images(
    demo: h5py.Group,
    image_group: str,
    image_keys: list[str],
    image_size: tuple[int, int],
    trajectory_length: int,
) -> dict[str, np.ndarray]:
    """Read and align image streams for one trajectory."""
    assert image_group in demo, f"episode group has no '{image_group}' subgroup"
    images = {}
    for key in image_keys:
        assert key in demo[image_group], f"{image_group}/{key} missing (have: {list(demo[image_group].keys())})"
        frames = np.asarray(demo[image_group][key][()])
        if frames.shape[0] == trajectory_length + 1:
            frames = frames[:-1]
        assert frames.shape[0] == trajectory_length, (
            f"{image_group}/{key} length {frames.shape[0]} must match trajectory length {trajectory_length} "
            f"(or be T+1 before alignment)"
        )
        images[key] = _resize_rgb_frames(frames, image_size)
    return images


def _read_trajectory(
    demo: h5py.Group,
    state_key: str,
    action_key: str,
    image_group: str = DEFAULT_IMAGE_GROUP,
    image_keys: list[str] | None = None,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> dict[str, np.ndarray]:
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
    trajectory = {"observations": observations, "actions": actions}
    if image_keys:
        trajectory["images"] = _read_images(demo, image_group, image_keys, image_size, observations.shape[0])
    return trajectory


def convert(
    hdf5_file: str,
    out_file: str,
    state_key: str,
    action_key: str,
    image_group: str = DEFAULT_IMAGE_GROUP,
    image_keys: list[str] | None = None,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> list[dict[str, np.ndarray]]:
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
            trajectories.append(
                _read_trajectory(data[demo_id], state_key, action_key, image_group, image_keys, image_size)
            )

    obs_dim = trajectories[0]["observations"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]
    total_steps = sum(t["observations"].shape[0] for t in trajectories)
    for i, t in enumerate(trajectories):
        assert t["observations"].shape[1] == obs_dim, f"episode {i} obs_dim {t['observations'].shape[1]} != {obs_dim}"
        assert t["actions"].shape[1] == act_dim, f"episode {i} act_dim {t['actions'].shape[1]} != {act_dim}"
        assert np.isfinite(t["observations"]).all(), f"episode {i} has non-finite observations"
        assert np.isfinite(t["actions"]).all(), f"episode {i} has non-finite actions"
        if image_keys:
            assert set(t["images"].keys()) == set(image_keys), f"episode {i} image keys do not match {image_keys}"
            for key in image_keys:
                image = t["images"][key]
                assert image.shape == (t["observations"].shape[0], 3, image_size[0], image_size[1]), (
                    f"episode {i} {key} has unexpected shape {image.shape}"
                )
                assert image.dtype == np.uint8, f"episode {i} {key} must be uint8, got {image.dtype}"

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"Wrote {len(trajectories)} trajectories ({total_steps} steps) to {out_file}")
    print(f"  observations: (T, {obs_dim})   actions: (T, {act_dim})")
    if image_keys:
        print(f"  images: {image_keys} -> (T, 3, {image_size[0]}, {image_size[1]}) uint8")
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Arena HDF5 to CCIL trajectory pickle.")
    parser.add_argument("--hdf5_file", required=True, help="Input Arena HDF5 dataset.")
    parser.add_argument("--out_file", required=True, help="Output CCIL pickle path.")
    parser.add_argument("--state_key", default=DEFAULT_STATE_KEY, help="obs/<state_key> used as observations.")
    parser.add_argument("--action_key", default=DEFAULT_ACTION_KEY, help="Top-level action field used as actions.")
    parser.add_argument("--image_group", default=DEFAULT_IMAGE_GROUP, help="HDF5 group containing camera frames.")
    parser.add_argument(
        "--image_keys", nargs="*", default=None, help="Camera keys to include under trajectory['images']."
    )
    parser.add_argument(
        "--image_size",
        nargs=2,
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        metavar=("HEIGHT", "WIDTH"),
        help="Output image size for --image_keys.",
    )
    args = parser.parse_args()
    convert(
        args.hdf5_file,
        args.out_file,
        args.state_key,
        args.action_key,
        args.image_group,
        args.image_keys,
        tuple(args.image_size),
    )


if __name__ == "__main__":
    main()
