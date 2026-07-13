# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the CCIL behavioral-cloning policy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field


@dataclass
class CCILBCPolicyConfig:
    """Configuration for :class:`~isaaclab_arena_ccil.policy.ccil_bc_policy.CCILBCPolicy`.

    The CCIL BC model is produced offline (Python 3.8 + custom d3rlpy) and brought into
    Arena as a dependency-free artifact: either a TorchScript module saved by d3rlpy's
    ``save_policy`` (preferred, scalers baked in) or a plain ``state_dict`` + ``meta.json``
    describing the MLP and normalization (cross-version fallback).
    """

    model_path: str = field(metadata={"help": "Path to the CCIL policy artifact (TorchScript .pt or state_dict .pt)."})
    """Path to the CCIL policy artifact (TorchScript .pt or state_dict .pt)."""

    meta_path: str | None = field(
        default=None,
        metadata={"help": "Path to ccil_bc_meta.json (required only for the state_dict fallback / ref-pair checks)."},
    )
    """Path to ``ccil_bc_meta.json``; required only for the plain-MLP fallback and ref-pair verification."""

    state_key: str = field(
        default="robot_joint_pos",
        metadata={"help": "Observation key under observation['policy'] fed to the policy."},
    )
    """Observation key under ``observation['policy']`` fed to the policy (matches the converter)."""

    state_adapter: str | None = field(
        default=None,
        metadata={"help": "Optional state/action adapter for dataset-specific layouts, e.g. test_obs_new."},
    )
    """Optional state/action adapter for dataset-specific layouts."""

    num_envs: int = field(default=1, metadata={"help": "Number of environments to simulate."})
    """Number of environments to simulate."""

    policy_device: str = field(default="cuda", metadata={"help": "Device for policy tensor operations."})
    """Device for policy tensor operations."""

    use_images: bool = field(default=False, metadata={"help": "Whether the policy consumes camera observations."})
    """Whether the policy consumes camera observations in addition to state."""

    camera_group_key: str = field(default="camera_obs", metadata={"help": "Top-level observation key for cameras."})
    """Top-level observation key for camera tensors."""

    image_keys: tuple[str, ...] = field(
        default=("zed_left_cam_rgb", "zed_right_cam_rgb"),
        metadata={"help": "Camera keys under ``camera_group_key`` to feed to the visual policy."},
    )
    """Camera keys under ``camera_group_key`` to feed to the visual policy."""

    image_size: tuple[int, int] = field(default=(128, 128), metadata={"help": "Visual policy image size (H, W)."})
    """Visual policy input image size as ``(height, width)``."""

    replay_video_dataset_path: str | None = field(
        default=None,
        metadata={"help": "Optional LeRobot dataset root whose recorded videos replace sim camera observations."},
    )
    """Optional LeRobot dataset root whose recorded videos replace sim camera observations."""

    replay_video_episode: int = field(default=0, metadata={"help": "LeRobot episode used for replay video override."})
    """LeRobot episode used for replay video override."""

    replay_video_stride: int = field(default=1, metadata={"help": "Dataset video frames to advance per policy query."})
    """Dataset video frames to advance per policy query."""

    replay_video_loop: bool = field(default=False, metadata={"help": "Loop replay video when the episode ends."})
    """Loop replay video when the episode ends."""

    show_replay_video: bool = field(
        default=False,
        metadata={"help": "Show dockable Kit windows for the recorded LeRobot replay videos."},
    )
    """Show dockable Kit windows for the recorded LeRobot replay videos."""

    replay_video_keys: tuple[str, ...] = field(
        default=("cam_zed_left", "cam_zed_right"),
        metadata={"help": "LeRobot video modality keys to read from replay dataset."},
    )
    """LeRobot video modality keys to read from replay dataset."""

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> CCILBCPolicyConfig:
        """Create configuration from parsed CLI arguments."""
        return cls(
            model_path=args.model_path,
            meta_path=getattr(args, "meta_path", None),
            state_key=getattr(args, "state_key", "robot_joint_pos"),
            state_adapter=getattr(args, "state_adapter", None),
            num_envs=getattr(args, "num_envs", 1),
            policy_device=getattr(args, "policy_device", "cuda"),
            use_images=getattr(args, "use_images", False),
            camera_group_key=getattr(args, "camera_group_key", "camera_obs"),
            image_keys=tuple(getattr(args, "image_keys", ("zed_left_cam_rgb", "zed_right_cam_rgb"))),
            image_size=tuple(getattr(args, "image_size", (128, 128))),
            replay_video_dataset_path=getattr(args, "replay_video_dataset_path", None),
            replay_video_episode=getattr(args, "replay_video_episode", 0),
            replay_video_stride=getattr(args, "replay_video_stride", 1),
            replay_video_loop=getattr(args, "replay_video_loop", False),
            show_replay_video=getattr(args, "show_replay_video", False),
            replay_video_keys=tuple(getattr(args, "replay_video_keys", ("cam_zed_left", "cam_zed_right"))),
        )
