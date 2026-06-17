# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bare Alex teleop sandbox for mocap calibration and batch demo collection.

No task objects, no success term — just Alex on a background with optional
Captury or OpenXR teleop. Pair with ``record_demos.py --timed_episode_s`` to
record fixed-length takes (e.g. gesture attempts) that are all exported as
success for manual curation later.

Mount ihmc-alex-sdk inside the container::

    ./docker/run_docker.sh -m /path/to/ihmc-alex-sdk

Record timed mocap takes::

    CAPTURY_HOST=<ip> CAPTURY_VISUALIZE_SKELETON=0 \\
    /isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/record_demos.py \\
        --device cuda --viz kit --enable_cameras \\
        --dataset_file /datasets/alex_sandbox.hdf5 \\
        --num_demos 0 --timed_episode_s 5 \\
        alex_teleop_sandbox \\
        --teleop_device captury \\
        --embodiment alex_v2_ability_hands

Free-play teleop (no recording)::

    CAPTURY_HOST=<ip> /isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/teleop.py \\
        --device cuda --viz kit \\
        alex_teleop_sandbox \\
        --teleop_device captury \\
        --embodiment alex_v2_ability_hands
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

# 0° yaw spawn — matches other Captury-tuned Alex envs (torso anchoring stays aligned).
_ALEX_SPAWN_POSE = ((-0.4, -0.48682, 0.94296), (0.0, 0.0, 0.0, 1.0))

_VALID_ALEX_EMBODIMENTS = (
    "alex_pink",
    "alex_ability_hands",
    "alex_ability_hands_joint_pos",
    "alex_v2_pink",
    "alex_v2_ability_hands",
    "alex_v2_ability_hands_joint_pos",
)


@register_environment
class AlexTeleopSandboxEnvironment(ExampleEnvironmentBase):
    """Minimal Alex environment for teleop tuning and timed demo collection."""

    name: str = "alex_teleop_sandbox"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        assert args_cli.embodiment in _VALID_ALEX_EMBODIMENTS, (
            f"Invalid Alex embodiment {args_cli.embodiment}; choose one of {_VALID_ALEX_EMBODIMENTS}"
        )

        background = self.asset_registry.get_asset_by_name(args_cli.background)()

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        spawn_xyz, spawn_rot = _ALEX_SPAWN_POSE
        embodiment.set_initial_pose(Pose(position_xyz=spawn_xyz, rotation_xyzw=spawn_rot))

        teleop_device = None
        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[background]),
            teleop_device=teleop_device,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--teleop_device", type=str, default=None, help="e.g. captury or openxr")
        parser.add_argument("--embodiment", type=str, default="alex_v2_ability_hands")
        parser.add_argument(
            "--background",
            type=str,
            default="ground_plane",
            help="Background asset (default ground_plane; try packing_table or kitchen for a table surface).",
        )
