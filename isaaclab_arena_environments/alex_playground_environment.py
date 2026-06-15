# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@register_environment
class AlexPlaygroundEnvironment(ExampleEnvironmentBase):
    """Bare Alex environment: just the robot on a background, no task or objects.

    Intended for refining teleoperation (e.g. Captury mocap) without task
    clutter — the robot stands at a table-height surface so you can check
    wrist/elbow/finger tracking and anchor alignment in isolation.

    Mount the ihmc-alex-sdk root so ability-hand assets resolve::

        ./docker/run_docker.sh -m /path/to/ihmc-alex-sdk

    Usage (Captury teleop, V2 ability hands)::

        CAPTURY_HOST=<ip> python isaaclab_arena/scripts/imitation_learning/teleop.py \\
            --device cpu --viz kit \\
            alex_playground \\
            --teleop_device captury \\
            --embodiment alex_v2_ability_hands

    Embodiments: ``alex_pink``, ``alex_ability_hands``, ``alex_v2_pink``,
    ``alex_v2_ability_hands``. Swap the stage with ``--background`` (e.g.
    ``table``, ``office_table_background``, ``packing_table``).
    """

    name: str = "alex_playground"

    _VALID_EMBODIMENTS = (
        "alex_pink",
        "alex_ability_hands",
        "alex_v2_pink",
        "alex_v2_ability_hands",
    )

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        assert (
            args_cli.embodiment in self._VALID_EMBODIMENTS
        ), f"Invalid Alex embodiment {args_cli.embodiment}; choose one of {self._VALID_EMBODIMENTS}"

        background = self.asset_registry.get_asset_by_name(args_cli.background)()

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(-0.40, -0.1, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

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
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--embodiment", type=str, default="alex_v2_ability_hands")
        parser.add_argument(
            "--background",
            type=str,
            default="packing_table",
            help="Background asset for the stage (e.g. packing_table, table, office_table_background).",
        )
