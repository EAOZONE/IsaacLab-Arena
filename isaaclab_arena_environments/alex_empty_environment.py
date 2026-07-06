# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Empty Alex environment for testing spawn positions.

Just Alex on a bare ground plane — no task objects, no success term. The spawn
pose is set from the CLI so different placements can be tried without editing
code.

View Alex holding still at a given pose (inside the container)::

    /isaac-sim/python.sh tools/view_alex_position.py --viz kit \\
        alex_empty \\
        --embodiment alex_v2_ability_hands \\
        --spawn_pos 0.0,0.0,0.94296 --spawn_yaw 90

Optionally place a USD asset at a trial pose alongside the robot::

    ... alex_empty \\
        --usd isaaclab_arena/assets/lever_sim/Lever.usd \\
        --usd_pos 0.6,0.0,0.9 --usd_yaw 90

The env also works with the usual runners (teleop.py, record_demos.py, the
GR00T playback script) — pass the same ``--spawn_pos`` / ``--spawn_yaw`` args.
"""

from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

# Default Alex spawn matches alex_teleop_sandbox (lever_eef calibration reference frame).
_DEFAULT_SPAWN_POS = (-0.4, -0.48682, 0.94296)

# Tuned lever-board pose when --usd points at Lever.usd (see Pictures/Screenshots 2026-07-04).
_LEVER_USD_DEFAULT_POS = (0.01931, -0.61124, 0.90566)
_LEVER_USD_DEFAULT_YAW = 90.0
_LEVER_USD_DEFAULT_SCALE = 0.0254

_VALID_ALEX_EMBODIMENTS = (
    "alex_pink",
    "alex_ability_hands",
    "alex_ability_hands_joint_pos",
    "alex_v2_pink",
    "alex_v2_ability_hands",
    "alex_v2_ability_hands_joint_pos",
)


@register_environment
class AlexEmptyEnvironment(ExampleEnvironmentBase):
    """Alex alone on a ground plane, spawn pose configurable from the CLI."""

    name: str = "alex_empty"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        assert args_cli.embodiment in _VALID_ALEX_EMBODIMENTS, (
            f"Invalid Alex embodiment {args_cli.embodiment}; choose one of {_VALID_ALEX_EMBODIMENTS}"
        )
        assert len(args_cli.spawn_pos) == 3, f"--spawn_pos needs 3 comma-separated values, got {args_cli.spawn_pos}"

        background = self.asset_registry.get_asset_by_name(args_cli.background)()

        scene_assets = [background]
        if args_cli.usd is not None:
            from pathlib import Path

            from isaaclab_arena.assets.object import Object

            assert len(args_cli.usd_pos) == 3, f"--usd_pos needs 3 comma-separated values, got {args_cli.usd_pos}"
            usd_stem = Path(args_cli.usd).stem.lower()
            if usd_stem == "lever" and tuple(args_cli.usd_pos) == (0.6, 0.0, 0.9):
                # Generic default; use the tuned board pose unless the caller overrides it.
                usd_pos = _LEVER_USD_DEFAULT_POS
                usd_yaw = _LEVER_USD_DEFAULT_YAW if args_cli.usd_yaw == 0.0 else args_cli.usd_yaw
                usd_scale = (
                    _LEVER_USD_DEFAULT_SCALE if args_cli.usd_scale == 1.0 else args_cli.usd_scale
                )
            else:
                usd_pos = tuple(args_cli.usd_pos)
                usd_yaw = args_cli.usd_yaw
                usd_scale = args_cli.usd_scale
            usd_half_yaw = math.radians(usd_yaw) / 2.0
            scene_assets.append(
                Object(
                    name=usd_stem.replace("(", "_").replace(")", "_"),
                    usd_path=args_cli.usd,
                    initial_pose=Pose(
                        position_xyz=usd_pos,
                        rotation_xyzw=(0.0, 0.70711, 0.70711, 0.0),
                    ),
                    scale=(usd_scale, usd_scale, usd_scale),
                )
            )

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        half_yaw = math.radians(args_cli.spawn_yaw) / 2.0
        embodiment.set_initial_pose(
            Pose(
                position_xyz=tuple(args_cli.spawn_pos),
                rotation_xyzw=(0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
            )
        )

        teleop_device = None
        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=scene_assets),
            teleop_device=teleop_device,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--teleop_device", type=str, default=None, help="e.g. captury or openxr")
        parser.add_argument("--embodiment", type=str, default="alex_v2_ability_hands")
        parser.add_argument(
            "--spawn_pos",
            type=lambda arg: [float(part) for part in arg.split(",")],
            default=list(_DEFAULT_SPAWN_POS),
            help=f"Robot spawn position x,y,z in world frame (default {','.join(str(v) for v in _DEFAULT_SPAWN_POS)}).",
        )
        parser.add_argument(
            "--spawn_yaw",
            type=float,
            default=0.0,
            help="Robot spawn yaw in degrees about world Z (default 0).",
        )
        parser.add_argument(
            "--background",
            type=str,
            default="ground_plane",
            help="Background asset (default ground_plane).",
        )
        parser.add_argument(
            "--usd",
            type=str,
            default=None,
            help="Optional USD file to place in the scene (path valid inside the container).",
        )
        parser.add_argument(
            "--usd_pos",
            type=lambda arg: [float(part) for part in arg.split(",")],
            default=[0.6, 0.0, 0.9],
            help="World position x,y,z for the --usd asset (default 0.6,0.0,0.9).",
        )
        parser.add_argument(
            "--usd_yaw",
            type=float,
            default=0.0,
            help="Yaw in degrees about world Z for the --usd asset (default 0).",
        )
        parser.add_argument(
            "--usd_scale",
            type=float,
            default=1.0,
            help="Uniform scale for the --usd asset (default 1.0; FBX exports are often in cm — try 0.01).",
        )
