# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Alex lever-turn RL environment (privileged state, no vision).

Train a teacher policy that learns to rotate the lever handle, then roll it out for
demo recording::

    /isaac-sim/python.sh submodules/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \\
        --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \\
        --task alex_lever_turn --num_envs 64 --max_iterations 4000
"""

from __future__ import annotations

import argparse
import math
import random
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import (
        IsaacLabArenaEnvironment,
    )

_DEFAULT_SPAWN_POS = (-0.4, -0.48682, 0.94296)
_DEFAULT_LEVER_USD = "isaaclab_arena/assets/lever_sim/Lever_revolute.usd"


@register_environment
class AlexLeverTurnEnvironment(ExampleEnvironmentBase):
    """Alex + lever board, trained with privileged-state RSL-RL."""

    name: str = "alex_lever_turn"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        import isaaclab_arena_examples.policy.base_rsl_rl_policy as base_rsl_rl_policy
        from isaaclab_arena.environments.isaaclab_arena_environment import (
            IsaacLabArenaEnvironment,
        )
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.lever_turn_task import LeverTurnTaskRL
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments import lever_scene_builder

        assert (
            len(args_cli.spawn_pos) == 3
        ), f"--spawn_pos needs 3 comma-separated values, got {args_cli.spawn_pos}"
        assert (
            len(args_cli.usd_pos) == 3
        ), f"--usd_pos needs 3 comma-separated values, got {args_cli.usd_pos}"

        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))

        usd_pos = tuple(args_cli.usd_pos)
        usd_yaw = args_cli.usd_yaw
        usd_scale = args_cli.usd_scale
        if tuple(args_cli.usd_pos) == (0.6, 0.0, 0.9):
            usd_pos = lever_scene_builder.LEVER_USD_DEFAULT_POS
            usd_yaw = (
                lever_scene_builder.LEVER_USD_DEFAULT_YAW
                if args_cli.usd_yaw == 0.0
                else args_cli.usd_yaw
            )
            usd_scale = (
                lever_scene_builder.LEVER_USD_DEFAULT_SCALE
                if args_cli.usd_scale == 1.0
                else args_cli.usd_scale
            )
        lever_visual_dr = bool(args_cli.lever_dr)
        lever_pose_dr = lever_visual_dr or bool(args_cli.lever_pose_dr)
        if lever_pose_dr:
            from pathlib import Path

            usd_stem = Path(args_cli.usd).stem.lower()
            if usd_stem in lever_scene_builder.LEVER_BASE_OBJECT_STEMS:
                xy_jitter = args_cli.lever_pose_dr_xy_jitter
                yaw_jitter = args_cli.lever_pose_dr_yaw_jitter_deg
                usd_pos = (
                    usd_pos[0] + random.uniform(-xy_jitter, xy_jitter),
                    usd_pos[1] + random.uniform(-xy_jitter, xy_jitter),
                    usd_pos[2],
                )
                usd_yaw = usd_yaw + random.uniform(-yaw_jitter, yaw_jitter)

        lever_assets, lever_object = lever_scene_builder.build_lever_scene_assets(
            usd_path=args_cli.usd,
            usd_pos=usd_pos,
            usd_yaw=usd_yaw,
            usd_scale=usd_scale,
            lever_dr=args_cli.lever_dr,
            table=args_cli.table,
            lever_dr_xy_jitter=args_cli.lever_pose_dr_xy_jitter,
            lever_dr_yaw_jitter_deg=args_cli.lever_pose_dr_yaw_jitter_deg,
            lever_pose_dr=lever_pose_dr,
            lever_visual_dr=lever_visual_dr,
        )

        mimic_mode = bool(args_cli.mimic)
        test_obs_new_io = bool(args_cli.test_obs_new_io)
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
            concatenate_observation_terms=not mimic_mode,
            # Mimic replays absolute wrist poses with the teleop gains. RL uses
            # softer policy gains and bounded delta-pose actions.
            use_teleop_actuators=mimic_mode or test_obs_new_io,
            use_rl_action_space=not mimic_mode and not test_obs_new_io,
            use_test_obs_new_io=test_obs_new_io,
        )
        half_yaw = math.radians(args_cli.spawn_yaw) / 2.0
        embodiment.set_initial_pose(
            Pose(
                position_xyz=tuple(args_cli.spawn_pos),
                rotation_xyzw=(0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)),
            )
        )

        scene = Scene(assets=[ground_plane, light, *lever_assets])

        task = LeverTurnTaskRL(
            lever_object=lever_object,
            embodiment=embodiment,
            episode_length_s=args_cli.episode_length_s,
            success_angle_threshold=args_cli.success_angle_threshold,
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            rl_framework_entry_point="rsl_rl_cfg_entry_point",
            rl_policy_cfg=f"{base_rsl_rl_policy.__name__}:RLPolicyCfg",
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--embodiment",
            type=str,
            default="alex_v2_ability_hands",
            help="Alex V2 ability-hands embodiment (Pink IK wrists + hand joints, fixed base).",
        )
        parser.add_argument(
            "--spawn_pos",
            type=lambda arg: [float(part) for part in arg.split(",")],
            default=list(_DEFAULT_SPAWN_POS),
            help=f"Robot spawn position x,y,z (default {','.join(str(v) for v in _DEFAULT_SPAWN_POS)}).",
        )
        parser.add_argument(
            "--spawn_yaw",
            type=float,
            default=0.0,
            help="Robot spawn yaw in degrees about world Z (default 0).",
        )
        parser.add_argument(
            "--usd",
            type=str,
            default=_DEFAULT_LEVER_USD,
            help=f"Lever board USD path (default {_DEFAULT_LEVER_USD}).",
        )
        parser.add_argument(
            "--usd_pos",
            type=lambda arg: [float(part) for part in arg.split(",")],
            default=[0.6, 0.0, 0.9],
            help="World position x,y,z for the lever board (default 0.6,0.0,0.9 — auto-tuned for lever_sim).",
        )
        parser.add_argument(
            "--usd_yaw",
            type=float,
            default=0.0,
            help="Extra yaw in degrees about world Z for the lever board (default 0).",
        )
        parser.add_argument(
            "--usd_scale",
            type=float,
            default=1.0,
            help="Uniform scale for the lever board (default 1.0; lever_sim uses 0.0254).",
        )
        parser.add_argument(
            "--lever_dr",
            action="store_true",
            help="Enable reset-time lever pose jitter and handle-color variation.",
        )
        parser.add_argument(
            "--lever_pose_dr",
            action="store_true",
            help="Enable reset-time lever pose jitter without relying on visual DR flags.",
        )
        parser.add_argument(
            "--lever_pose_dr_xy_jitter",
            type=float,
            default=0.01,
            help="Half-range for reset-time lever x/y jitter in meters (default 0.01).",
        )
        parser.add_argument(
            "--lever_pose_dr_yaw_jitter_deg",
            type=float,
            default=5.0,
            help="Half-range for reset-time lever yaw jitter in degrees (default 5).",
        )
        parser.add_argument(
            "--table",
            type=str,
            default="none",
            help="Workbench under the lever board ('seattle_lab' or 'none'). RL trains lever+robot only.",
        )
        parser.add_argument(
            "--episode_length_s",
            type=float,
            default=10.0,
            help="Episode length in seconds (default 10).",
        )
        parser.add_argument(
            "--success_angle_threshold",
            type=float,
            default=0.35,
            help="Hinge rotation (rad) from rest for success termination and sparse bonus (default 0.35).",
        )
        parser.add_argument(
            "--test_obs_new_io",
            action="store_true",
            help="Expose 48-D test_obs_new policy observations and 46-D test_obs_new actions.",
        )
