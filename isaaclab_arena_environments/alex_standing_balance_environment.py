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

_VALID_ALEX_STANDING_EMBODIMENTS = (
    "alex_standing_rl",
    "alex_v2_standing_rl",
    "alex_wbc_standing_rl",
    "alex_v2_wbc_standing_rl",
)


@register_environment
class AlexStandingBalanceEnvironment(ExampleEnvironmentBase):
    """RL environment for Alex in-place standing under arm disturbances."""

    name: str = "alex_standing_balance"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        import isaaclab_arena_alex.policy.alex_standing_rl_policy_cfg as alex_standing_rl_policy_cfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.alex_standing_balance_task import AlexStandingBalanceTaskRL
        from isaaclab_arena.utils.pose import Pose

        assert args_cli.embodiment in _VALID_ALEX_STANDING_EMBODIMENTS, (
            f"Invalid embodiment {args_cli.embodiment}; choose one of {_VALID_ALEX_STANDING_EMBODIMENTS}"
        )

        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0)))

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(concatenate_observation_terms=True)
        scene = Scene(assets=[ground_plane, light])
        task = AlexStandingBalanceTaskRL(embodiment=embodiment, episode_length_s=args_cli.episode_length_s)

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=None,
            rl_framework_entry_point="rsl_rl_cfg_entry_point",
            rl_policy_cfg=f"{alex_standing_rl_policy_cfg.__name__}:AlexStandingRLPolicyCfg",
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default="alex_standing_rl")
        parser.add_argument("--episode_length_s", type=float, default=10.0)
        parser.add_argument(
            "--rl_training_mode",
            action="store_true",
            help="Accepted for parity with other Arena RL envs (standing task always trains).",
        )
