# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""In-place standing balance RL task for Alex."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena_alex.alex_env.mdp.alex_standing_rl_mdp import (
    alex_bad_orientation,
    alex_base_diverged,
    alex_lin_vel_xy_l2,
)
from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.alex_constants import (
    ALEX_LOWER_BODY_JOINT_NAMES,
    ALEX_STANDING_TARGET_HEIGHT,
)

_LOWER_BODY_SCENE = SceneEntityCfg("robot", joint_names=list(ALEX_LOWER_BODY_JOINT_NAMES))


@register_task
class AlexStandingBalanceTaskRL(TaskBase):
    """Train a lower-body policy to keep Alex upright under arm disturbances."""

    def __init__(
        self,
        embodiment: EmbodimentBase,
        episode_length_s: float = 10.0,
        minimum_height: float = 0.55,
        orientation_limit_xy: float = 0.75,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.embodiment = embodiment
        self.minimum_height = minimum_height
        self.orientation_limit_xy = orientation_limit_xy

        self.scene_config = None
        self.events_cfg = None
        self.rewards_cfg = AlexStandingBalanceRewardCfg()
        self.termination_cfg = AlexStandingBalanceTerminationsCfg()
        self.termination_cfg.base_low.params["minimum_height"] = minimum_height
        self.termination_cfg.bad_orientation.params["limit_xy"] = orientation_limit_xy

    def get_scene_cfg(self):
        return self.scene_config

    def get_events_cfg(self):
        return self.events_cfg

    def get_rewards_cfg(self):
        return self.rewards_cfg

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError("Alex standing RL task does not support mimic.")

    def get_metrics(self) -> list[MetricBase]:
        return []


@configclass
class AlexStandingBalanceRewardCfg:
    """Reward terms encouraging upright, in-place standing."""

    is_alive = RewTerm(func=mdp_isaac_lab.is_alive, weight=1.0)
    base_height = RewTerm(
        func=mdp_isaac_lab.base_height_l2,
        weight=-8.0,
        params={"target_height": ALEX_STANDING_TARGET_HEIGHT},
    )
    flat_orientation = RewTerm(func=mdp_isaac_lab.flat_orientation_l2, weight=-6.0)
    lin_vel_xy = RewTerm(func=alex_lin_vel_xy_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp_isaac_lab.ang_vel_xy_l2, weight=-0.5)
    dof_torques = RewTerm(
        func=mdp_isaac_lab.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": _LOWER_BODY_SCENE},
    )
    action_rate = RewTerm(func=mdp_isaac_lab.action_rate_l2, weight=-0.002)


@configclass
class AlexStandingBalanceTerminationsCfg:
    """Episode termination for falls and timeouts."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    base_low: TerminationTermCfg = TerminationTermCfg(
        func=mdp_isaac_lab.root_height_below_minimum,
        params={"minimum_height": 0.55, "asset_cfg": SceneEntityCfg("robot")},
    )
    bad_orientation: TerminationTermCfg = TerminationTermCfg(
        func=alex_bad_orientation,
        params={"limit_xy": 0.75, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_diverged: TerminationTermCfg = TerminationTermCfg(
        func=alex_base_diverged,
        params={"max_lin_vel": 10.0, "max_height": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
