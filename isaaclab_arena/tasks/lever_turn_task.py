# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from dataclasses import MISSING
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.common.lever_turn_mimic import (
    LeverTurnMimicEnvCfg,
    make_lever_turn_subtask_obs_cfg,
)
from isaaclab_arena.tasks.rewards.lever_turn_rewards import (
    RIGHT_ABILITY_HAND_JOINT_NAMES,
    HingeAngleFromRest,
    HingeAngleFromRestObs,
    LeverTurnedBonus,
    LeverTurnSuccess,
    grasp_readiness,
    hand_object_distance,
    lever_angular_velocity_norm,
    lever_position_in_frame,
)
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


@register_task
class LeverTurnTaskRL(TaskBase):
    """Privileged-state RL task: turn a raw rigid-body lever handle away from its reset pose."""

    def __init__(
        self,
        lever_object: Asset,
        embodiment: EmbodimentBase,
        episode_length_s: float = 10.0,
        success_angle_threshold: float = 0.35,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.lever_object = lever_object
        self.embodiment = embodiment
        self.success_angle_threshold = success_angle_threshold

        robot_name = self.embodiment.get_embodiment_name_in_scene()
        self.observation_cfg = LeverTurnObservationsCfg(
            lever_object=self.lever_object,
            robot_name=robot_name,
        )
        self.rewards_cfg = LeverTurnRewardCfg(
            lever_object=self.lever_object,
            robot_name=robot_name,
            success_angle_threshold=self.success_angle_threshold,
        )
        self.termination_cfg = self.make_rl_termination_cfg()

        self.scene_config = None
        self.events_cfg = None

    def make_rl_termination_cfg(self):
        success = TerminationTermCfg(
            func=LeverTurnSuccess,
            params={
                "object_cfg": SceneEntityCfg(self.lever_object.name),
                "angle_threshold": self.success_angle_threshold,
            },
        )
        return LeverTurnTerminationsCfg(success=success)

    def get_scene_cfg(self) -> Any:
        return self.scene_config

    def get_events_cfg(self) -> Any:
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> Any:
        return LeverTurnMimicEnvCfg(
            arm_mode=arm_mode, lever_object_name=self.lever_object.name
        )

    def get_mimic_subtask_obs_cfg(self) -> Any:
        return make_lever_turn_subtask_obs_cfg(self.lever_object)

    def get_termination_cfg(self) -> Any:
        return self.termination_cfg

    def get_observation_cfg(self) -> Any:
        return self.observation_cfg

    def get_rewards_cfg(self) -> Any:
        return self.rewards_cfg

    def get_commands_cfg(self) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.lever_object,
            offset=np.array([-1.0, -1.0, 1.2]),
        )


@configclass
class LeverTurnTerminationsCfg:
    """Termination terms for the lever-turn RL task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING


@configclass
class LeverTurnObservationsCfg:
    """Observation specifications for the lever-turn RL task."""

    task_obs: ObsGroup = MISSING

    def __init__(self, lever_object: Asset, robot_name: str):
        @configclass
        class TaskObsCfg(ObsGroup):
            hinge_angle = ObsTerm(
                func=HingeAngleFromRestObs,
                params={"object_cfg": SceneEntityCfg(lever_object.name)},
            )
            hinge_angular_velocity = ObsTerm(
                func=lever_angular_velocity_norm,
                params={"object_cfg": SceneEntityCfg(lever_object.name)},
            )
            lever_position_in_robot_frame = ObsTerm(
                func=lever_position_in_frame,
                params={
                    "root_frame_cfg": SceneEntityCfg(robot_name),
                    "object_cfg": SceneEntityCfg(lever_object.name),
                },
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.task_obs = TaskObsCfg()


@configclass
class LeverTurnRewardCfg:
    """Reward terms for the lever-turn RL task."""

    reaching_handle: RewardTermCfg = MISSING
    grasp_readiness: RewardTermCfg = MISSING
    hinge_turn_progress: RewardTermCfg = MISSING
    hinge_turned_bonus: RewardTermCfg = MISSING
    action_rate: RewardTermCfg = MISSING
    joint_vel: RewardTermCfg = MISSING

    def __init__(
        self,
        lever_object: Asset,
        robot_name: str,
        success_angle_threshold: float,
    ):
        self.reaching_handle = RewardTermCfg(
            func=hand_object_distance,
            params={
                "std": 0.1,
                "object_cfg": SceneEntityCfg(lever_object.name),
                "robot_cfg": SceneEntityCfg(
                    robot_name, body_names=["RIGHT_GRIPPER_Z_LINK"]
                ),
            },
            weight=1.0,
        )
        self.grasp_readiness = RewardTermCfg(
            func=grasp_readiness,
            params={
                "std": 0.1,
                "object_cfg": SceneEntityCfg(lever_object.name),
                "robot_cfg": SceneEntityCfg(
                    robot_name, body_names=["RIGHT_GRIPPER_Z_LINK"]
                ),
                "hand_joint_cfg": SceneEntityCfg(
                    robot_name, joint_names=RIGHT_ABILITY_HAND_JOINT_NAMES
                ),
            },
            weight=8.0,
        )
        self.hinge_turn_progress = RewardTermCfg(
            func=HingeAngleFromRest,
            params={"object_cfg": SceneEntityCfg(lever_object.name)},
            weight=5.0,
        )
        self.hinge_turned_bonus = RewardTermCfg(
            func=LeverTurnedBonus,
            params={
                "object_cfg": SceneEntityCfg(lever_object.name),
                "angle_threshold": success_angle_threshold,
            },
            weight=15.0,
        )
        self.action_rate = RewardTermCfg(
            func=mdp_isaac_lab.action_rate_l2, weight=-0.0001
        )
        self.joint_vel = RewardTermCfg(
            func=mdp_isaac_lab.joint_vel_l2,
            weight=-0.0001,
            params={"asset_cfg": SceneEntityCfg(robot_name)},
        )
