# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab Mimic configuration for the raw rigid-body lever task."""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS
from isaaclab_arena.tasks.rewards.lever_turn_rewards import LeverEngaged

LEVER_ENGAGED_ANGLE_THRESHOLD = 0.05
"""Lever motion in radians used to split approach from turn-and-hold."""


def make_lever_turn_subtask_obs_cfg(
    lever_object: Asset, engaged_angle_threshold: float = LEVER_ENGAGED_ANGLE_THRESHOLD
) -> ObsGroup:
    """Build the monotonic signal used for headless lever-demo annotation."""

    @configclass
    class LeverTurnSubtaskObsCfg(ObsGroup):
        lever_engaged: ObsTerm = ObsTerm(
            func=LeverEngaged,
            params={
                "object_cfg": SceneEntityCfg(lever_object.name),
                "angle_threshold": engaged_angle_threshold,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    return LeverTurnSubtaskObsCfg()


def _make_active_arm_subtasks(object_name: str) -> list[SubTaskConfig]:
    return [
        SubTaskConfig(
            object_ref=object_name,
            subtask_term_signal="lever_engaged",
            subtask_term_offset_range=(0, 5),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.003,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        ),
        SubTaskConfig(
            object_ref=object_name,
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.003,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        ),
    ]


@configclass
class LeverTurnMimicEnvCfg(MimicEnvCfg):
    """Mimic configuration for approaching, engaging, and turning a lever."""

    arm_mode: ArmMode = ArmMode.RIGHT
    lever_object_name: str = "lever_revolute"

    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "demo_src_leverturn_isaac_lab_task_D0"
        for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
            setattr(self.datagen_config, key, value)

        active_subtasks = _make_active_arm_subtasks(self.lever_object_name)
        if self.arm_mode == ArmMode.SINGLE_ARM:
            self.subtask_configs["robot"] = active_subtasks
        elif self.arm_mode in (ArmMode.LEFT, ArmMode.RIGHT):
            self.subtask_configs[self.arm_mode.value] = active_subtasks
            self.subtask_configs[self.arm_mode.get_other_arm().value] = [
                SubTaskConfig(
                    object_ref=self.lever_object_name,
                    subtask_term_signal=None,
                    subtask_term_offset_range=(0, 0),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.0,
                    num_interpolation_steps=0,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                )
            ]
        else:
            raise ValueError(f"Embodiment arm mode {self.arm_mode} not supported")
