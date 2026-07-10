# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
GR00T N1.6 modality config for the real-robot Alex EEF dataset
(https://huggingface.co/datasets/H2Ozone/test_obs_new).

State is a 38-dim layout: left/right gripper poses in the robot's world frame
using IHMC hand control frames (pos 3 + quat xyzw 4 each, scalar-last), 20
ability-hand finger joints (per-finger q1/q2, left then right), 2 neck joints,
and 2 spine joints. Action shares the same layout minus the spine (36-dim) —
spine is not a commanded stream in this dataset. Group slicing matches
``alex_test_obs_new_modality.json``.

The HF dataset is LeRobot v3.0; convert it to the episode-per-file layout
GR00T's loader expects with ``convert_lerobot_v3_to_gr00t.py`` first (the
training Docker entrypoint does this automatically).

Register under ``NEW_EMBODIMENT`` for finetuning::

    --embodiment-tag NEW_EMBODIMENT \\
    --modality-config-path isaaclab_arena_gr00t/embodiments/alex/alex_test_obs_new_data_config.py

Action horizon (16) must match the closed-loop policy server config if you
deploy the checkpoint back into Arena.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

ALEX_TEST_OBS_NEW_ACTION_HORIZON = 16

_STATE_GROUPS = [
    "left_wrist_pose",
    "right_wrist_pose",
    "left_hand",
    "right_hand",
    "neck",
    "spine",
]

_ACTION_GROUPS = [
    "left_wrist_pose",
    "right_wrist_pose",
    "left_hand",
    "right_hand",
    "neck",
]

# sin/cos embedding suits revolute joint angles; wrist poses (positions + quats)
# are kept raw.
_SIN_COS_STATE_GROUPS = ["left_hand", "right_hand", "neck", "spine"]

alex_test_obs_new_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "cam_zed_left",
            "cam_zed_right",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=_STATE_GROUPS,
        sin_cos_embedding_keys=_SIN_COS_STATE_GROUPS,
    ),
    # Wrist poses are kept as raw pos+quat (NON_EEF/DEFAULT) rather than GR00T's
    # rot6d/rotvec EEF formats so the stored representation matches the dataset and
    # what the IK streamer / sim replay consume.
    "action": ModalityConfig(
        delta_indices=list(range(ALEX_TEST_OBS_NEW_ACTION_HORIZON)),
        modality_keys=_ACTION_GROUPS,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in _ACTION_GROUPS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(alex_test_obs_new_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
