# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
GR00T N1.6 modality config for the Alex joint-space lever datasets
(https://huggingface.co/datasets/H2Ozone/alex_lever,
https://huggingface.co/datasets/H2Ozone/lever_fingers).

Unlike ``alex_data_config.py`` (EEF wrist-pose actions resolved by the IHMC IK
streamer), this config trains directly on absolute joint-position targets:
state is 35-dim (spine 1 + arm 6x2 + ability-hand 10x2 + neck 2), action is
33-dim (same layout without the neck). Group slicing matches
``alex_lever_modality.json``.

The HF dataset is LeRobot v3.0; convert it to the episode-per-file layout
GR00T's loader expects with ``convert_lerobot_v3_to_gr00t.py`` first (the
training Docker entrypoint does this automatically).

Register under ``NEW_EMBODIMENT`` for finetuning::

    --embodiment-tag NEW_EMBODIMENT \\
    --modality-config-path isaaclab_arena_gr00t/embodiments/alex/alex_lever_data_config.py

Action horizon (16) must match the closed-loop policy server config if you
deploy the checkpoint back into Arena.

Note: the recorded ``left_hand``/``right_hand`` action columns are all-zero in
alex_lever (the hands were not commanded during the lever demos). The converter
fills them from the measured hand state (``--action_from_state_dims 13:33``,
the training Docker's default) so fingers carry a real action signal and the
normalization stats are not degenerate.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

ALEX_LEVER_ACTION_HORIZON = 16

_STATE_JOINT_GROUPS = [
    "spine",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "neck",
]

_ACTION_JOINT_GROUPS = [
    "spine",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
]

alex_lever_joint_space_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "zed_left_cam",
            "zed_right_cam",
        ],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=_STATE_JOINT_GROUPS,
        sin_cos_embedding_keys=_STATE_JOINT_GROUPS,
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ALEX_LEVER_ACTION_HORIZON)),
        modality_keys=_ACTION_JOINT_GROUPS,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in _ACTION_JOINT_GROUPS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(alex_lever_joint_space_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
