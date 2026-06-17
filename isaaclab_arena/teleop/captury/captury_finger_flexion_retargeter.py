# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Direct finger-flexion retargeter for the Ability Hand on the Captury path.

DexPilot fingertip-position retargeting is bistable for the ulnar fingers
(ring/pinky latch at a joint limit and stop tracking). Captury streams full
finger bone chains, so this retargeter maps each finger's measured bend angle
straight to its driven ``q1`` joint, overriding the four finger q1 values from
the upstream DexPilot output while passing the thumb joints through unchanged.
"""

import numpy as np

from isaacteleop.retargeting_engine.interface import BaseRetargeter, RetargeterIO, RetargeterIOType
from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType, TensorGroupType
from isaacteleop.retargeting_engine.tensor_types import FloatType, HandInput, HandInputIndex

from isaaclab_arena.teleop.captury.captury_skeleton import (
    ABILITY_FINGER_CLOSED_ANGLE_DEG,
    ABILITY_FINGER_OPEN_ANGLE_DEG,
    ABILITY_FINGER_Q1_MAX,
    captury_ability_hand_finger_q1,
)

_FINGER_NAMES = ("index", "middle", "ring", "pinky")


class CapturyFingerFlexionRetargeter(BaseRetargeter):
    """Override the four Ability Hand finger ``q1`` joints with bend-angle flexion.

    Inputs:
        * ``hand_{side}`` — Captury ``HandInput`` (26 joints) for the bend angles.
        * ``dex_joints`` — the upstream DexPilot independent-joint output, used
          for any joint this retargeter does not compute (the thumb).

    Output:
        * ``hand_joints`` — the same independent-joint layout as ``dex_joints``,
          with finger ``q1`` values replaced by the flexion mapping.
    """

    DEX_INPUT = "dex_joints"

    def __init__(
        self,
        independent_joint_names: list[str],
        hand_side: str,
        name: str,
        *,
        q1_max: float = ABILITY_FINGER_Q1_MAX,
        open_angle_deg: float = ABILITY_FINGER_OPEN_ANGLE_DEG,
        closed_angle_deg: float = ABILITY_FINGER_CLOSED_ANGLE_DEG,
    ) -> None:
        self._independent_joint_names = independent_joint_names
        self._hand_side = hand_side.lower()
        assert self._hand_side in ("left", "right"), f"hand_side must be left/right, got {hand_side!r}"
        self._hand_input_key = f"hand_{self._hand_side}"
        self._q1_max = q1_max
        self._open_angle_deg = open_angle_deg
        self._closed_angle_deg = closed_angle_deg
        # Map each output slot to the finger it represents (None = pass through dex).
        self._slot_finger: list[str | None] = []
        for joint_name in independent_joint_names:
            finger = next((f for f in _FINGER_NAMES if joint_name.endswith(f"{f}_q1")), None)
            self._slot_finger.append(finger)
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {
            self._hand_input_key: OptionalType(HandInput()),
            self.DEX_INPUT: TensorGroupType(
                "dex_independent_joints",
                [FloatType(joint_name) for joint_name in self._independent_joint_names],
            ),
        }

    def output_spec(self) -> RetargeterIOType:
        return {
            "hand_joints": TensorGroupType(
                "flexion_independent_joints",
                [FloatType(joint_name) for joint_name in self._independent_joint_names],
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        dex = inputs[self.DEX_INPUT]
        out = outputs["hand_joints"]
        hand_group = inputs[self._hand_input_key]

        finger_q1: dict[str, float] = {}
        if not hand_group.is_none:
            positions = np.from_dlpack(hand_group[HandInputIndex.JOINT_POSITIONS])
            valid = np.from_dlpack(hand_group[HandInputIndex.JOINT_VALID])
            finger_q1 = captury_ability_hand_finger_q1(
                positions,
                valid,
                q1_max=self._q1_max,
                open_angle_deg=self._open_angle_deg,
                closed_angle_deg=self._closed_angle_deg,
            )

        for i, finger in enumerate(self._slot_finger):
            if finger is not None and finger in finger_q1:
                out[i] = float(finger_q1[finger])
            else:
                # Thumb joints (and any finger missing keypoints) keep the dex value.
                out[i] = float(dex[i])
