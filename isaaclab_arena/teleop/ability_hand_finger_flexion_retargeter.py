# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Direct OpenXR finger-flexion retargeting for Psyonic Ability Hands."""

import numpy as np

try:
    from isaacteleop.retargeting_engine.interface import (
        BaseRetargeter,
        RetargeterIO,
        RetargeterIOType,
    )
    from isaacteleop.retargeting_engine.interface.tensor_group_type import (
        OptionalType,
        TensorGroupType,
    )
    from isaacteleop.retargeting_engine.tensor_types import (
        FloatType,
        HandInput,
        HandInputIndex,
    )
except ModuleNotFoundError as exc:
    BaseRetargeter = None
    RetargeterIO = RetargeterIOType = OptionalType = TensorGroupType = None
    FloatType = HandInput = HandInputIndex = None
    _ISAAC_TELEOP_IMPORT_ERROR = exc
else:
    _ISAAC_TELEOP_IMPORT_ERROR = None

from isaaclab_arena.teleop.captury.captury_skeleton import (
    ABILITY_FINGER_CLOSED_ANGLE_DEG,
    ABILITY_FINGER_OPEN_ANGLE_DEG,
    ABILITY_FINGER_Q1_MAX,
    OPENXR_INDEX_PROXIMAL,
    OPENXR_INDEX_TIP,
    OPENXR_LITTLE_PROXIMAL,
    OPENXR_LITTLE_TIP,
    OPENXR_MIDDLE_PROXIMAL,
    OPENXR_MIDDLE_TIP,
    OPENXR_RING_PROXIMAL,
    OPENXR_RING_TIP,
    OPENXR_THUMB_METACARPAL,
    OPENXR_THUMB_TIP,
    OPENXR_WRIST,
)

ABILITY_THUMB_Q1_OPEN = -1.74
ABILITY_THUMB_Q1_CLOSED = 0.0
_FINGER_Q1_JOINTS = ("index_q1", "middle_q1", "ring_q1", "pinky_q1")
_THUMB_Q1_JOINT = "thumb_q1"
_FLEXION_CHAINS = {
    "index": (OPENXR_INDEX_PROXIMAL, OPENXR_INDEX_TIP),
    "middle": (OPENXR_MIDDLE_PROXIMAL, OPENXR_MIDDLE_TIP),
    "ring": (OPENXR_RING_PROXIMAL, OPENXR_RING_TIP),
    "pinky": (OPENXR_LITTLE_PROXIMAL, OPENXR_LITTLE_TIP),
    "thumb": (OPENXR_THUMB_METACARPAL, OPENXR_THUMB_TIP),
}


def ability_hand_q1_from_openxr_flexion(
    positions: np.ndarray,
    valid: np.ndarray,
    *,
    finger_q1_max: float = ABILITY_FINGER_Q1_MAX,
    thumb_q1_open: float = ABILITY_THUMB_Q1_OPEN,
    thumb_q1_closed: float = ABILITY_THUMB_Q1_CLOSED,
    open_angle_deg: float = ABILITY_FINGER_OPEN_ANGLE_DEG,
    closed_angle_deg: float = ABILITY_FINGER_CLOSED_ANGLE_DEG,
) -> dict[str, float]:
    """Map OpenXR joint geometry to Ability Hand q1 targets.

    Non-thumb fingers close from 0 to ``finger_q1_max``. Thumb q1 is reversed
    by the hand model: open is negative, while closed is zero.
    """
    if not valid[OPENXR_WRIST]:
        return {}
    wrist = positions[OPENXR_WRIST]
    span = max(open_angle_deg - closed_angle_deg, 1.0e-6)
    out: dict[str, float] = {}
    for digit, (proximal, tip) in _FLEXION_CHAINS.items():
        if not (valid[proximal] and valid[tip]):
            continue
        to_wrist = wrist - positions[proximal]
        to_tip = positions[tip] - positions[proximal]
        n_w = np.linalg.norm(to_wrist)
        n_t = np.linalg.norm(to_tip)
        if n_w < 1.0e-6 or n_t < 1.0e-6:
            continue
        cos_a = float(np.clip(np.dot(to_wrist, to_tip) / (n_w * n_t), -1.0, 1.0))
        angle_deg = np.degrees(np.arccos(cos_a))
        close_fraction = float(np.clip((open_angle_deg - angle_deg) / span, 0.0, 1.0))
        if digit == "thumb":
            out[digit] = thumb_q1_open + close_fraction * (
                thumb_q1_closed - thumb_q1_open
            )
        else:
            out[digit] = close_fraction * finger_q1_max
    return out


if BaseRetargeter is not None:

    class AbilityHandFingerFlexionRetargeter(BaseRetargeter):
        """Override Ability Hand q1 joints from OpenXR bend geometry.

        DexPilot still feeds any joint not computed here, currently thumb q2. This
        keeps thumb spread/opposition available while making open-close tracking
        deterministic and sign-correct.
        """

        DEX_INPUT = "dex_joints"

        def __init__(
            self, independent_joint_names: list[str], hand_side: str, name: str
        ) -> None:
            self._independent_joint_names = independent_joint_names
            self._hand_side = hand_side.lower()
            assert self._hand_side in (
                "left",
                "right",
            ), f"hand_side must be left/right, got {hand_side!r}"
            self._hand_input_key = f"hand_{self._hand_side}"
            self._slot_digit: list[str | None] = []
            for joint_name in independent_joint_names:
                suffix = joint_name.split("_ability_hand_")[-1]
                if suffix in _FINGER_Q1_JOINTS:
                    self._slot_digit.append(suffix.removesuffix("_q1"))
                elif suffix == _THUMB_Q1_JOINT:
                    self._slot_digit.append("thumb")
                else:
                    self._slot_digit.append(None)
            super().__init__(name=name)

        def input_spec(self) -> RetargeterIOType:
            return {
                self._hand_input_key: OptionalType(HandInput()),
                self.DEX_INPUT: TensorGroupType(
                    "dex_independent_joints",
                    [
                        FloatType(joint_name)
                        for joint_name in self._independent_joint_names
                    ],
                ),
            }

        def output_spec(self) -> RetargeterIOType:
            return {
                "hand_joints": TensorGroupType(
                    "ability_hand_flexion_joints",
                    [
                        FloatType(joint_name)
                        for joint_name in self._independent_joint_names
                    ],
                )
            }

        def _compute_fn(
            self, inputs: RetargeterIO, outputs: RetargeterIO, context
        ) -> None:
            dex = inputs[self.DEX_INPUT]
            out = outputs["hand_joints"]
            hand_group = inputs[self._hand_input_key]

            q1: dict[str, float] = {}
            if not hand_group.is_none:
                positions = np.from_dlpack(hand_group[HandInputIndex.JOINT_POSITIONS])
                valid = np.from_dlpack(hand_group[HandInputIndex.JOINT_VALID])
                q1 = ability_hand_q1_from_openxr_flexion(positions, valid)

            for i, digit in enumerate(self._slot_digit):
                if digit is not None and digit in q1:
                    out[i] = float(q1[digit])
                else:
                    out[i] = float(dex[i])

else:

    class AbilityHandFingerFlexionRetargeter:
        DEX_INPUT = "dex_joints"

        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "AbilityHandFingerFlexionRetargeter requires isaacteleop"
            ) from _ISAAC_TELEOP_IMPORT_ERROR
