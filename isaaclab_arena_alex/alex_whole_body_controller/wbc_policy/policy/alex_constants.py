# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Joint naming and default pose constants for Alex lower-body WBC."""

from __future__ import annotations

# Lower-body joints commanded by the standing controller (legs + spine).
ALEX_LOWER_BODY_JOINT_NAMES: tuple[str, ...] = (
    "LEFT_HIP_X",
    "LEFT_HIP_Z",
    "LEFT_HIP_Y",
    "LEFT_KNEE_Y",
    "LEFT_ANKLE_Y",
    "LEFT_ANKLE_X",
    "RIGHT_HIP_X",
    "RIGHT_HIP_Z",
    "RIGHT_HIP_Y",
    "RIGHT_KNEE_Y",
    "RIGHT_ANKLE_Y",
    "RIGHT_ANKLE_X",
    "SPINE_Z",
)

# Slight knee/hip flex keeps the URDF near the training height (~0.93 m pelvis). Knee
# angles use the Isaac Lab / URDF sign convention (flexion is positive).
ALEX_STANDING_NOMINAL_JOINT_POS: dict[str, float] = {
    "LEFT_HIP_Y": -0.12,
    "RIGHT_HIP_Y": -0.12,
    "LEFT_KNEE_Y": 0.22,
    "RIGHT_KNEE_Y": 0.22,
    "LEFT_ANKLE_Y": 0.10,
    "RIGHT_ANKLE_Y": 0.10,
}

# Supported lower-body policy backends.
ALEX_WBC_VERSION_STANDING_PD = "standing_pd"
ALEX_WBC_VERSION_RL = "rl"

# Upper-body joints held during standing-RL training on nubs Alex (no wrists/grippers).
ALEX_UPPER_BODY_NUBS_JOINT_NAMES: tuple[str, ...] = (
    "NECK_Z",
    "NECK_Y",
    "LEFT_SHOULDER_Y",
    "LEFT_SHOULDER_X",
    "LEFT_SHOULDER_Z",
    "LEFT_ELBOW_Y",
    "RIGHT_SHOULDER_Y",
    "RIGHT_SHOULDER_X",
    "RIGHT_SHOULDER_Z",
    "RIGHT_ELBOW_Y",
)

# Upper-body joints for ability-hands Alex (includes wrists/grippers).
ALEX_UPPER_BODY_JOINT_NAMES: tuple[str, ...] = (
    *ALEX_UPPER_BODY_NUBS_JOINT_NAMES,
    "LEFT_WRIST_Z",
    "LEFT_WRIST_X",
    "LEFT_GRIPPER_Z",
    "RIGHT_WRIST_Z",
    "RIGHT_WRIST_X",
    "RIGHT_GRIPPER_Z",
)

# Full-body nominal pose for resets (lower body uses standing crouch + arms at teleop defaults).
ALEX_STANDING_FULL_JOINT_POS: dict[str, float] = {
    **ALEX_STANDING_NOMINAL_JOINT_POS,
    "LEFT_ELBOW_Y": -1.5708,
    "RIGHT_ELBOW_Y": -1.5708,
}

# RL action post-processing must match :class:`JointPositionActionCfg` on the training embodiment.
ALEX_STANDING_RL_ACTION_SCALE = 0.25

# Target pelvis height [m] for the standing task (matches Alex spawn height).
ALEX_STANDING_TARGET_HEIGHT = 0.93

# Lower-body offsets matching training ``init_state.joint_pos`` (used for RL action decoding).
ALEX_STANDING_LOWER_BODY_OFFSET: tuple[float, ...] = tuple(
    ALEX_STANDING_FULL_JOINT_POS.get(name, 0.0) for name in ALEX_LOWER_BODY_JOINT_NAMES
)
