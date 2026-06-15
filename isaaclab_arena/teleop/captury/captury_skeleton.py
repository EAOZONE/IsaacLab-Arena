# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conversion from Captury skeleton poses to OpenXR-style hand arrays.

Captury Live streams one global (world-space) transform per skeleton joint:
translation in millimeters and an XYZ Euler rotation in degrees, in a Y-up
right-handed world frame.  This module converts the joints of one hand into
the 26-joint OpenXR hand layout used by isaacteleop's ``HandInput`` tensor
type (positions in meters, XYZW quaternions, validity mask), so that Captury
data can drive the same retargeting pipelines as OpenXR hand tracking.

This module deliberately depends only on numpy/scipy (no isaacteleop, no
Isaac Sim) so it can be unit-tested anywhere.
"""

import numpy as np
import re
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation

# OpenXR hand joint indices (XR_HAND_JOINT_COUNT_EXT = 26).
# Values must match isaacteleop.retargeting_engine.tensor_types.HandJointIndex.
NUM_OPENXR_HAND_JOINTS = 26
OPENXR_PALM = 0
OPENXR_WRIST = 1
OPENXR_THUMB_METACARPAL = 2
OPENXR_THUMB_PROXIMAL = 3
OPENXR_THUMB_DISTAL = 4
OPENXR_THUMB_TIP = 5
OPENXR_INDEX_METACARPAL = 6
OPENXR_INDEX_PROXIMAL = 7
OPENXR_INDEX_INTERMEDIATE = 8
OPENXR_INDEX_DISTAL = 9
OPENXR_INDEX_TIP = 10
OPENXR_MIDDLE_METACARPAL = 11
OPENXR_MIDDLE_PROXIMAL = 12
OPENXR_MIDDLE_INTERMEDIATE = 13
OPENXR_MIDDLE_DISTAL = 14
OPENXR_MIDDLE_TIP = 15
OPENXR_RING_METACARPAL = 16
OPENXR_RING_PROXIMAL = 17
OPENXR_RING_INTERMEDIATE = 18
OPENXR_RING_DISTAL = 19
OPENXR_RING_TIP = 20
OPENXR_LITTLE_METACARPAL = 21
OPENXR_LITTLE_PROXIMAL = 22
OPENXR_LITTLE_INTERMEDIATE = 23
OPENXR_LITTLE_DISTAL = 24
OPENXR_LITTLE_TIP = 25

MM_TO_M = 1.0e-3

# Captury finger chains have up to 4 segments per finger (FBX-style naming:
# e.g. "LeftHandThumb1".."LeftHandThumb3" plus an optional end-effector joint).
# OpenXR finger chains have 5 joints (metacarpal..tip); the thumb has 4.
# Segment 1 maps to the proximal joint; metacarpals are generally not present
# in Captury skeletons and are left invalid (the downstream retargeters only
# require the wrist and the joints that are marked valid).
_FINGER_SEGMENT_TO_OPENXR = {
    "thumb": {1: OPENXR_THUMB_METACARPAL, 2: OPENXR_THUMB_PROXIMAL, 3: OPENXR_THUMB_DISTAL, 4: OPENXR_THUMB_TIP},
    "index": {
        1: OPENXR_INDEX_PROXIMAL,
        2: OPENXR_INDEX_INTERMEDIATE,
        3: OPENXR_INDEX_DISTAL,
        4: OPENXR_INDEX_TIP,
    },
    "middle": {
        1: OPENXR_MIDDLE_PROXIMAL,
        2: OPENXR_MIDDLE_INTERMEDIATE,
        3: OPENXR_MIDDLE_DISTAL,
        4: OPENXR_MIDDLE_TIP,
    },
    "ring": {
        1: OPENXR_RING_PROXIMAL,
        2: OPENXR_RING_INTERMEDIATE,
        3: OPENXR_RING_DISTAL,
        4: OPENXR_RING_TIP,
    },
    "pinky": {
        1: OPENXR_LITTLE_PROXIMAL,
        2: OPENXR_LITTLE_INTERMEDIATE,
        3: OPENXR_LITTLE_DISTAL,
        4: OPENXR_LITTLE_TIP,
    },
}

# Matches FBX-style Captury joint names, e.g. "LeftHand", "RightHandThumb2",
# "LeftHandIndex3". An optional "EE"/"End" suffix marks the fingertip.
_WRIST_NAME_RE = re.compile(r"^(Left|Right)Hand$", re.IGNORECASE)
_FINGER_NAME_RE = re.compile(
    r"^(Left|Right)Hand(Thumb|Index|Middle|Ring|Pinky|Little)(\d)?(EE|End)?$",
    re.IGNORECASE,
)
# Upper-arm chain: "LeftArm" is the upper arm (its origin is the shoulder
# joint), "LeftForeArm" is the forearm (its origin is the elbow joint).
_ARM_NAME_RE = re.compile(r"^(Left|Right)(Arm|ForeArm)$", re.IGNORECASE)
# Torso reference joint for operator-to-robot anchoring (first match wins).
_TORSO_NAME_PRIORITY = ("Spine3", "Spine2", "Spine1", "Spine", "Hips", "Root")


@dataclass
class CapturyHandJointMap:
    """Mapping from one arm's Captury joints to OpenXR hand joint slots.

    Indices refer to positions in the streamed Captury transforms array (the
    order of ``CapturyActor.joints``).
    """

    wrist: int
    """Index of the hand (wrist) joint in the Captury transforms array."""

    fingers: dict[int, int] = field(default_factory=dict)
    """Maps OpenXR hand joint index -> Captury transforms array index."""

    shoulder: int | None = None
    """Index of the upper-arm (shoulder-joint) joint, or ``None`` if absent.

    Used as the origin of the shoulder->elbow direction for arm retargeting.
    """

    elbow: int | None = None
    """Index of the forearm (elbow-joint) joint, or ``None`` if absent."""

    @property
    def has_upper_arm(self) -> bool:
        """Whether both shoulder and elbow joints are mapped."""
        return self.shoulder is not None and self.elbow is not None


@dataclass
class CapturyArmTrackingHints:
    """Per-arm pose hints for teleop elbow IK (world frame, Z-up USD)."""

    upper_direction: np.ndarray
    """Unit vector shoulder -> elbow in the world frame."""

    elbow_position: np.ndarray
    """Operator elbow joint position in the world frame [m]."""

    elbow_orientation: np.ndarray
    """Operator forearm-bone rotation (3, 3) in the world frame.

    Sourced from the Captury ``ForeArm`` joint; drives elbow flexion because
    the shoulder->elbow direction alone only resolves arm swivel.
    """


@dataclass
class CapturySkeletonMap:
    """Joint maps for both arms of a Captury skeleton."""

    left: CapturyHandJointMap
    right: CapturyHandJointMap
    torso: int | None = None
    """Index of the torso reference joint (e.g. ``Spine3``), or ``None`` if absent."""

    @property
    def max_joint_index(self) -> int:
        indices = [self.left.wrist, self.right.wrist]
        indices += list(self.left.fingers.values()) + list(self.right.fingers.values())
        for arm in (self.left, self.right):
            if arm.shoulder is not None:
                indices.append(arm.shoulder)
            if arm.elbow is not None:
                indices.append(arm.elbow)
        if self.torso is not None:
            indices.append(self.torso)
        return max(indices)


# Default joint order of the standard Captury Live skeleton (no fingers).
# Verify against your actor in Captury Live; override via
# ``build_skeleton_map_from_joint_names`` if your skeleton differs.
DEFAULT_CAPTURY_JOINT_NAMES = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Spine4",
    "Neck",
    "Head",
    "HeadEE",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandEE",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandEE",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "LeftFootEE",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "RightFootEE",
]

# Full Captury Live skeleton with fingers and twist joints, in streaming order.
# Captured live from Captury_getActors (a 76-joint actor). The ``*Twist`` and
# ``*EE`` joints are present but ignored by the name parser.
CAPTURY_FULL_SKELETON_JOINT_NAMES = [
    "Root", "Hips", "Spine", "Spine1", "Spine2", "Spine3", "Spine4", "Neck", "Head", "HeadEE",
    "LeftShoulder", "LeftArm", "LeftArmTwist", "LeftForeArm", "LeftForeArmTwist", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEE",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndexEE",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddleEE",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRingEE",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinkyEE", "LeftHandEE",
    "RightShoulder", "RightArm", "RightArmTwist", "RightForeArm", "RightForeArmTwist", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEE",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndexEE",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddleEE",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRingEE",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinkyEE", "RightHandEE",
    "LeftUpLeg", "LeftUpLegTwist", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftFootEE",
    "RightUpLeg", "RightUpLegTwist", "RightLeg", "RightFoot", "RightToeBase", "RightFootEE",
]  # fmt: skip

# Built-in skeletons keyed by joint count, for auto-selecting the joint order
# from a streamed pose without querying actor metadata.
_CAPTURY_SKELETONS_BY_JOINT_COUNT = {
    len(DEFAULT_CAPTURY_JOINT_NAMES): DEFAULT_CAPTURY_JOINT_NAMES,
    len(CAPTURY_FULL_SKELETON_JOINT_NAMES): CAPTURY_FULL_SKELETON_JOINT_NAMES,
}


def captury_joint_names_for_count(joint_count: int) -> list[str] | None:
    """Return a built-in joint-name list matching a streamed skeleton's size.

    Args:
        joint_count: Number of joints (transforms) in a streamed pose.

    Returns:
        The matching joint-name list, or ``None`` if no built-in skeleton has
        that joint count (caller should fall back to an explicit
        ``captury_joint_names`` or the default).
    """
    return _CAPTURY_SKELETONS_BY_JOINT_COUNT.get(joint_count)


def build_skeleton_map_from_joint_names(joint_names: list[str]) -> CapturySkeletonMap:
    """Build a :class:`CapturySkeletonMap` by matching Captury joint names.

    Recognizes FBX-style names used by Captury skeletons: ``LeftHand`` /
    ``RightHand`` for the wrists, ``LeftArm`` / ``LeftForeArm`` for the
    upper-arm chain (shoulder and elbow joints), and
    ``<Side>Hand<Finger><Segment>`` for the finger chains (e.g.
    ``LeftHandThumb1``, ``RightHandIndex3``, ``LeftHandMiddleEE``).

    Args:
        joint_names: Joint names in streaming order (``CapturyActor.joints``).

    Returns:
        The skeleton map for both hands.

    Raises:
        ValueError: If either wrist joint cannot be found.
    """
    wrists: dict[str, int] = {}
    fingers: dict[str, dict[int, int]] = {"left": {}, "right": {}}
    shoulders: dict[str, int] = {}
    elbows: dict[str, int] = {}

    for index, name in enumerate(joint_names):
        wrist_match = _WRIST_NAME_RE.match(name)
        if wrist_match:
            wrists[wrist_match.group(1).lower()] = index
            continue
        arm_match = _ARM_NAME_RE.match(name)
        if arm_match:
            side = arm_match.group(1).lower()
            if arm_match.group(2).lower() == "arm":
                shoulders[side] = index
            else:  # "ForeArm" -> elbow joint
                elbows[side] = index
            continue
        finger_match = _FINGER_NAME_RE.match(name)
        if finger_match:
            side = finger_match.group(1).lower()
            finger = finger_match.group(2).lower()
            if finger == "little":
                finger = "pinky"
            segment_str = finger_match.group(3)
            is_end = finger_match.group(4) is not None
            segment_map = _FINGER_SEGMENT_TO_OPENXR[finger]
            if is_end:
                openxr_index = segment_map[4]
            elif segment_str is not None:
                segment = int(segment_str)
                if segment not in segment_map:
                    continue
                openxr_index = segment_map[segment]
            else:
                continue
            fingers[side][openxr_index] = index

    if "left" not in wrists or "right" not in wrists:
        raise ValueError(f"Could not find LeftHand/RightHand wrist joints in Captury joint names: {joint_names}")

    torso_index = next((joint_names.index(name) for name in _TORSO_NAME_PRIORITY if name in joint_names), None)

    return CapturySkeletonMap(
        left=CapturyHandJointMap(
            wrist=wrists["left"],
            fingers=fingers["left"],
            shoulder=shoulders.get("left"),
            elbow=elbows.get("left"),
        ),
        right=CapturyHandJointMap(
            wrist=wrists["right"],
            fingers=fingers["right"],
            shoulder=shoulders.get("right"),
            elbow=elbows.get("right"),
        ),
        torso=torso_index,
    )


def default_captury_skeleton_map() -> CapturySkeletonMap:
    """Skeleton map for the standard (finger-less) Captury Live skeleton."""
    return build_skeleton_map_from_joint_names(DEFAULT_CAPTURY_JOINT_NAMES)


def captury_make_torso_relative(joint_matrices: np.ndarray, torso_index: int) -> np.ndarray:
    """Re-express joint poses in the operator torso frame.

    After this transform the torso joint sits at the identity and all other
    joints (wrists, fingers, elbows) are relative to the operator torso.
    Composing with a robot-torso ``world_T_anchor`` then maps the operator onto
    the robot.

    Args:
        joint_matrices: (N, 4, 4) world-space joint transforms [m].
        torso_index: Index of the torso reference joint in ``joint_matrices``.

    Returns:
        (N, 4, 4) joint transforms relative to the torso.
    """
    joint_matrices = np.asarray(joint_matrices, dtype=np.float64)
    torso_inv = np.linalg.inv(joint_matrices[torso_index])
    return torso_inv @ joint_matrices


def prepare_captury_joint_matrices(
    transforms: np.ndarray,
    skeleton_map: CapturySkeletonMap,
    *,
    anchor_T_captury: np.ndarray | None = None,
    euler_degrees: bool = True,
    torso_relative: bool = True,
) -> np.ndarray:
    """Convert streamed Captury transforms to pipeline-ready joint matrices.

    Applies unit conversion, optional calibration, and (by default) re-bases all
    joints into the operator torso frame so they can be anchored to the robot
    torso each simulation step.

    Args:
        transforms: (N, 6) Captury pose array.
        skeleton_map: Skeleton joint map (provides the torso index).
        anchor_T_captury: Optional fixed calibration transform.
        euler_degrees: Whether streamed Euler angles are in degrees.
        torso_relative: When ``True`` and a torso joint is mapped, express all
            joints relative to the operator torso.

    Returns:
        (N, 4, 4) float64 joint transforms ready for hand conversion / viz.
    """
    matrices = captury_transforms_to_matrices(transforms, euler_degrees=euler_degrees)
    if anchor_T_captury is not None:
        matrices = anchor_T_captury @ matrices
    if torso_relative and skeleton_map.torso is not None and skeleton_map.torso < matrices.shape[0]:
        matrices = captury_make_torso_relative(matrices, skeleton_map.torso)
    return matrices


def captury_transforms_to_matrices(
    transforms: np.ndarray,
    euler_degrees: bool = True,
) -> np.ndarray:
    """Convert streamed Captury transforms to homogeneous matrices in meters.

    Args:
        transforms: (N, 6) array of [tx, ty, tz, rx, ry, rz] per joint, with
            translation [mm] and global XYZ Euler rotation.
        euler_degrees: Whether the Euler angles are in degrees (Captury
            default) rather than radians.

    Returns:
        (N, 4, 4) float64 array of world-space joint transforms [m].
    """
    transforms = np.asarray(transforms, dtype=np.float64)
    assert transforms.ndim == 2 and transforms.shape[1] == 6, f"Expected (N, 6) transforms, got {transforms.shape}"
    num_joints = transforms.shape[0]
    matrices = np.tile(np.eye(4, dtype=np.float64), (num_joints, 1, 1))
    matrices[:, :3, :3] = Rotation.from_euler("xyz", transforms[:, 3:6], degrees=euler_degrees).as_matrix()
    matrices[:, :3, 3] = transforms[:, 0:3] * MM_TO_M
    return matrices


def captury_upper_arm_directions(
    joint_matrices: np.ndarray,
    skeleton_map: CapturySkeletonMap,
) -> dict[str, np.ndarray | None]:
    """Compute per-arm shoulder->elbow unit direction vectors.

    These directions drive scale-invariant elbow tracking: the robot elbow is
    placed at the robot's own upper-arm length along this direction, so the
    operator's absolute arm length is irrelevant.

    Args:
        joint_matrices: (N, 4, 4) world-space joint transforms [m], as returned
            by :func:`captury_transforms_to_matrices`. Must already be expressed
            in the frame the caller wants the directions in (e.g. apply the
            anchor / world transforms before calling).
        skeleton_map: Skeleton map providing shoulder and elbow joint indices.

    Returns:
        ``{"left": dir, "right": dir}`` where each value is a (3,) float64 unit
        vector, or ``None`` for an arm whose shoulder/elbow joints are absent,
        out of range, or coincident.
    """
    num_joints = joint_matrices.shape[0]
    directions: dict[str, np.ndarray | None] = {}
    for side, arm in (("left", skeleton_map.left), ("right", skeleton_map.right)):
        if not arm.has_upper_arm or arm.shoulder >= num_joints or arm.elbow >= num_joints:
            directions[side] = None
            continue
        shoulder_pos = joint_matrices[arm.shoulder, :3, 3]
        elbow_pos = joint_matrices[arm.elbow, :3, 3]
        vec = elbow_pos - shoulder_pos
        norm = np.linalg.norm(vec)
        directions[side] = vec / norm if norm > 1.0e-6 else None
    return directions


def captury_arm_tracking_hints(
    joint_matrices: np.ndarray,
    skeleton_map: CapturySkeletonMap,
) -> dict[str, CapturyArmTrackingHints | None]:
    """Build per-arm IK hints from tracked Captury upper-arm and forearm joints.

    Args:
        joint_matrices: (N, 4, 4) joint transforms in the world frame [m].
        skeleton_map: Skeleton map with shoulder, elbow, and wrist indices.

    Returns:
        ``{"left": hints | None, "right": hints | None}``.
    """
    num_joints = joint_matrices.shape[0]
    hints: dict[str, CapturyArmTrackingHints | None] = {}
    for side, arm in (("left", skeleton_map.left), ("right", skeleton_map.right)):
        if (
            not arm.has_upper_arm
            or arm.shoulder >= num_joints
            or arm.elbow >= num_joints
            or arm.wrist >= num_joints
        ):
            hints[side] = None
            continue
        shoulder_pos = joint_matrices[arm.shoulder, :3, 3]
        elbow_pos = joint_matrices[arm.elbow, :3, 3]
        upper = elbow_pos - shoulder_pos
        upper_norm = np.linalg.norm(upper)
        if upper_norm < 1.0e-6:
            hints[side] = None
            continue
        hints[side] = CapturyArmTrackingHints(
            upper_direction=upper / upper_norm,
            elbow_position=elbow_pos.astype(np.float64),
            elbow_orientation=joint_matrices[arm.elbow, :3, :3].astype(np.float64),
        )
    return hints


def elbow_target_in_base_frame(
    shoulder_pos_w: np.ndarray,
    elbow_pos_w: np.ndarray,
    base_pos_w: np.ndarray,
    base_rot_w: np.ndarray,
    direction_w: np.ndarray,
) -> np.ndarray:
    """Place a robot elbow target in the base (e.g. pelvis) frame.

    The robot elbow is positioned at the robot's own upper-arm length along the
    operator's shoulder->elbow direction, anchored at the robot shoulder. The
    operator's absolute arm length never enters, so the result is invariant to
    operator body scale.

    Args:
        shoulder_pos_w: Robot shoulder-joint position in world [m].
        elbow_pos_w: Robot elbow-joint position in world [m]; its distance to
            the shoulder defines the (constant) robot upper-arm length.
        base_pos_w: Robot base-link (pelvis) position in world [m].
        base_rot_w: Robot base-link world rotation matrix (3, 3).
        direction_w: Operator shoulder->elbow unit direction in world.

    Returns:
        (3,) float64 elbow target position expressed in the base-link frame.
    """
    shoulder_pos_w = np.asarray(shoulder_pos_w, dtype=np.float64)
    elbow_pos_w = np.asarray(elbow_pos_w, dtype=np.float64)
    base_pos_w = np.asarray(base_pos_w, dtype=np.float64)
    base_rot_w = np.asarray(base_rot_w, dtype=np.float64)
    direction_w = np.asarray(direction_w, dtype=np.float64)

    upper_arm_len = float(np.linalg.norm(elbow_pos_w - shoulder_pos_w))
    base_rot_inv = base_rot_w.T
    shoulder_in_base = base_rot_inv @ (shoulder_pos_w - base_pos_w)
    direction_in_base = base_rot_inv @ direction_w
    return shoulder_in_base + upper_arm_len * direction_in_base


def _orthonormal_frame(z_axis: np.ndarray, y_hint: np.ndarray) -> np.ndarray:
    """Build a right-handed rotation matrix from a Z axis and an up hint."""
    z = z_axis / np.linalg.norm(z_axis)
    x = np.cross(y_hint, z)
    x_norm = np.linalg.norm(x)
    assert x_norm > 1.0e-9, "Degenerate hand geometry: up hint parallel to Z axis"
    x /= x_norm
    y = np.cross(z, x)
    frame = np.eye(3, dtype=np.float64)
    frame[:, 0] = x
    frame[:, 1] = y
    frame[:, 2] = z
    return frame


def synthesize_openxr_wrist_rotation(
    wrist_pos: np.ndarray,
    middle_ref_pos: np.ndarray,
    index_ref_pos: np.ndarray,
    ring_ref_pos: np.ndarray,
    side: str,
) -> np.ndarray:
    """Synthesize an OpenXR-convention wrist rotation from hand geometry.

    Follows the OpenXR hand-joint convention: +Z (backward) points from the
    middle finger toward the wrist, +Y points out of the back (dorsal side)
    of the hand, and +X completes the right-handed frame.  Building the frame
    from joint positions makes the wrist orientation independent of Captury's
    internal bone-frame conventions.

    Args:
        wrist_pos: Wrist joint position [m].
        middle_ref_pos: Position of a joint along the middle finger
            (metacarpal or proximal) [m].
        index_ref_pos: Position of an index finger joint [m].
        ring_ref_pos: Position of a ring finger joint [m].
        side: ``"left"`` or ``"right"``.

    Returns:
        (3, 3) rotation matrix.
    """
    distal = np.asarray(middle_ref_pos, dtype=np.float64) - np.asarray(wrist_pos, dtype=np.float64)
    lateral = np.asarray(index_ref_pos, dtype=np.float64) - np.asarray(ring_ref_pos, dtype=np.float64)
    dorsal = np.cross(distal, lateral) if side == "right" else np.cross(lateral, distal)
    dorsal_norm = np.linalg.norm(dorsal)
    assert dorsal_norm > 1.0e-9, "Degenerate hand geometry: finger reference joints are collinear with the wrist"
    return _orthonormal_frame(-distal, dorsal / dorsal_norm)


def captury_hand_to_openxr_arrays(
    joint_matrices: np.ndarray,
    hand_map: CapturyHandJointMap,
    side: str,
    wrist_rotation_offset_xyzw: tuple[float, float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one hand of a Captury pose to OpenXR ``HandInput`` arrays.

    The wrist orientation is synthesized geometrically (OpenXR convention)
    when the index, middle, and ring finger joints are available; otherwise
    the Captury wrist bone orientation is used.  In either case a fixed
    calibration offset (``wrist_rotation_offset_xyzw``) is composed on the
    right to absorb a constant rotation between the resulting wrist frame and
    the frame the downstream retargeter expects.

    Args:
        joint_matrices: (N, 4, 4) world-space joint transforms [m], as
            returned by :func:`captury_transforms_to_matrices`.
        hand_map: Captury-to-OpenXR joint mapping for this hand.
        side: ``"left"`` or ``"right"``.
        wrist_rotation_offset_xyzw: Optional fixed rotation (XYZW quaternion)
            composed on the right of the wrist frame (synthesized or bone),
            calibrating out a constant wrist-orientation offset.

    Returns:
        Tuple of ``(positions, orientations, valid)``:
        ``positions`` (26, 3) float32 [m], ``orientations`` (26, 4) float32
        XYZW quaternions, ``valid`` (26,) uint8 mask.
    """
    assert side in ("left", "right"), f"side must be 'left' or 'right', got {side!r}"
    positions = np.zeros((NUM_OPENXR_HAND_JOINTS, 3), dtype=np.float32)
    orientations = np.zeros((NUM_OPENXR_HAND_JOINTS, 4), dtype=np.float32)
    orientations[:, 3] = 1.0
    valid = np.zeros(NUM_OPENXR_HAND_JOINTS, dtype=np.uint8)

    num_joints = joint_matrices.shape[0]
    if hand_map.wrist >= num_joints:
        return positions, orientations, valid

    wrist_pos = joint_matrices[hand_map.wrist, :3, 3]

    # Finger joints: positions from the mapped Captury joints. The dex-hand
    # retargeters consume keypoint positions, so bone orientations are passed
    # through unmodified.
    for openxr_index, captury_index in hand_map.fingers.items():
        if captury_index >= num_joints:
            continue
        positions[openxr_index] = joint_matrices[captury_index, :3, 3]
        orientations[openxr_index] = Rotation.from_matrix(joint_matrices[captury_index, :3, :3]).as_quat()
        valid[openxr_index] = 1

    # Wrist orientation: geometric synthesis when possible.
    finger_refs = _wrist_frame_reference_indices(hand_map)
    if finger_refs is not None:
        middle_ref, index_ref, ring_ref = finger_refs
        wrist_rot = synthesize_openxr_wrist_rotation(
            wrist_pos,
            joint_matrices[middle_ref, :3, 3],
            joint_matrices[index_ref, :3, 3],
            joint_matrices[ring_ref, :3, 3],
            side,
        )
    else:
        wrist_rot = joint_matrices[hand_map.wrist, :3, :3]

    # Fixed calibration offset, composed on the right of the wrist frame. Applied
    # in both branches: a finger-tracked skeleton still needs it to absorb a
    # constant rotation between the synthesized OpenXR-convention frame and the
    # runtime wrist frame the downstream Se3 retargeter expects.
    if wrist_rotation_offset_xyzw is not None:
        wrist_rot = wrist_rot @ Rotation.from_quat(wrist_rotation_offset_xyzw).as_matrix()

    positions[OPENXR_WRIST] = wrist_pos
    orientations[OPENXR_WRIST] = Rotation.from_matrix(wrist_rot).as_quat()
    valid[OPENXR_WRIST] = 1

    # Palm: synthesized between the wrist and the middle finger reference
    # (or coincident with the wrist when no fingers are tracked).
    if finger_refs is not None:
        positions[OPENXR_PALM] = 0.5 * (wrist_pos + joint_matrices[finger_refs[0], :3, 3])
    else:
        positions[OPENXR_PALM] = wrist_pos
    orientations[OPENXR_PALM] = orientations[OPENXR_WRIST]
    valid[OPENXR_PALM] = 1

    return positions, orientations, valid


def _wrist_frame_reference_indices(hand_map: CapturyHandJointMap) -> tuple[int, int, int] | None:
    """Pick Captury joint indices to anchor the geometric wrist frame.

    Returns:
        ``(middle_ref, index_ref, ring_ref)`` Captury transform indices, or
        ``None`` when the required finger joints are not mapped.
    """
    middle_ref = hand_map.fingers.get(OPENXR_MIDDLE_METACARPAL, hand_map.fingers.get(OPENXR_MIDDLE_PROXIMAL))
    index_ref = hand_map.fingers.get(OPENXR_INDEX_METACARPAL, hand_map.fingers.get(OPENXR_INDEX_PROXIMAL))
    ring_ref = hand_map.fingers.get(OPENXR_RING_METACARPAL, hand_map.fingers.get(OPENXR_RING_PROXIMAL))
    if middle_ref is None or index_ref is None or ring_ref is None:
        return None
    return middle_ref, index_ref, ring_ref
