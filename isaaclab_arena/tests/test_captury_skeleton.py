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

"""Unit tests for the Captury skeleton conversion (pure numpy, no Isaac Sim)."""

import numpy as np
from scipy.spatial.transform import Rotation

import pytest

from isaaclab_arena.teleop.captury.captury_client import CapturyClient
from isaaclab_arena.teleop.captury.captury_skeleton import (
    DEFAULT_CAPTURY_JOINT_NAMES,
    OPENXR_INDEX_PROXIMAL,
    OPENXR_INDEX_TIP,
    OPENXR_MIDDLE_PROXIMAL,
    OPENXR_PALM,
    OPENXR_RING_PROXIMAL,
    OPENXR_THUMB_METACARPAL,
    OPENXR_WRIST,
    build_skeleton_map_from_joint_names,
    captury_arm_tracking_hints,
    captury_hand_to_openxr_arrays,
    captury_make_torso_relative,
    captury_transforms_to_matrices,
    captury_upper_arm_directions,
    default_captury_skeleton_map,
    elbow_target_in_base_frame,
    prepare_captury_joint_matrices,
    synthesize_openxr_wrist_rotation,
)


def test_default_skeleton_map_wrists():
    skeleton_map = default_captury_skeleton_map()
    assert skeleton_map.left.wrist == DEFAULT_CAPTURY_JOINT_NAMES.index("LeftHand")
    assert skeleton_map.right.wrist == DEFAULT_CAPTURY_JOINT_NAMES.index("RightHand")
    assert skeleton_map.torso == DEFAULT_CAPTURY_JOINT_NAMES.index("Spine3")
    # The standard skeleton has no fingers.
    assert skeleton_map.left.fingers == {}
    assert skeleton_map.right.fingers == {}


def test_skeleton_map_with_fingers():
    joint_names = [
        "Hips",
        "LeftHand",
        "LeftHandThumb1",
        "LeftHandIndex1",
        "LeftHandIndexEE",
        "LeftHandMiddle1",
        "LeftHandRing1",
        "RightHand",
        "RightHandPinky2",
    ]
    skeleton_map = build_skeleton_map_from_joint_names(joint_names)
    assert skeleton_map.left.wrist == 1
    assert skeleton_map.left.fingers[OPENXR_THUMB_METACARPAL] == 2
    assert skeleton_map.left.fingers[OPENXR_INDEX_PROXIMAL] == 3
    assert skeleton_map.left.fingers[OPENXR_INDEX_TIP] == 4
    assert skeleton_map.left.fingers[OPENXR_MIDDLE_PROXIMAL] == 5
    assert skeleton_map.left.fingers[OPENXR_RING_PROXIMAL] == 6
    assert skeleton_map.right.wrist == 7
    assert len(skeleton_map.right.fingers) == 1


def test_skeleton_map_missing_wrist_raises():
    with pytest.raises(ValueError):
        build_skeleton_map_from_joint_names(["Hips", "Spine", "LeftHand"])


def test_transforms_to_matrices_units_and_rotation():
    # One joint: translation 1000 mm along X, rotation 90 deg about Y.
    transforms = np.array([[1000.0, 0.0, 0.0, 0.0, 90.0, 0.0]])
    matrices = captury_transforms_to_matrices(transforms)
    assert matrices.shape == (1, 4, 4)
    np.testing.assert_allclose(matrices[0, :3, 3], [1.0, 0.0, 0.0], atol=1e-12)
    # 90 deg about Y maps +X to -Z.
    np.testing.assert_allclose(matrices[0, :3, :3] @ [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], atol=1e-12)


def _flat_hand_positions(side: str) -> dict[str, np.ndarray]:
    """Synthetic hand in Y-up world: palm down, fingers pointing -Z."""
    thumb_side = -1.0 if side == "right" else 1.0
    return {
        "wrist": np.array([0.0, 1.0, 0.0]),
        "middle": np.array([0.0, 1.0, -0.1]),
        "index": np.array([0.02 * thumb_side, 1.0, -0.1]),
        "ring": np.array([-0.02 * thumb_side, 1.0, -0.1]),
    }


@pytest.mark.parametrize("side", ["left", "right"])
def test_synthesized_wrist_frame_is_openxr_convention(side):
    hand = _flat_hand_positions(side)
    rotation = synthesize_openxr_wrist_rotation(hand["wrist"], hand["middle"], hand["index"], hand["ring"], side)
    # Right-handed orthonormal frame.
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(rotation) > 0.99
    # +Z (backward) points from the fingers toward the wrist: world +Z here.
    np.testing.assert_allclose(rotation[:, 2], [0.0, 0.0, 1.0], atol=1e-9)
    # +Y (dorsal) points out of the back of the hand: world +Y here.
    np.testing.assert_allclose(rotation[:, 1], [0.0, 1.0, 0.0], atol=1e-9)


def test_hand_arrays_with_fingers():
    joint_names = [
        "RightHand",
        "RightHandMiddle1",
        "RightHandIndex1",
        "RightHandRing1",
    ]
    skeleton_map = build_skeleton_map_from_joint_names(
        ["LeftHand"] + joint_names  # left wrist needed for map construction
    )
    hand = _flat_hand_positions("right")
    transforms = np.zeros((5, 6))
    transforms[0, 0:3] = [0.0, 0.0, 0.0]  # LeftHand (unused)
    transforms[1, 0:3] = hand["wrist"] * 1000.0
    transforms[2, 0:3] = hand["middle"] * 1000.0
    transforms[3, 0:3] = hand["index"] * 1000.0
    transforms[4, 0:3] = hand["ring"] * 1000.0

    matrices = captury_transforms_to_matrices(transforms)
    positions, orientations, valid = captury_hand_to_openxr_arrays(matrices, skeleton_map.right, "right")

    assert valid[OPENXR_WRIST] == 1
    assert valid[OPENXR_PALM] == 1
    assert valid[OPENXR_MIDDLE_PROXIMAL] == 1
    np.testing.assert_allclose(positions[OPENXR_WRIST], hand["wrist"], atol=1e-6)
    # Palm is between the wrist and the middle finger reference.
    np.testing.assert_allclose(positions[OPENXR_PALM], 0.5 * (hand["wrist"] + hand["middle"]), atol=1e-6)
    # Synthesized wrist frame for this flat hand is the identity.
    np.testing.assert_allclose(orientations[OPENXR_WRIST], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    # Quaternions are normalized.
    np.testing.assert_allclose(np.linalg.norm(orientations[OPENXR_WRIST]), 1.0, atol=1e-6)


def test_wrist_rotation_offset_applies_with_fingers():
    # The synthesized wrist frame for a flat hand is the identity, so a fixed
    # offset must appear verbatim on the wrist quaternion. Regression guard: the
    # offset was previously ignored whenever finger joints were present.
    skeleton_map = build_skeleton_map_from_joint_names(
        ["LeftHand", "RightHand", "RightHandMiddle1", "RightHandIndex1", "RightHandRing1"]
    )
    hand = _flat_hand_positions("right")
    transforms = np.zeros((5, 6))
    transforms[1, 0:3] = hand["wrist"] * 1000.0
    transforms[2, 0:3] = hand["middle"] * 1000.0
    transforms[3, 0:3] = hand["index"] * 1000.0
    transforms[4, 0:3] = hand["ring"] * 1000.0
    matrices = captury_transforms_to_matrices(transforms)

    offset = Rotation.from_euler("z", 30.0, degrees=True).as_quat()  # XYZW
    _, orientations, valid = captury_hand_to_openxr_arrays(
        matrices, skeleton_map.right, "right", wrist_rotation_offset_xyzw=tuple(offset)
    )
    assert valid[OPENXR_WRIST] == 1
    # Identity synthesized frame composed with the offset == the offset itself.
    result = Rotation.from_quat(orientations[OPENXR_WRIST])
    assert (result * Rotation.from_quat(offset).inv()).magnitude() < 1e-6


def test_hand_arrays_without_fingers_marks_fingers_invalid():
    skeleton_map = default_captury_skeleton_map()
    transforms = np.zeros((len(DEFAULT_CAPTURY_JOINT_NAMES), 6))
    transforms[skeleton_map.left.wrist, 0:3] = [100.0, 1200.0, -300.0]
    matrices = captury_transforms_to_matrices(transforms)

    positions, orientations, valid = captury_hand_to_openxr_arrays(matrices, skeleton_map.left, "left")

    assert valid[OPENXR_WRIST] == 1
    assert valid[OPENXR_PALM] == 1
    assert valid[OPENXR_MIDDLE_PROXIMAL] == 0
    assert valid[2:].sum() == 0
    np.testing.assert_allclose(positions[OPENXR_WRIST], [0.1, 1.2, -0.3], atol=1e-9)


def test_default_skeleton_map_has_upper_arm():
    skeleton_map = default_captury_skeleton_map()
    assert skeleton_map.left.shoulder == DEFAULT_CAPTURY_JOINT_NAMES.index("LeftArm")
    assert skeleton_map.left.elbow == DEFAULT_CAPTURY_JOINT_NAMES.index("LeftForeArm")
    assert skeleton_map.right.shoulder == DEFAULT_CAPTURY_JOINT_NAMES.index("RightArm")
    assert skeleton_map.right.elbow == DEFAULT_CAPTURY_JOINT_NAMES.index("RightForeArm")
    assert skeleton_map.left.has_upper_arm and skeleton_map.right.has_upper_arm


def test_skeleton_map_without_arm_joints():
    skeleton_map = build_skeleton_map_from_joint_names(["LeftHand", "RightHand"])
    assert not skeleton_map.left.has_upper_arm
    assert not skeleton_map.right.has_upper_arm


def test_upper_arm_directions():
    skeleton_map = default_captury_skeleton_map()
    transforms = np.zeros((len(DEFAULT_CAPTURY_JOINT_NAMES), 6))
    transforms[skeleton_map.left.shoulder, 0:3] = [200.0, 1400.0, 0.0]  # mm
    transforms[skeleton_map.left.elbow, 0:3] = [200.0, 1100.0, 0.0]  # 300mm down -> -Y
    transforms[skeleton_map.right.shoulder, 0:3] = [-200.0, 1400.0, 0.0]
    transforms[skeleton_map.right.elbow, 0:3] = [-200.0, 1400.0, 300.0]  # -> +Z
    matrices = captury_transforms_to_matrices(transforms)
    directions = captury_upper_arm_directions(matrices, skeleton_map)
    np.testing.assert_allclose(directions["left"], [0.0, -1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(directions["right"], [0.0, 0.0, 1.0], atol=1e-9)


def test_arm_tracking_hints_include_forearm_orientation():
    skeleton_map = default_captury_skeleton_map()
    transforms = np.zeros((len(DEFAULT_CAPTURY_JOINT_NAMES), 6))
    transforms[skeleton_map.left.shoulder, 0:3] = [200.0, 1400.0, 0.0]
    transforms[skeleton_map.left.elbow, 0:3] = [200.0, 1100.0, 0.0]
    transforms[skeleton_map.left.elbow, 3:6] = [0.0, 45.0, 0.0]
    transforms[skeleton_map.left.wrist, 0:3] = [200.0, 900.0, 200.0]
    transforms[skeleton_map.right.shoulder, 0:3] = [-200.0, 1400.0, 0.0]
    transforms[skeleton_map.right.elbow, 0:3] = [-200.0, 1100.0, 0.0]
    transforms[skeleton_map.right.wrist, 0:3] = [-200.0, 900.0, 0.0]
    matrices = captury_transforms_to_matrices(transforms)
    hints = captury_arm_tracking_hints(matrices, skeleton_map)
    directions = captury_upper_arm_directions(matrices, skeleton_map)
    assert hints["left"] is not None
    assert hints["right"] is not None
    np.testing.assert_allclose(hints["left"].upper_direction, directions["left"], atol=1e-9)
    assert hints["left"].elbow_orientation.shape == (3, 3)
    assert np.linalg.det(hints["left"].elbow_orientation) > 0.99


def test_upper_arm_directions_none_when_absent():
    skeleton_map = build_skeleton_map_from_joint_names(["LeftHand", "RightHand"])
    matrices = captury_transforms_to_matrices(np.zeros((2, 6)))
    directions = captury_upper_arm_directions(matrices, skeleton_map)
    assert directions["left"] is None and directions["right"] is None


def test_elbow_target_in_base_frame_uses_robot_arm_length():
    # Robot: shoulder at (0.2, 0, 1.4), elbow 0.3 m below -> robot upper-arm = 0.3 m.
    shoulder_w = np.array([0.2, 0.0, 1.4])
    elbow_w = np.array([0.2, 0.0, 1.1])
    pelvis_w = np.array([0.0, 0.0, 1.0])
    pelvis_rot = np.eye(3)
    # Operator arm points along +x (and much longer than the robot's).
    direction_w = np.array([1.0, 0.0, 0.0])

    target = elbow_target_in_base_frame(shoulder_w, elbow_w, pelvis_w, pelvis_rot, direction_w)
    # shoulder_in_base = shoulder_w - pelvis_w = (0.2, 0, 0.4); + 0.3 m along +x.
    np.testing.assert_allclose(target, [0.5, 0.0, 0.4], atol=1e-9)


def test_elbow_target_respects_base_rotation():
    shoulder_w = np.array([0.2, 0.0, 1.4])
    elbow_w = np.array([0.2, 0.0, 1.0])  # robot upper-arm = 0.4 m
    pelvis_w = np.array([0.0, 0.0, 1.0])
    # Pelvis yawed +90 deg about z: world +x maps to base -y.
    pelvis_rot = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    direction_w = np.array([1.0, 0.0, 0.0])

    target = elbow_target_in_base_frame(shoulder_w, elbow_w, pelvis_w, pelvis_rot, direction_w)
    shoulder_in_base = pelvis_rot.T @ (shoulder_w - pelvis_w)
    expected = shoulder_in_base + 0.4 * (pelvis_rot.T @ direction_w)
    np.testing.assert_allclose(target, expected, atol=1e-9)
    # World +x -> base -y for a +90deg yaw.
    np.testing.assert_allclose(pelvis_rot.T @ direction_w, [0.0, -1.0, 0.0], atol=1e-9)


def test_full_skeleton_parses_with_fingers_and_arms():
    from isaaclab_arena.teleop.captury.captury_skeleton import (
        CAPTURY_FULL_SKELETON_JOINT_NAMES,
        captury_joint_names_for_count,
    )

    assert len(CAPTURY_FULL_SKELETON_JOINT_NAMES) == 76
    m = build_skeleton_map_from_joint_names(CAPTURY_FULL_SKELETON_JOINT_NAMES)
    idx = CAPTURY_FULL_SKELETON_JOINT_NAMES.index
    assert m.left.wrist == idx("LeftHand") and m.right.wrist == idx("RightHand")
    assert m.torso == idx("Spine3")
    assert m.left.shoulder == idx("LeftArm") and m.left.elbow == idx("LeftForeArm")
    assert m.left.has_upper_arm and m.right.has_upper_arm
    assert len(m.left.fingers) == 20 and len(m.right.fingers) == 20
    # Twist joints must not be mistaken for the shoulder/elbow.
    assert m.left.shoulder != idx("LeftArmTwist")
    # Count-based selection returns this skeleton for 76 joints.
    assert captury_joint_names_for_count(76) is CAPTURY_FULL_SKELETON_JOINT_NAMES
    assert captury_joint_names_for_count(29) is not None
    assert captury_joint_names_for_count(123) is None


def test_torso_relative_places_torso_at_origin():
    skeleton_map = default_captury_skeleton_map()
    transforms = np.zeros((len(DEFAULT_CAPTURY_JOINT_NAMES), 6))
    transforms[skeleton_map.torso, 0:3] = [100.0, 1500.0, -200.0]
    transforms[skeleton_map.left.wrist, 0:3] = [400.0, 1100.0, -400.0]

    matrices = prepare_captury_joint_matrices(transforms, skeleton_map, torso_relative=True)
    np.testing.assert_allclose(matrices[skeleton_map.torso, :3, 3], [0.0, 0.0, 0.0], atol=1e-9)
    # Wrist offset from torso is preserved in the torso frame.
    np.testing.assert_allclose(matrices[skeleton_map.left.wrist, :3, 3], [0.3, -0.4, -0.2], atol=1e-9)


def test_torso_relative_disabled_keeps_world_offsets():
    skeleton_map = default_captury_skeleton_map()
    transforms = np.zeros((len(DEFAULT_CAPTURY_JOINT_NAMES), 6))
    transforms[skeleton_map.torso, 0:3] = [100.0, 1500.0, -200.0]

    matrices = prepare_captury_joint_matrices(transforms, skeleton_map, torso_relative=False)
    np.testing.assert_allclose(matrices[skeleton_map.torso, :3, 3], [0.1, 1.5, -0.2], atol=1e-9)


def test_captury_make_torso_relative():
    skeleton_map = default_captury_skeleton_map()
    transforms = np.zeros((len(DEFAULT_CAPTURY_JOINT_NAMES), 6))
    transforms[skeleton_map.torso, 0:3] = [0.0, 1000.0, 0.0]
    matrices = captury_transforms_to_matrices(transforms)
    relative = captury_make_torso_relative(matrices, skeleton_map.torso)
    np.testing.assert_allclose(relative[skeleton_map.torso], np.eye(4), atol=1e-9)


def test_captury_client_pose_buffer():
    client = CapturyClient(host="unused")
    assert client.get_latest_transforms() is None

    pose = {
        "actor": 7,
        "timestamp": 123,
        "transforms": [
            {"translation": [1.0, 2.0, 3.0], "rotation": [10.0, 20.0, 30.0]},
            {"translation": [4.0, 5.0, 6.0], "rotation": [0.0, 0.0, 0.0]},
        ],
    }
    client._on_pose(7, pose)
    transforms = client.get_latest_transforms()
    assert transforms is not None and transforms.shape == (2, 6)
    np.testing.assert_allclose(transforms[0], [1.0, 2.0, 3.0, 10.0, 20.0, 30.0])

    # Poses from a different actor are ignored once an actor is selected.
    client._on_pose(
        8, {"actor": 8, "timestamp": 124, "transforms": [{"translation": [9, 9, 9], "rotation": [0, 0, 0]}]}
    )
    transforms = client.get_latest_transforms()
    assert transforms.shape == (2, 6)

    # Stale poses are treated as missing.
    client.stale_timeout_s = -1.0
    assert client.get_latest_transforms() is None
