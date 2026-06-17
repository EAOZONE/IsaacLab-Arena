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

"""IsaacTeleop pipeline source node that emits hand data from Captury Live."""

import logging
import numpy as np
import os
from scipy.spatial.transform import Rotation
from typing import Protocol

from isaacteleop.retargeting_engine.interface import OptionalTensorGroup
from isaacteleop.retargeting_engine.interface.base_retargeter import BaseRetargeter
from isaacteleop.retargeting_engine.interface.retargeter_core_types import (
    OutputSelector,
    RetargeterIO,
    RetargeterIOType,
)
from isaacteleop.retargeting_engine.interface.retargeter_subgraph import RetargeterSubgraph
from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType
from isaacteleop.retargeting_engine.tensor_types import HandInput, HandInputIndex

from isaaclab_arena.teleop.captury.captury_skeleton import (
    NUM_OPENXR_HAND_JOINTS,
    OPENXR_INDEX_TIP,
    OPENXR_LITTLE_TIP,
    OPENXR_MIDDLE_TIP,
    OPENXR_RING_TIP,
    OPENXR_THUMB_TIP,
    OPENXR_WRIST,
    CapturySkeletonMap,
    captury_hand_to_openxr_arrays,
    default_captury_skeleton_map,
    prepare_captury_joint_matrices,
)

logger = logging.getLogger(__name__)

# Set CAPTURY_DEBUG_FINGERS=1 (or a stride N) to log per-hand finger-spread
# metrics every N frames. Markerless finger tracking that under-reports spread
# shows up here as small wrist->fingertip / inter-fingertip distances, which is
# what drives the DexPilot ability-hand retargeting to a "stuck closed" pose.
_DEBUG_FINGERS_STRIDE = int(os.environ.get("CAPTURY_DEBUG_FINGERS", "0") or "0")


class SupportsLatestTransforms(Protocol):
    """Duck type for pose providers (satisfied by :class:`CapturyClient`)."""

    def get_latest_transforms(self) -> np.ndarray | None:
        """Return the latest (N, 6) skeleton pose, or ``None`` when unavailable."""


class CapturyHandsSource(BaseRetargeter):
    """Pipeline source node converting Captury skeleton poses to ``HandInput``.

    Drop-in replacement for isaacteleop's OpenXR ``HandsSource``: it exposes
    the same ``LEFT``/``RIGHT`` output names, the same
    ``OptionalType(HandInput())`` output spec, and the same ``transformed()``
    convenience method, so existing pipeline builders work with either source.

    Unlike ``HandsSource`` this is a plain no-input leaf node (not a DeviceIO
    source), so pipelines built from it can be executed directly with
    ``execute_pipeline`` — no OpenXR session is required.

    Hand poses are emitted in "anchor space": Captury world coordinates
    (Y-up) converted to meters and optionally re-based by a fixed
    ``anchor_T_captury`` calibration transform.  Pipelines apply the usual
    ``world_T_anchor`` transform downstream to place the operator in the
    simulation world, exactly as with OpenXR hand tracking.
    """

    LEFT = "hand_left"
    RIGHT = "hand_right"

    def __init__(
        self,
        name: str,
        pose_provider: SupportsLatestTransforms,
        skeleton_map: CapturySkeletonMap | None = None,
        anchor_T_captury: np.ndarray | None = None,
        wrist_rotation_offset_xyzw: tuple[float, float, float, float] | None = None,
        euler_degrees: bool = True,
        torso_relative: bool = True,
    ):
        """Initialize the source node.

        Args:
            name: Unique name for this source node.
            pose_provider: Object returning the latest Captury skeleton pose
                as an (N, 6) array (see
                :meth:`~isaaclab_arena.teleop.captury.CapturyClient.get_latest_transforms`).
            skeleton_map: Captury-to-OpenXR joint mapping. Defaults to the
                standard Captury Live skeleton.
            anchor_T_captury: Optional (4, 4) transform [m] re-basing Captury
                world coordinates into the teleop anchor frame (e.g. to move
                the mocap studio origin). Identity when ``None``.
            wrist_rotation_offset_xyzw: Optional fixed rotation offset (XYZW
                quaternion) composed on the right of the wrist frame
                (synthesized or bone) to calibrate out a constant wrist-
                orientation offset.
            euler_degrees: Whether streamed Euler angles are in degrees.
            torso_relative: Express all joints relative to the operator torso
                before the pipeline applies ``world_T_anchor``. Enables mapping
                the tracked torso onto the robot torso each step.
        """
        self._pose_provider = pose_provider
        self._skeleton_map = skeleton_map if skeleton_map is not None else default_captury_skeleton_map()
        self._anchor_T_captury = (
            np.asarray(anchor_T_captury, dtype=np.float64) if anchor_T_captury is not None else None
        )
        self._wrist_rotation_offset_xyzw = wrist_rotation_offset_xyzw
        self._euler_degrees = euler_degrees
        self._torso_relative = torso_relative
        self._debug_frame_count = 0
        super().__init__(name)

    def input_spec(self) -> RetargeterIOType:
        """No pipeline inputs: data is pulled from the pose provider."""
        return {}

    def output_spec(self) -> RetargeterIOType:
        """Declare standard hand outputs (Optional — absent when not tracked)."""
        return {
            self.LEFT: OptionalType(HandInput()),
            self.RIGHT: OptionalType(HandInput()),
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        transforms = self._pose_provider.get_latest_transforms()
        if transforms is None:
            outputs[self.LEFT].set_none()
            outputs[self.RIGHT].set_none()
            return

        joint_matrices = prepare_captury_joint_matrices(
            transforms,
            self._skeleton_map,
            anchor_T_captury=self._anchor_T_captury,
            euler_degrees=self._euler_degrees,
            torso_relative=self._torso_relative,
        )

        for side, output_name, hand_map in (
            ("left", self.LEFT, self._skeleton_map.left),
            ("right", self.RIGHT, self._skeleton_map.right),
        ):
            group = outputs[output_name]
            if self._skeleton_map.max_joint_index >= joint_matrices.shape[0]:
                group.set_none()
                continue
            positions, orientations, valid = captury_hand_to_openxr_arrays(
                joint_matrices,
                hand_map,
                side,
                wrist_rotation_offset_xyzw=self._wrist_rotation_offset_xyzw,
            )
            if _DEBUG_FINGERS_STRIDE:
                self._log_finger_spread(side, positions, valid)
            self._write_hand_group(group, positions, orientations, valid)
        if _DEBUG_FINGERS_STRIDE:
            self._debug_frame_count += 1

    def _log_finger_spread(self, side: str, positions: np.ndarray, valid: np.ndarray) -> None:
        """Log finger-spread metrics that drive the DexPilot ability-hand grasp.

        Reports, for the keypoints DexPilot actually consumes (wrist + 5
        fingertips), the wrist->fingertip distances and the thumb->finger
        distances. Small values mean the operator's hand is reaching the
        retargeter as under-spread/curled, which produces a "stuck closed"
        robot hand regardless of how open the real hand is.
        """
        if self._debug_frame_count % _DEBUG_FINGERS_STRIDE != 0:
            return
        if positions.shape[0] < NUM_OPENXR_HAND_JOINTS or not valid.any():
            return
        tips = {
            "thumb": OPENXR_THUMB_TIP,
            "index": OPENXR_INDEX_TIP,
            "middle": OPENXR_MIDDLE_TIP,
            "ring": OPENXR_RING_TIP,
            "pinky": OPENXR_LITTLE_TIP,
        }
        wrist = positions[OPENXR_WRIST]
        reach = {name: float(np.linalg.norm(positions[idx] - wrist)) for name, idx in tips.items()}
        thumb = positions[tips["thumb"]]
        pinch = {
            name: float(np.linalg.norm(positions[tips[name]] - thumb))
            for name in ("index", "middle", "ring", "pinky")
        }
        n_valid = int(valid.sum())
        logger.info(
            "[captury fingers %-5s] valid=%2d/26 wrist->tip(cm)=%s thumb->tip(cm)=%s",
            side,
            n_valid,
            {k: round(v * 100, 1) for k, v in reach.items()},
            {k: round(v * 100, 1) for k, v in pinch.items()},
        )

    @staticmethod
    def _write_hand_group(
        group: OptionalTensorGroup,
        positions: np.ndarray,
        orientations: np.ndarray,
        valid: np.ndarray,
    ) -> None:
        if not valid.any():
            group.set_none()
            return
        group[HandInputIndex.JOINT_POSITIONS] = positions
        group[HandInputIndex.JOINT_ORIENTATIONS] = orientations
        group[HandInputIndex.JOINT_RADII] = np.zeros(positions.shape[0], dtype=np.float32)
        group[HandInputIndex.JOINT_VALID] = valid

    def transformed(self, transform_input: OutputSelector) -> RetargeterSubgraph:
        """Apply a 4x4 transform to all hand joint poses (mirrors ``HandsSource``).

        Args:
            transform_input: An OutputSelector providing a TransformMatrix
                (e.g., ``value_input.output(ValueInput.VALUE)``).

        Returns:
            A subgraph with ``hand_left`` / ``hand_right`` outputs containing
            the transformed hand data.
        """
        from isaacteleop.retargeting_engine.utilities.hand_transform import HandTransform

        xform_node = HandTransform(f"{self.name}_transform")
        return xform_node.connect({
            self.LEFT: self.output(self.LEFT),
            self.RIGHT: self.output(self.RIGHT),
            "transform": transform_input,
        })


def make_identity_anchor_transform() -> np.ndarray:
    """Identity anchor calibration transform (Captury world == anchor frame)."""
    return np.eye(4, dtype=np.float64)


def make_anchor_transform(
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """Build a simple ``anchor_T_captury`` calibration transform.

    Args:
        translation_m: Translation [m] applied after rotation.
        yaw_deg: Rotation about the Captury up axis (Y) in degrees.

    Returns:
        (4, 4) float64 transform matrix.
    """
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix()
    mat[:3, 3] = translation_m
    return mat
