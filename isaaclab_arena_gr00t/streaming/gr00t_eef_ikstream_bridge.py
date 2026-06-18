# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stream a GR00T policy's end-effector action into the IHMC RDX IK streamer.

A GR00T policy trained on the Alex ability-hands EEF action space outputs, per step, a
34-dim action ``[ left_wrist_pose(7) | right_wrist_pose(7) | hand_joints(20) ]`` where each
wrist pose is ``pos(3) + quat(4)``. The IHMC ``RDXCapturyKinematicStreaming`` UI consumes
hand target poses through a ``CapturyPoseReceiver`` that listens on a UDP port and feeds the
``KinematicsStreamingToolboxInputMessage`` (whole-body IK), which resolves the arm/torso
joints on the robot. This module turns the GR00T wrist-pose output into that UDP stream — so
the policy commands *where the hands go* and the robot's IK streamer solves the rest.

Wire format (reverse-engineered from ``CapturyPoseReceiver.decode``, little-endian)::

    int64   timestamp_microseconds
    per segment in ORDER = (LEFT_HAND, RIGHT_HAND, HEAD, CHEST):
        float32 pos_x, pos_y, pos_z
        float32 quat_x, quat_y, quat_z, quat_w   # scalar-LAST (euclid Quaternion(x,y,z,w))
        float32 valid                            # > 0 -> pose is used, else ignored

Total = 8 + 4 * (8 * 4) = 136 bytes. The receiver always reads all four segments, so a
packet must include head/chest slots even when only the hands are driven (send valid=0 for
unused segments).

Note on conventions/frames:
* Isaac Lab / Arena quaternions are scalar-FIRST ``(w, x, y, z)``; the wire is scalar-LAST.
  ``send_hand_poses`` takes scalar-first quats by default and reorders them.
* The wrist poses are in whatever frame the policy was trained in (Arena: the Alex
  ``PELVIS_LINK`` frame). The receiver applies them as hand desireds in the streaming
  toolbox's working frame; aligning those frames is a deployment-time calibration concern,
  not handled here.
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

# Default UDP port the IHMC CapturyPoseReceiver binds (matches the Captury->RDX bridge).
DEFAULT_IKSTREAMER_PORT = 2102

# Per-segment payload: pos(3) + quat xyzw(4) + valid(1) = 8 float32.
_FLOATS_PER_SEGMENT = 8
_HEADER_STRUCT = struct.Struct("<q")  # int64 timestamp (microseconds), little-endian
_SEGMENT_STRUCT = struct.Struct("<8f")


class Segment(IntEnum):
    """Pose segments in ``CapturyPoseReceiver.Segment.ORDER`` (wire order)."""

    LEFT_HAND = 0
    RIGHT_HAND = 1
    HEAD = 2
    CHEST = 3


# Number of segments every packet must carry, in wire order.
_SEGMENT_ORDER = (Segment.LEFT_HAND, Segment.RIGHT_HAND, Segment.HEAD, Segment.CHEST)

PACKET_SIZE_BYTES = _HEADER_STRUCT.size + len(_SEGMENT_ORDER) * _SEGMENT_STRUCT.size  # 136

# GR00T EEF action layout: [left_wrist_pose(7) | right_wrist_pose(7) | hand_joints(20)].
LEFT_WRIST_POSE_SLICE = slice(0, 7)
RIGHT_WRIST_POSE_SLICE = slice(7, 14)
HAND_JOINTS_SLICE = slice(14, 34)


@dataclass(frozen=True)
class SegmentPose:
    """One segment's target: position (m), quaternion, and a validity flag."""

    position: np.ndarray  # (3,)
    quaternion: np.ndarray  # (4,) scalar-LAST (x, y, z, w) — already in wire order
    valid: bool = True

    @staticmethod
    def invalid() -> SegmentPose:
        """A zeroed, ignored segment (identity orientation, valid=False)."""
        return SegmentPose(np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0], np.float32), valid=False)


def _to_xyzw(quat: np.ndarray, scalar_first: bool) -> np.ndarray:
    """Return the quaternion as scalar-last (x, y, z, w) for the wire."""
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    if scalar_first:  # (w, x, y, z) -> (x, y, z, w)
        return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)
    return quat


def encode_pose_packet(timestamp_us: int, segments: dict[Segment, SegmentPose]) -> bytes:
    """Encode a 136-byte CapturyPoseReceiver UDP packet.

    Segments absent from ``segments`` are emitted as invalid (zeroed) slots, since the
    receiver always reads all four. Quaternions in ``segments`` must already be scalar-last.
    """
    buf = bytearray(_HEADER_STRUCT.pack(int(timestamp_us)))
    for segment in _SEGMENT_ORDER:
        pose = segments.get(segment) or SegmentPose.invalid()
        pos = np.asarray(pose.position, dtype=np.float32).reshape(3)
        quat = np.asarray(pose.quaternion, dtype=np.float32).reshape(4)
        buf += _SEGMENT_STRUCT.pack(
            pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3], 1.0 if pose.valid else 0.0
        )
    assert len(buf) == PACKET_SIZE_BYTES, f"packet is {len(buf)} bytes, expected {PACKET_SIZE_BYTES}"
    return bytes(buf)


def split_gr00t_action(action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a 34-dim GR00T EEF action into (left_wrist_pose, right_wrist_pose, hand_joints).

    ``action`` may be a single (34,) vector or a batch (B, 34); the leading batch dim is
    preserved on each returned slice.
    """
    action = np.asarray(action, dtype=np.float32)
    assert action.shape[-1] == 34, f"expected a 34-dim EEF action, got shape {action.shape}"
    return action[..., LEFT_WRIST_POSE_SLICE], action[..., RIGHT_WRIST_POSE_SLICE], action[..., HAND_JOINTS_SLICE]


class IKStreamerBridge:
    """UDP sender that streams hand target poses to an IHMC ``CapturyPoseReceiver``."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_IKSTREAMER_PORT, scalar_first_quat: bool = True):
        """Create the bridge.

        Args:
            host: IKStreamer host (the machine running ``RDXCapturyKinematicStreaming``).
            port: UDP port the ``CapturyPoseReceiver`` binds (default ``2102``).
            scalar_first_quat: Interpret input quaternions as scalar-first ``(w, x, y, z)``
                (Isaac Lab / Arena convention) and reorder to the wire's scalar-last layout.
                Set False if callers already pass ``(x, y, z, w)``.
        """
        self._addr = (host, port)
        self._scalar_first = scalar_first_quat
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_hand_poses(
        self,
        left_pose: np.ndarray | None,
        right_pose: np.ndarray | None,
        head_pose: np.ndarray | None = None,
        chest_pose: np.ndarray | None = None,
        timestamp_us: int | None = None,
    ) -> bytes:
        """Pack and send one pose packet. Each ``*_pose`` is a 7-vec ``[pos(3), quat(4)]`` or None.

        Returns the bytes sent (useful for logging/tests).
        """
        if timestamp_us is None:
            timestamp_us = time.time_ns() // 1000
        segments: dict[Segment, SegmentPose] = {}
        for segment, pose in (
            (Segment.LEFT_HAND, left_pose),
            (Segment.RIGHT_HAND, right_pose),
            (Segment.HEAD, head_pose),
            (Segment.CHEST, chest_pose),
        ):
            if pose is None:
                continue
            pose = np.asarray(pose, dtype=np.float32).reshape(7)
            segments[segment] = SegmentPose(
                position=pose[:3],
                quaternion=_to_xyzw(pose[3:7], self._scalar_first),
                valid=True,
            )
        packet = encode_pose_packet(timestamp_us, segments)
        self._sock.sendto(packet, self._addr)
        return packet

    def send_gr00t_action(self, action: np.ndarray, timestamp_us: int | None = None) -> bytes:
        """Extract the wrist poses from a 34-dim GR00T EEF action and stream them.

        The 20-dim hand-joint block is not part of the wrist-pose stream (the Ability Hand
        fingers are driven through their own interface, not the IK streaming toolbox).
        """
        left_pose, right_pose, _hands = split_gr00t_action(np.asarray(action).reshape(-1))
        return self.send_hand_poses(left_pose, right_pose, timestamp_us=timestamp_us)

    def close(self) -> None:
        """Close the UDP socket."""
        self._sock.close()

    def __enter__(self) -> IKStreamerBridge:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
