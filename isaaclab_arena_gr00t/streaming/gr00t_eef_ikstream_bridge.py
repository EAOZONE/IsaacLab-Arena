# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stream Arena end-effector targets into the IHMC RDX whole-body IK (Arena -> RDX).

A GR00T policy (or a replayed dataset) on the Alex ability-hands EEF action space produces, per
step, a 34-dim action ``[ left_wrist_pose(7) | right_wrist_pose(7) | hand_joints(20) ]`` where
each wrist pose is ``pos(3) + quat(4)``. The RDX side (``RDXAlexArenaIKStreamingPanel`` +
``ArenaIKStreamReceiver`` in the ihmc-alex repo) listens on a UDP port, decodes these hand
targets, and feeds them to the ``KinematicsStreamingToolboxInputMessage`` (whole-body IK), which
resolves the arm/torso joints on the robot. This module is the sender: the policy commands
*where the hands go* and the robot's IK solver reproduces the rest.

Wire format (canonical Arena IK stream, little-endian)::

    int64   timestamp_microseconds
    per segment in ORDER = (LEFT_HAND, RIGHT_HAND, HEAD, CHEST):
        float32 pos_x, pos_y, pos_z
        float32 quat_x, quat_y, quat_z, quat_w   # scalar-LAST (euclid Quaternion(x,y,z,w))
        float32 valid                            # > 0 -> pose is used, else ignored

Total = 8 + 4 * (8 * 4) = 136 bytes. ``ArenaIKStreamReceiver`` always reads all four segments,
so a packet must include head/chest slots even when only the hands are driven (send valid=0 for
unused segments).

Note on conventions/frames:
* The wire is scalar-LAST ``(x, y, z, w)``. The Alex EEF action blocks streamed here are
  *also* scalar-last (the Se3AbsRetargeter emits xyzw and the Pink IK action term consumes
  xyzw), so :func:`create_ikstreamer_bridge_from_args` builds the bridge with
  ``scalar_first_quat=False`` and passes the wrist quats through unchanged. ``send_hand_poses``
  still accepts scalar-first ``(w, x, y, z)`` quats (the generic Isaac Lab convention) when a
  caller constructs the bridge with the default ``scalar_first_quat=True``.
* The wrist poses are in the frame the policy was trained in (Arena: the Alex ``PELVIS_LINK``
  frame). ``RDXAlexArenaIKStreamingPanel`` reinterprets each pose in the synced robot's pelvis
  frame and converts it to world before handing it to the IK solver, so the RDX robot moves the
  same way Arena does regardless of where it stands in the world.
"""

from __future__ import annotations

import argparse
import socket
import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

# Default UDP port the RDX-side ArenaIKStreamReceiver binds.
DEFAULT_IKSTREAMER_PORT = 2102

# Per-segment payload: pos(3) + quat xyzw(4) + valid(1) = 8 float32.
_FLOATS_PER_SEGMENT = 8
_HEADER_STRUCT = struct.Struct("<q")  # int64 timestamp (microseconds), little-endian
_SEGMENT_STRUCT = struct.Struct("<8f")


class Segment(IntEnum):
    """Pose segments in the Arena IK stream wire order."""

    LEFT_HAND = 0
    RIGHT_HAND = 1
    HEAD = 2
    CHEST = 3


# Number of segments every packet must carry, in wire order.
_SEGMENT_ORDER = (Segment.LEFT_HAND, Segment.RIGHT_HAND, Segment.HEAD, Segment.CHEST)

PACKET_SIZE_BYTES = _HEADER_STRUCT.size + len(_SEGMENT_ORDER) * _SEGMENT_STRUCT.size  # 136

# GR00T EEF action layout: [left_wrist_pose(7) | right_wrist_pose(7) | hand_joints(20)].
EEF_ACTION_DIM = 34
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


def _quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse, for unit quats) of a scalar-last quaternion."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` of two scalar-last (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector ``v`` by scalar-last quaternion ``q``."""
    qv = np.asarray(q[:3], dtype=np.float64)
    qw = float(q[3])
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def world_wrist_pose_to_base_frame(
    pose_xyzw: np.ndarray, base_pos: np.ndarray, base_quat_xyzw: np.ndarray
) -> np.ndarray:
    """Express a wrist pose given in the env/world frame relative to a base (pelvis) frame.

    The Arena Pink IK action term consumes wrist poses in the env-origin (world) frame and
    converts them to the ``PELVIS_LINK`` frame internally before solving (see
    ``pink_task_space_actions.py::_transform_poses_to_base_link_frame``). The RDX
    ``ArenaIKStreamReceiver`` instead expects targets already in the pelvis frame (it reinterprets
    them in the synced robot's pelvis and converts to world). So we must apply the same world->pelvis
    transform here before streaming.

    Args:
        pose_xyzw: 7-vec ``[pos(3), quat xyzw(4)]`` in the env/world frame.
        base_pos: Pelvis position (3,) in the same env/world frame as ``pose_xyzw``.
        base_quat_xyzw: Pelvis orientation as scalar-last ``(x, y, z, w)``.

    Returns:
        7-vec ``[pos(3), quat xyzw(4)]`` expressed in the pelvis frame.
    """
    p = np.asarray(pose_xyzw[:3], dtype=np.float64)
    q = np.asarray(pose_xyzw[3:7], dtype=np.float64)
    base_p = np.asarray(base_pos, dtype=np.float64)
    base_q = np.asarray(base_quat_xyzw, dtype=np.float64)
    base_q_inv = _quat_conjugate_xyzw(base_q)
    pos_rel = _quat_rotate_xyzw(base_q_inv, p - base_p)
    quat_rel = _quat_mul_xyzw(base_q_inv, q)
    return np.concatenate([pos_rel, quat_rel]).astype(np.float32)


def encode_pose_packet(timestamp_us: int, segments: dict[Segment, SegmentPose]) -> bytes:
    """Encode a 136-byte Arena IK stream UDP packet.

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
    assert (
        action.shape[-1] == EEF_ACTION_DIM
    ), f"expected a {EEF_ACTION_DIM}-dim EEF action, got shape {action.shape}"
    return action[..., LEFT_WRIST_POSE_SLICE], action[..., RIGHT_WRIST_POSE_SLICE], action[..., HAND_JOINTS_SLICE]


class IKStreamerBridge:
    """UDP sender that streams hand target poses to the RDX ``ArenaIKStreamReceiver``."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_IKSTREAMER_PORT,
        scalar_first_quat: bool = True,
        debug: bool = False,
    ):
        """Create the bridge.

        Args:
            host: RDX host (the machine running ``AlexRDXSimulationUI`` with Arena IK streaming).
            port: UDP port the ``ArenaIKStreamReceiver`` binds (default ``2102``).
            scalar_first_quat: Interpret input quaternions as scalar-first ``(w, x, y, z)``
                (Isaac Lab / Arena convention) and reorder to the wire's scalar-last layout.
                Set False if callers already pass ``(x, y, z, w)``.
            debug: Print each sent packet to the console.
        """
        self._addr = (host, port)
        self._scalar_first = scalar_first_quat
        self._debug = debug
        self._pelvis_offset_yaw = 0.0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def set_pelvis_offset_yaw(self, yaw_deg: float):
        """Set a yaw offset (in degrees) to apply to all poses before streaming.

        Use this if the RDX robot's pelvis frame is rotated relative to the Arena pelvis frame.
        """
        self._pelvis_offset_yaw = np.radians(yaw_deg)

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

            p = pose[:3]
            q = _to_xyzw(pose[3:7], self._scalar_first)

            # Apply pelvis offset if set
            if self._pelvis_offset_yaw != 0.0:
                # Basic yaw rotation without scipy
                cy = np.cos(self._pelvis_offset_yaw)
                sy = np.sin(self._pelvis_offset_yaw)
                x, y, z = p
                p = np.array([x * cy - y * sy, x * sy + y * cy, z], dtype=np.float32)
                
                # Quaternion rotation (yaw only)
                # q1 = [0, 0, sin(yaw/2), cos(yaw/2)] in xyzw
                z1 = np.sin(self._pelvis_offset_yaw / 2.0)
                w1 = np.cos(self._pelvis_offset_yaw / 2.0)
                
                # Multiplay q_new = q1 * q2
                x2, y2, z2, w2 = q
                q = np.array([
                    w1 * x2 - z1 * y2,
                    w1 * y2 + z1 * x2,
                    w1 * z2 + z1 * w2,
                    w1 * w2 - z1 * z2
                ], dtype=np.float32)

            segments[segment] = SegmentPose(
                position=p,
                quaternion=q,
                valid=True,
            )
        packet = encode_pose_packet(timestamp_us, segments)
        self._sock.sendto(packet, self._addr)

        if self._debug:
            print(f"--- IK Stream Packet (ts: {timestamp_us}) ---")
            for seg in (Segment.LEFT_HAND, Segment.RIGHT_HAND):
                pose = segments.get(seg)
                if pose and pose.valid:
                    p = pose.position
                    q = pose.quaternion  # xyzw
                    print(f"  {seg.name}: pos=[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}] quat=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}]")

        return packet

    def send_gr00t_action(self, action: np.ndarray, timestamp_us: int | None = None) -> bytes:
        """Extract the wrist poses from a 34-dim GR00T EEF action and stream them.

        The 20-dim hand-joint block is not part of the wrist-pose stream (the Ability Hand
        fingers are driven through their own interface, not the IK streaming toolbox).

        Left/right are passed through in Arena order: slice 0:7 (left wrist, targeting
        ``LEFT_GRIPPER_Z_LINK`` in Arena's Pink IK) maps to the wire's ``LEFT_HAND`` segment,
        which ``ArenaIKStreamReceiver`` feeds to ``getHand(RobotSide.LEFT)``. Do NOT swap them —
        both ends use the same pelvis-frame, scalar-last convention, so the RDX robot reproduces
        the same hand motion Arena commands.
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


def add_ikstreamer_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for mirroring Arena EEF actions into the RDX IK streamer."""
    existing_options = {opt for action in parser._actions for opt in action.option_strings}
    if "--stream_ikstreamer" in existing_options:
        return

    parser.add_argument(
        "--stream_ikstreamer",
        action="store_true",
        default=False,
        help=(
            "Mirror each env-0 EEF action step to the RDX whole-body IK over UDP (Arena -> RDX)."
            " Start AlexRDXSimulationUI and enable its 'Arena IK Streaming' panel before running."
        ),
    )
    parser.add_argument(
        "--ikstreamer_host",
        type=str,
        default="127.0.0.1",
        help="Host the RDX ArenaIKStreamReceiver binds (used with --stream_ikstreamer).",
    )
    parser.add_argument(
        "--ikstreamer_port",
        type=int,
        default=DEFAULT_IKSTREAMER_PORT,
        help="UDP port the RDX ArenaIKStreamReceiver binds (used with --stream_ikstreamer).",
    )
    parser.add_argument(
        "--debug_ikstreamer",
        action="store_true",
        default=False,
        help="Print each streamed EEF pose to the console for debugging.",
    )
    parser.add_argument(
        "--ikstreamer_yaw_offset",
        type=float,
        default=0.0,
        help="Yaw offset (degrees) for the streamed poses relative to the pelvis.",
    )


def create_ikstreamer_bridge_from_args(args: argparse.Namespace) -> IKStreamerBridge | None:
    """Create an :class:`IKStreamerBridge` when ``--stream_ikstreamer`` is set."""
    if not getattr(args, "stream_ikstreamer", False):
        return None
    # The Alex EEF action blocks (and the dataset they were trained from) carry wrist
    # quaternions scalar-LAST (x, y, z, w): the Se3AbsRetargeter emits xyzw and the Pink IK
    # action term consumes xyzw (see ALEX_ABILITY_HAND_*_EE_ACTION_KEYS in embodiments/alex/alex.py).
    # The wire is also scalar-last, so pass them through unchanged — do NOT reorder.
    bridge = IKStreamerBridge(
        host=args.ikstreamer_host,
        port=args.ikstreamer_port,
        scalar_first_quat=False,
        debug=getattr(args, "debug_ikstreamer", False),
    )
    if yaw_offset := getattr(args, "ikstreamer_yaw_offset", 0.0):
        bridge.set_pelvis_offset_yaw(yaw_offset)

    print(
        "Streaming env-0 EEF actions to RDX IK streamer at"
        f" {args.ikstreamer_host}:{args.ikstreamer_port}."
    )
    return bridge


# Cache of resolved base-link body indices, keyed by id(robot) -> (index, base_link_name).
_BASE_LINK_IDX_CACHE: dict[int, tuple[int, str]] = {}

# Arena Alex base/pelvis link (the frame the Pink IK action term resolves wrist targets in).
_BASE_LINK_NAME = "PELVIS_LINK"


def _base_link_pose_in_env(env, env_index: int, base_link_name: str = _BASE_LINK_NAME):
    """Return ``(pos, quat_xyzw)`` of the robot base/pelvis link in the env-origin frame.

    Matches the frame the Pink IK action term uses: the body's world pose minus the env origin.
    ``None`` if the base link cannot be resolved (e.g. a non-Alex robot), so the caller can fall back.
    """
    unwrapped = getattr(env, "unwrapped", env)
    try:
        robot = unwrapped.scene["robot"]
    except Exception:
        return None
    cache_key = id(robot)
    cached = _BASE_LINK_IDX_CACHE.get(cache_key)
    if cached is None or cached[1] != base_link_name:
        indices, _names = robot.find_bodies([base_link_name])
        if not indices:
            return None
        cached = (int(indices[0]), base_link_name)
        _BASE_LINK_IDX_CACHE[cache_key] = cached
    base_idx = cached[0]
    body_states = robot.data.body_link_state_w
    if not hasattr(body_states, "detach"):  # Warp array -> torch view (as the Pink IK action term does)
        import warp as wp

        body_states = wp.to_torch(body_states)
    state = body_states[env_index, base_idx, :7]
    pos = state[:3].detach().cpu().numpy().astype(np.float64)
    quat_xyzw = state[3:7].detach().cpu().numpy().astype(np.float64)
    env_origins = unwrapped.scene.env_origins
    if not hasattr(env_origins, "detach"):
        import warp as wp

        env_origins = wp.to_torch(env_origins)
    env_origin = env_origins[env_index].detach().cpu().numpy().astype(np.float64)
    return pos - env_origin, quat_xyzw


def stream_env_action_to_ikstreamer(
    bridge: IKStreamerBridge,
    actions: np.ndarray | torch.Tensor,
    *,
    env=None,
    env_index: int = 0,
    dim_mismatch_warned: list[bool] | None = None,
) -> None:
    """Stream one env row from a batched action tensor to the IK streamer.

    The 34-dim EEF action carries each wrist pose in the env/world frame (that is what the Pink IK
    action term consumes — it converts to the pelvis frame internally). The RDX receiver expects
    pelvis-frame targets, so when ``env`` is given we transform the wrist poses world->pelvis using
    the live ``PELVIS_LINK`` pose before streaming. Without ``env`` we fall back to streaming the raw
    (world-frame) poses and warn once, since RDX would otherwise misplace the hands.

    ``dim_mismatch_warned`` is an optional single-element list used to emit the dimension
    warning at most once (pass ``[False]`` from the rollout loop).
    """
    action_value = actions[env_index]
    if hasattr(action_value, "detach"):
        action_row = action_value.detach().cpu().numpy()
    else:
        action_row = np.asarray(action_value, dtype=np.float32)
    if action_row.shape[-1] != EEF_ACTION_DIM:
        if dim_mismatch_warned is not None and dim_mismatch_warned[0]:
            return
        print(
            f"Warning: --stream_ikstreamer expects a {EEF_ACTION_DIM}-dim EEF action but got"
            f" {action_row.shape[-1]}; skipping IK streaming."
        )
        if dim_mismatch_warned is not None:
            dim_mismatch_warned[0] = True
        return

    base_pose = _base_link_pose_in_env(env, env_index) if env is not None else None
    if base_pose is None:
        if dim_mismatch_warned is not None and not dim_mismatch_warned[0]:
            print(
                "Warning: --stream_ikstreamer could not resolve the robot pelvis pose; streaming raw"
                " world-frame wrist poses (RDX expects pelvis-frame, so the hands may be misplaced)."
            )
            dim_mismatch_warned[0] = True
        bridge.send_gr00t_action(action_row)
        return

    base_pos, base_quat_xyzw = base_pose
    left_world, right_world, _hands = split_gr00t_action(action_row)
    left_pelvis = world_wrist_pose_to_base_frame(left_world, base_pos, base_quat_xyzw)
    right_pelvis = world_wrist_pose_to_base_frame(right_world, base_pos, base_quat_xyzw)
    bridge.send_hand_poses(left_pelvis, right_pelvis)
