# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GR00T->IKStreamer EEF UDP bridge wire format.

The decode helper here mirrors the Java ``ArenaIKStreamReceiver.decode`` byte layout, so a
passing test means the packet Arena emits is parseable by the RDX receiver unchanged.
"""

import argparse
import numpy as np
import socket
import struct

from isaaclab_arena_gr00t.streaming.gr00t_eef_ikstream_bridge import (
    PACKET_SIZE_BYTES,
    Segment,
    SegmentPose,
    IKStreamerBridge,
    create_ikstreamer_bridge_from_args,
    encode_pose_packet,
    split_gr00t_action,
)

_SEGMENT_ORDER = (Segment.LEFT_HAND, Segment.RIGHT_HAND, Segment.HEAD, Segment.CHEST)


def _decode_packet(data: bytes) -> tuple[int, dict[Segment, dict]]:
    """Decode a packet exactly like the Java ArenaIKStreamReceiver (LE int64 + 4*8 floats)."""
    assert len(data) == PACKET_SIZE_BYTES
    (timestamp_us,) = struct.unpack_from("<q", data, 0)
    out: dict[Segment, dict] = {}
    offset = 8
    for segment in _SEGMENT_ORDER:
        px, py, pz, qx, qy, qz, qw, valid = struct.unpack_from("<8f", data, offset)
        offset += 32
        out[segment] = {
            "pos": np.array([px, py, pz], dtype=np.float32),
            "quat_xyzw": np.array([qx, qy, qz, qw], dtype=np.float32),
            "valid": valid > 0.0,
        }
    return timestamp_us, out


def test_packet_size():
    packet = encode_pose_packet(0, {})
    assert len(packet) == PACKET_SIZE_BYTES == 136


def test_encode_roundtrip_per_segment():
    left = SegmentPose(np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.0, 0.0, 1.0]), valid=True)
    right = SegmentPose(np.array([-0.4, 0.5, 0.6]), np.array([0.7071, 0.0, 0.0, 0.7071]), valid=True)
    packet = encode_pose_packet(123456, {Segment.LEFT_HAND: left, Segment.RIGHT_HAND: right})
    ts, decoded = _decode_packet(packet)

    assert ts == 123456
    np.testing.assert_allclose(decoded[Segment.LEFT_HAND]["pos"], left.position, atol=1e-6)
    np.testing.assert_allclose(decoded[Segment.LEFT_HAND]["quat_xyzw"], left.quaternion, atol=1e-6)
    assert decoded[Segment.LEFT_HAND]["valid"] is True
    np.testing.assert_allclose(decoded[Segment.RIGHT_HAND]["pos"], right.position, atol=1e-6)
    # Head/chest were omitted -> emitted as invalid slots.
    assert decoded[Segment.HEAD]["valid"] is False
    assert decoded[Segment.CHEST]["valid"] is False


def test_send_hand_poses_reorders_scalar_first_quat():
    # Input quat is scalar-first (w, x, y, z); the wire must be scalar-last (x, y, z, w).
    wxyz = np.array([0.7071, 0.7071, 0.0, 0.0], dtype=np.float32)  # w, x, y, z
    left_pose = np.concatenate([[1.0, 2.0, 3.0], wxyz])

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2.0)
    port = receiver.getsockname()[1]
    with IKStreamerBridge(host="127.0.0.1", port=port, scalar_first_quat=True) as bridge:
        bridge.send_hand_poses(left_pose=left_pose, right_pose=None, timestamp_us=0)
    data, _ = receiver.recvfrom(4096)
    receiver.close()

    _, decoded = _decode_packet(data)
    # (w,x,y,z)=(0.7071,0.7071,0,0) -> wire (x,y,z,w)=(0.7071,0,0,0.7071)
    np.testing.assert_allclose(decoded[Segment.LEFT_HAND]["quat_xyzw"], [0.7071, 0.0, 0.0, 0.7071], atol=1e-6)
    assert decoded[Segment.LEFT_HAND]["valid"] and not decoded[Segment.RIGHT_HAND]["valid"]


def test_factory_bridge_passes_scalar_last_quat_through():
    # The Alex EEF action blocks are scalar-LAST (x, y, z, w): the retargeter emits xyzw and
    # the Pink IK action term consumes xyzw. The bridge built from CLI args must therefore NOT
    # reorder the quat, otherwise the wrist orientation reaching the RDX IK streamer is scrambled.
    quat_xyzw = np.array([0.7071, 0.0, 0.0, 0.7071], dtype=np.float32)  # x, y, z, w
    action = np.zeros(34, dtype=np.float32)
    action[0:3] = [1.0, 2.0, 3.0]
    action[3:7] = quat_xyzw

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2.0)
    port = receiver.getsockname()[1]

    args = argparse.Namespace(stream_ikstreamer=True, ikstreamer_host="127.0.0.1", ikstreamer_port=port)
    bridge = create_ikstreamer_bridge_from_args(args)
    assert bridge is not None
    with bridge:
        bridge.send_gr00t_action(action, timestamp_us=0)
    data, _ = receiver.recvfrom(4096)
    receiver.close()

    _, decoded = _decode_packet(data)
    # Quat must arrive on the wire exactly as authored (no scalar-first reordering).
    np.testing.assert_allclose(decoded[Segment.LEFT_HAND]["quat_xyzw"], quat_xyzw, atol=1e-6)


def test_factory_returns_none_without_flag():
    args = argparse.Namespace(stream_ikstreamer=False, ikstreamer_host="127.0.0.1", ikstreamer_port=2102)
    assert create_ikstreamer_bridge_from_args(args) is None


def test_split_gr00t_action():
    action = np.arange(34, dtype=np.float32)
    left, right, hands = split_gr00t_action(action)
    np.testing.assert_array_equal(left, np.arange(0, 7))
    np.testing.assert_array_equal(right, np.arange(7, 14))
    np.testing.assert_array_equal(hands, np.arange(14, 34))
    assert hands.shape == (20,)


def test_bridge_sends_over_udp_loopback():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))  # ephemeral port
    receiver.settimeout(2.0)
    port = receiver.getsockname()[1]

    action = np.zeros(34, dtype=np.float32)
    action[0:3] = [0.3, -0.1, 1.2]  # left pos
    action[3:7] = [1.0, 0.0, 0.0, 0.0]  # left quat (w,x,y,z) identity
    action[7:10] = [0.35, 0.1, 1.2]  # right pos
    action[10:14] = [1.0, 0.0, 0.0, 0.0]  # right quat

    with IKStreamerBridge(host="127.0.0.1", port=port) as bridge:
        sent = bridge.send_gr00t_action(action, timestamp_us=999)

    data, _ = receiver.recvfrom(4096)
    receiver.close()
    assert data == sent
    ts, decoded = _decode_packet(data)
    assert ts == 999
    np.testing.assert_allclose(decoded[Segment.LEFT_HAND]["pos"], [0.3, -0.1, 1.2], atol=1e-6)
    np.testing.assert_allclose(decoded[Segment.RIGHT_HAND]["pos"], [0.35, 0.1, 1.2], atol=1e-6)
    # identity (w,x,y,z)=(1,0,0,0) -> wire (x,y,z,w)=(0,0,0,1)
    np.testing.assert_allclose(decoded[Segment.LEFT_HAND]["quat_xyzw"], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
    assert decoded[Segment.LEFT_HAND]["valid"] and decoded[Segment.RIGHT_HAND]["valid"]
    assert not decoded[Segment.HEAD]["valid"]
