# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bridge a live Java client (Gr00tClient/Gr00tUpdateThread, in the ihmc repo-group) to GR00T's
own policy server, so real/SCS2-simulated Alex can run closed-loop GR00T inference with no
IsaacSim anywhere in the loop.

This module deliberately has **no isaaclab / isaacsim / isaaclab_arena_g1 imports** - only
``gr00t`` (the Isaac-GR00T model package), ``numpy``, ``msgpack``, and ``websockets``. It can run
on any GPU box that has the ``gr00t`` package and PyTorch installed, with no simulator.

Wire protocol (server side of what ``us.ihmc.openpi.OpenpiClient`` /
``us.ihmc.gr00t.Gr00tClient`` already speak - see those classes for the Java side):

* On connect, send one binary frame first (arbitrary payload; the Java client just needs *a*
  first message to unblock its handshake wait).
* For each binary request frame received: msgpack-decode a map with keys ``cam_zed_left``,
  ``cam_zed_right``, ``state``, ``prompt``. The three array-valued keys are each wrapped as
  ``{"__ndarray__": True, "data": <bytes>, "dtype": <str>, "shape": [...]}``.
* Respond with one binary frame: msgpack map ``{"actions": <ndarray-wrapped float64 array of
  shape [chunk_length, STATE_SIZE]>, "policy_timing": {"infer_ms": float},
  "server_timing": {"infer_ms": float}}``.
* On error, send a **text** frame (the Java handler treats text frames as server-side errors,
  binary frames as normal responses).

State/action layout (36 values; matches ``us.ihmc.alex.gr00t.Gr00tYoRegistry`` on the Java side
and ``embodiments/alex/alex_lever_eef_modality.json`` here)::

    [0:7]   left_wrist_pose  (pos xyz + quat xyzw, scalar-last)
    [7:14]  right_wrist_pose (pos xyz + quat xyzw, scalar-last)
    [14:24] left_hand        (5 fingers x 2 joints, radians)
    [24:34] right_hand       (5 fingers x 2 joints, radians)
    [34:36] neck             (neck_z, neck_y, radians)

GR00T's own wire format, verified against ``submodules/Isaac-GR00T/gr00t/policy/gr00t_policy.py``
(``Gr00tPolicy.check_observation``/``check_action``) and ``gr00t/policy/server_client.py``: this is
**not** the flat ``"video.<key>"``/``"state.<key>"`` convention used by ``Gr00tSimPolicyWrapper``
for sim environments - ``run_gr00t_server.py`` wraps ``Gr00tPolicy`` directly, which expects a
nested dict::

    observation = {
        "video":    {video_key: np.ndarray[uint8,   (B, T, H, W, 3)]},   # one entry per video key
        "state":    {state_key: np.ndarray[float32, (B, T, D)]},        # one entry per state key
        "language": {language_key: [["task text"]] * B},                # list[B] of list[T] of str
    }

and returns a **flat** action dict ``{action_key: np.ndarray[float32, (B, T, D)]}`` (no "action."
prefix). The exact key names (video/state/action/language) and temporal horizons (``T``, via
``delta_indices``) are queried live from the server's ``get_modality_config`` endpoint at startup
rather than hardcoded, since they're checkpoint-specific; only the per-key *dimension ranges*
below (which match the Java-side schema) are hardcoded, keyed by name so key order doesn't matter.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass

import msgpack
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

# --------------------------------------------------------------------------- #
# GR00T Alex EEF embodiment layout (matches alex_lever_eef_modality.json).
# --------------------------------------------------------------------------- #

STATE_SIZE = 36
ACTION_SIZE = 36
ACTION_CHUNK_LENGTH = 50  # Matches OpenpiClient's fixed action-chunk buffer; see Open Item #4.

# name -> (start, end) in the flat 36-value Java-side buffer. Keyed by name (not position), since
# the live server's modality_config key *order* is not guaranteed to match this order.
STATE_RANGES = {
    "left_wrist_pose": (0, 7),
    "right_wrist_pose": (7, 14),
    "left_hand": (14, 24),
    "right_hand": (24, 34),
    "neck": (34, 36),
}
ACTION_RANGES = STATE_RANGES  # Same field layout for state and action.

# Names Gr00tClient sends the two ZED images under (see Gr00tClient.request()).
WIRE_VIDEO_KEYS = ("cam_zed_left", "cam_zed_right")


def ndarray_to_wire(array: np.ndarray) -> dict:
    """Build the {__ndarray__, data, dtype, shape} wire dict OpenpiClient/Gr00tClient expect.
    Dict key order matters here: OpenpiClient's unpacker reads fields positionally (by order, not
    by name), so this must always insert keys in exactly this order."""
    return {
        "__ndarray__": True,
        "data": array.tobytes(),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def unpack_ndarray(d: dict) -> np.ndarray:
    return np.frombuffer(d["data"], dtype=np.dtype(d["dtype"])).reshape(d["shape"])


@dataclass
class Gr00tBridgeConfig:
    listen_host: str
    listen_port: int
    gr00t_host: str
    gr00t_port: int
    task_description: str


class Gr00tOpenpiBridgeServer:
    """Owns one connection to GR00T's PolicyClient and serves the openpi wire protocol.

    Key names and temporal horizons (video/state/action/language) are queried live from the
    server via ``get_modality_config`` at startup rather than hardcoded, since they depend on the
    specific checkpoint/embodiment config running behind ``run_gr00t_server.py``.
    """

    def __init__(self, config: Gr00tBridgeConfig):
        self.config = config
        # Imported lazily so this module can be imported (e.g. for unit-testing the slicing
        # helpers above) without the gr00t package installed.
        from gr00t.policy.server_client import PolicyClient as Gr00tPolicyClient

        self._client = Gr00tPolicyClient(host=config.gr00t_host, port=config.gr00t_port, strict=False)
        if not self._client.ping():
            raise ConnectionError(f"Cannot reach GR00T policy server at {config.gr00t_host}:{config.gr00t_port}")

        modality_configs = self._client.get_modality_config()
        self.video_keys: list[str] = list(modality_configs["video"].modality_keys)
        self.state_keys: list[str] = list(modality_configs["state"].modality_keys)
        self.action_keys: list[str] = list(modality_configs["action"].modality_keys)
        language_keys = list(modality_configs["language"].modality_keys)
        assert len(language_keys) == 1, f"Only one language key is supported, got {language_keys}"
        self.language_key = language_keys[0]

        self.video_horizon = len(modality_configs["video"].delta_indices)
        self.state_horizon = len(modality_configs["state"].delta_indices)
        if self.video_horizon != 1 or self.state_horizon != 1:
            print(f"WARNING: server expects video_horizon={self.video_horizon}, state_horizon={self.state_horizon} "
                  "(temporal history); this bridge only has the latest frame and will naively repeat it, which is "
                  "likely wrong for checkpoints that actually use history.")

        unknown_state_keys = set(self.state_keys) - set(STATE_RANGES)
        unknown_action_keys = set(self.action_keys) - set(ACTION_RANGES)
        assert not unknown_state_keys, f"Server state keys not in the Java-side 36-value layout: {unknown_state_keys}"
        assert not unknown_action_keys, (
            f"Server action keys not in the Java-side 36-value layout: {unknown_action_keys}"
        )

        print(f"GR00T modality config: video={self.video_keys}, state={self.state_keys}, "
              f"action={self.action_keys}, language_key={self.language_key!r}")

    def infer(self, request: dict) -> dict:
        server_start = time.perf_counter()

        state = unpack_ndarray(request["state"])  # flat float32[36]

        observation: dict[str, dict] = {"video": {}, "state": {}, "language": {}}
        for video_key in self.video_keys:
            assert video_key in request, (
                f"Server expects video key {video_key!r}, but Gr00tClient only sends {WIRE_VIDEO_KEYS}"
            )
            image = unpack_ndarray(request[video_key])  # uint8, shape (3, H, W), CHW
            hwc = np.transpose(image, (1, 2, 0))  # -> (H, W, 3)
            observation["video"][video_key] = np.tile(hwc[np.newaxis, np.newaxis], (1, self.video_horizon, 1, 1, 1))
        for state_key in self.state_keys:
            start, end = STATE_RANGES[state_key]
            value = state[start:end].astype(np.float32)
            observation["state"][state_key] = np.tile(value[np.newaxis, np.newaxis], (1, self.state_horizon, 1))
        task_description = request.get("prompt") or self.config.task_description
        observation["language"][self.language_key] = [[task_description]]

        policy_start = time.perf_counter()
        action_dict, _ = self._client.get_action(observation)
        policy_ms = (time.perf_counter() - policy_start) * 1000.0

        any_action = next(iter(action_dict.values()))
        horizon = np.asarray(any_action).shape[1]  # (B, T, D) - T is the action horizon.
        actions = np.zeros((ACTION_CHUNK_LENGTH, ACTION_SIZE), dtype=np.float64)
        for step in range(ACTION_CHUNK_LENGTH):
            # Clamp (hold the last predicted step) rather than wrap back to t=0: GR00T's horizon is
            # a single forward-in-time trajectory, not a loop, so wrapping caused the robot to snap
            # backwards through the whole trajectory every `horizon` steps once past the real data.
            t = min(step, horizon - 1)
            for action_key in self.action_keys:
                start, end = ACTION_RANGES[action_key]
                actions[step, start:end] = np.asarray(action_dict[action_key])[0, t, :]

        server_ms = (time.perf_counter() - server_start) * 1000.0
        return {"actions": actions, "policy_timing_ms": policy_ms, "server_timing_ms": server_ms}

    async def handle_connection(self, websocket: WebSocketServerProtocol) -> None:
        await websocket.send(msgpack.packb({"status": "ready"}))

        async for message in websocket:
            if not isinstance(message, (bytes, bytearray)):
                continue  # Ignore stray text frames from the client.

            try:
                request = msgpack.unpackb(message, raw=False)
                result = self.infer(request)

                # Plain dict, keys in insertion order: OpenpiClient's unpacker on the Java side
                # reads fields positionally (by order, not by name), so this order must match
                # exactly what us.ihmc.gr00t.Gr00tClient.unpack() expects.
                response = {
                    "actions": ndarray_to_wire(result["actions"]),
                    "policy_timing": {"infer_ms": float(result["policy_timing_ms"])},
                    "server_timing": {"infer_ms": float(result["server_timing_ms"])},
                }
                await websocket.send(msgpack.packb(response, use_bin_type=True))
            except Exception as exception:  # noqa: BLE001 - report all failures to the client
                await websocket.send(f"{type(exception).__name__}: {exception}")


async def main_async(config: Gr00tBridgeConfig) -> None:
    server = Gr00tOpenpiBridgeServer(config)
    async with websockets.serve(server.handle_connection, config.listen_host, config.listen_port):
        print(f"GR00T openpi-bridge listening on ws://{config.listen_host}:{config.listen_port}, "
              f"forwarding to GR00T server at {config.gr00t_host}:{config.gr00t_port}")
        await asyncio.Future()  # run forever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8000)  # Matches OpenpiClient's fixed port.
    parser.add_argument("--gr00t-host", default="localhost")
    parser.add_argument("--gr00t-port", type=int, default=5555)  # Matches run_gr00t_server.py's default.
    parser.add_argument("--task-description", default="")
    args = parser.parse_args()

    config = Gr00tBridgeConfig(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        gr00t_host=args.gr00t_host,
        gr00t_port=args.gr00t_port,
        task_description=args.task_description,
    )
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
