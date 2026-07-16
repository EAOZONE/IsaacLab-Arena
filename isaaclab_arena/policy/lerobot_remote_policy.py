# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0

"""Schema-checked remote LeRobot policy for the Alex ability-hand embodiment."""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.embodiments.alex.alex import ABILITY_HAND_TELEOP_JOINT_ORDER
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_gr00t.streaming.gr00t_eef_ikstream_bridge import (
    IKStreamerBridge,
    add_ikstreamer_cli_args,
    create_ikstreamer_bridge_from_args,
    stream_env_action_to_ikstreamer,
)

_STATE_DIM = 48
_POLICY_ACTION_DIM = 46
_ARENA_ACTION_DIM = 34
_LEFT_FOREARM_LINK = "LEFT_WRIST_Z_LINK"
_RIGHT_FOREARM_LINK = "RIGHT_WRIST_Z_LINK"
_HEAD_LINK = "HEAD_LINK"
_SPINE_JOINTS = ["SPINE_Z", "SPINE_Y"]
_GROUPED_HAND_NAMES = [
    f"{side}_ability_hand_{finger}_{joint}"
    for side in ("left", "right")
    for finger in ("index", "middle", "ring", "pinky", "thumb")
    for joint in ("q1", "q2")
]
_GROUPED_FROM_PINK = [ABILITY_HAND_TELEOP_JOINT_ORDER.index(name) for name in _GROUPED_HAND_NAMES]
_PINK_FROM_GROUPED = np.argsort(_GROUPED_FROM_PINK).tolist()


@dataclass
class LeRobotRemotePolicyArgs:
    remote_url: str
    rollout_manifest: str
    policy_device: str = "cpu"
    stream_ikstreamer: bool = False
    ikstreamer_host: str = "127.0.0.1"
    ikstreamer_port: int = 2102
    debug_ikstreamer: bool = False
    ikstreamer_yaw_offset: float = 0.0


def _as_torch(value) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else wp.to_torch(value)


def _quat_wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Return Isaac body quats in the H2Ozone/test_obs_new xyzw layout."""
    return quat


def _body_pose(env, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.unwrapped.scene["robot"]
    ids, _ = robot.find_bodies([name])
    idx = int(ids[0])
    pos = _as_torch(robot.data.body_pos_w)[:, idx] - env.unwrapped.scene.env_origins
    quat = _quat_wxyz_to_xyzw(_as_torch(robot.data.body_quat_w)[:, idx])
    return pos, quat


def _alex_state(env) -> np.ndarray:
    robot = env.unwrapped.scene["robot"]
    left_pos, left_quat = _body_pose(env, "LEFT_GRIPPER_Z_LINK")
    right_pos, right_quat = _body_pose(env, "RIGHT_GRIPPER_Z_LINK")
    _, left_forearm = _body_pose(env, _LEFT_FOREARM_LINK)
    _, right_forearm = _body_pose(env, _RIGHT_FOREARM_LINK)
    _, head = _body_pose(env, _HEAD_LINK)
    hand_ids, _ = robot.find_joints(ABILITY_HAND_TELEOP_JOINT_ORDER, preserve_order=True)
    hand_pink = _as_torch(robot.data.joint_pos)[:, hand_ids]
    hand_grouped = hand_pink[:, _GROUPED_FROM_PINK]
    joint_names = list(robot.joint_names)
    joint_pos = _as_torch(robot.data.joint_pos)
    spine = torch.stack(
        [joint_pos[:, joint_names.index(name)] if name in joint_names else torch.zeros_like(joint_pos[:, 0]) for name in _SPINE_JOINTS],
        dim=1,
    )
    state = torch.cat(
        [left_pos, left_quat, right_pos, right_quat, left_forearm, right_forearm, head, hand_grouped, spine],
        dim=1,
    )
    if state.shape[1] != _STATE_DIM:
        raise ValueError(f"Alex state adapter produced {state.shape[1]} values, expected {_STATE_DIM}")
    return state.detach().cpu().numpy()


def _camera_features(observation: dict[str, Any], expected: list[str]) -> dict[str, np.ndarray]:
    cameras = observation.get("camera_obs") or {}
    result = {}
    for feature in expected:
        side = "left" if "left" in feature else "right" if "right" in feature else None
        candidates = [name for name in cameras if side is None or side in name]
        if not candidates:
            raise ValueError(f"No Arena camera matches checkpoint feature {feature!r}")
        value = cameras[candidates[0]]
        result[feature] = _as_torch(value).detach().cpu().numpy()
    return result


def _encode_request(features: dict[str, np.ndarray], task: str) -> bytes:
    output = io.BytesIO()
    mapping = {}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for index, (key, value) in enumerate(features.items()):
            filename = f"feature-{index}.npy"
            data = io.BytesIO()
            np.save(data, value, allow_pickle=False)
            archive.writestr(filename, data.getvalue())
            mapping[key] = filename
        archive.writestr("meta.json", json.dumps({"features": mapping, "task": task, "robot_type": "alex"}))
    return output.getvalue()


class LeRobotRemotePolicy(PolicyBase):
    """Runs any LeRobot policy whose saved schema matches Alex test_obs_new."""

    name = "lerobot_remote"
    config_class = LeRobotRemotePolicyArgs

    def __init__(self, config: LeRobotRemotePolicyArgs):
        super().__init__(config)
        self.remote_url = config.remote_url.rstrip("/")
        self.manifest = json.loads(config.rollout_manifest)
        with urllib.request.urlopen(f"{self.remote_url}/schema", timeout=10) as response:
            self.schema = json.loads(response.read())
        inputs = self.schema.get("input_features") or {}
        outputs = self.schema.get("output_features") or {}
        state_shape = (inputs.get("observation.state") or {}).get("shape")
        action_shape = (outputs.get("action") or {}).get("shape")
        if state_shape != [_STATE_DIM] or action_shape != [_POLICY_ACTION_DIM]:
            raise ValueError(
                "Checkpoint is not compatible with alex_v2_ability_hands/test_obs_new: "
                f"state={state_shape}, action={action_shape}; expected [{_STATE_DIM}] and [{_POLICY_ACTION_DIM}]"
            )
        self.image_features = sorted(name for name in inputs if name.startswith("observation.images."))
        if not self.image_features:
            raise ValueError("Alex visual rollout requires at least one checkpoint image feature")
        self._chunk: torch.Tensor | None = None
        self._chunk_index = 0
        self.task_description = ""

        self._ikstreamer_bridge: IKStreamerBridge | None = create_ikstreamer_bridge_from_args(config)
        self._ikstreamer_dim_mismatch_warned = [False]

    @property
    def is_remote(self) -> bool:
        return True

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("Remote LeRobot policy")
        group.add_argument("--remote_url", required=True)
        group.add_argument("--rollout_manifest", required=True)
        add_ikstreamer_cli_args(parser)
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "LeRobotRemotePolicy":
        return LeRobotRemotePolicy(
            LeRobotRemotePolicyArgs(
                remote_url=args.remote_url,
                rollout_manifest=args.rollout_manifest,
                policy_device=getattr(args, "policy_device", "cpu"),
                stream_ikstreamer=getattr(args, "stream_ikstreamer", False),
                ikstreamer_host=getattr(args, "ikstreamer_host", "127.0.0.1"),
                ikstreamer_port=getattr(args, "ikstreamer_port", 2102),
                debug_ikstreamer=getattr(args, "debug_ikstreamer", False),
                ikstreamer_yaw_offset=getattr(args, "ikstreamer_yaw_offset", 0.0),
            )
        )

    def _request_chunk(self, env, observation) -> torch.Tensor:
        features = {"observation.state": _alex_state(env)}
        features.update(_camera_features(observation, self.image_features))
        request = urllib.request.Request(
            f"{self.remote_url}/predict",
            data=_encode_request(features, self.task_description or ""),
            headers={"Content-Type": "application/zip"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            actions = np.load(io.BytesIO(response.read()), allow_pickle=False)
        if actions.ndim != 3 or actions.shape[2] != _POLICY_ACTION_DIM:
            raise ValueError(f"Policy returned actions with invalid shape {actions.shape}")
        return torch.from_numpy(actions).to(device=env.unwrapped.device, dtype=torch.float32)

    @staticmethod
    def _to_arena_action(action: torch.Tensor) -> torch.Tensor:
        # Policy: wrists(14), forearms(8), neck(4), hands grouped by side/finger(20).
        # Arena Pink IK: wrists(14), hands in ABILITY_HAND_TELEOP_JOINT_ORDER(20).
        wrists = action[:, :14]
        grouped_hands = action[:, 26:46]
        return torch.cat([wrists, grouped_hands[:, _PINK_FROM_GROUPED]], dim=1)

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        if self._chunk is None or self._chunk_index >= self._chunk.shape[1]:
            self._chunk = self._request_chunk(env, observation)
            self._chunk_index = 0
        action = self._to_arena_action(self._chunk[:, self._chunk_index])
        self._chunk_index += 1
        if action.shape[1] != _ARENA_ACTION_DIM:
            raise ValueError(f"Arena action adapter produced {action.shape[1]} values")
        if self._ikstreamer_bridge is not None:
            stream_env_action_to_ikstreamer(
                self._ikstreamer_bridge,
                action,
                env=env,
                env_index=0,
                dim_mismatch_warned=self._ikstreamer_dim_mismatch_warned,
            )
        return action

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._chunk = None
        self._chunk_index = 0
        request = urllib.request.Request(f"{self.remote_url}/reset", data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=10):
            pass

    def close(self) -> None:
        if self._ikstreamer_bridge is not None:
            self._ikstreamer_bridge.close()
            self._ikstreamer_bridge = None

    def shutdown_remote(self, kill_server: bool = False) -> None:
        """LeLab owns the shared remote server/container lifecycle."""
        self.close()
