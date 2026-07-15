# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CCIL behavioral-cloning policy for closed-loop evaluation in Isaac Lab Arena.

The model is trained offline with CCIL (custom d3rlpy fork, Python 3.8) and exported to
a dependency-free artifact. This policy loads that artifact and runs pure PyTorch at
inference — no d3rlpy, no remote server.

Two artifact formats are supported (resolved automatically):

1. **TorchScript** (preferred): a module saved by d3rlpy ``save_policy`` mapping a raw
   observation ``(B, obs_dim)`` to a raw action ``(B, act_dim)`` with the standard /
   min-max scalers baked in. Loaded via ``torch.jit.load``.
2. **state_dict + meta.json** (fallback for TorchScript cross-version issues): a plain MLP
   ``state_dict`` plus ``meta.json`` describing ``hidden_units``/``activation`` and the
   observation (standard) and action (min-max) normalization, reconstructed here.

Observation is ``robot_joint_pos`` (49-dof). The action is the 34-dim raw Pink IK action —
left/right EE target poses (pos 3 + quat 4 each = 14) followed by 20 ability-hand finger
joints — so it must be applied through an IK-in-the-loop embodiment (``alex_ability_hands``)
that resolves the EE targets to a whole-body solution, not the direct joint-position
embodiment. (Train on ``processed_actions`` instead for the legacy direct-joint path.)
"""

from __future__ import annotations

import argparse
import gymnasium as gym
import json
import logging
import os
import torch
import torch.nn.functional as F
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_ccil.policy.config.ccil_bc_policy_config import CCILBCPolicyConfig

logger = logging.getLogger(__name__)

_ACTIVATIONS: dict[str, type[torch.nn.Module]] = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "gelu": torch.nn.GELU,
    "elu": torch.nn.ELU,
}

# test_obs_new stores each hand as index/middle/ring/pinky/thumb q1/q2
# pairs. Arena's Pink IK action uses ABILITY_HAND_TELEOP_JOINT_ORDER.
_TEST_OBS_NEW_HAND_TO_TELEOP = (
    0,
    2,
    4,
    6,
    10,
    12,
    14,
    16,
    8,
    18,
    1,
    3,
    5,
    7,
    11,
    13,
    15,
    17,
    9,
    19,
)


def _build_mlp(input_dim: int, output_dim: int, hidden_units: list[int], activation: str) -> torch.nn.Sequential:
    """Build a plain MLP matching d3rlpy's default vector encoder + action head."""
    act_cls = _ACTIVATIONS[activation.lower()]
    layers: list[torch.nn.Module] = []
    prev = input_dim
    for h in hidden_units:
        layers.append(torch.nn.Linear(prev, h))
        layers.append(act_cls())
        prev = h
    layers.append(torch.nn.Linear(prev, output_dim))
    return torch.nn.Sequential(*layers)


class CCILBCPolicy(PolicyBase):
    """CCIL behavioral-cloning policy (state-based MLP) running in-process in Arena."""

    name = "ccil_bc"
    config_class = CCILBCPolicyConfig

    def __init__(self, config: CCILBCPolicyConfig):
        super().__init__(config)
        self.device = config.policy_device
        self.state_key = config.state_key
        self.state_adapter = config.state_adapter
        self.num_envs = config.num_envs
        self.use_images = config.use_images
        self.camera_group_key = config.camera_group_key
        self.image_keys = tuple(config.image_keys)
        self.image_size = tuple(config.image_size)
        self.replay_video_keys = tuple(config.replay_video_keys)
        self._video_source = None
        if config.replay_video_dataset_path is not None:
            from isaaclab_arena_gr00t.policy.dataset_video_override import LerobotDatasetVideoSource

            self._video_source = LerobotDatasetVideoSource(
                dataset_root=config.replay_video_dataset_path,
                episode_index=config.replay_video_episode,
                stride=config.replay_video_stride,
                loop=config.replay_video_loop,
                show_ui=config.show_replay_video,
            )
            logger.info(
                "CCILBCPolicy: replay video override enabled: %s episode %s stride %s loop=%s show_ui=%s",
                config.replay_video_dataset_path,
                config.replay_video_episode,
                config.replay_video_stride,
                config.replay_video_loop,
                config.show_replay_video,
            )

        assert os.path.exists(config.model_path), f"CCIL model not found: {config.model_path}"
        self._meta: dict = {}
        if config.meta_path is not None:
            assert os.path.exists(config.meta_path), f"CCIL meta not found: {config.meta_path}"
            with open(config.meta_path) as f:
                self._meta = json.load(f)

        self._state_mean: torch.Tensor | None = None
        self._constant_state_mask: torch.Tensor | None = None
        if "state_norm" in self._meta:
            state_norm = self._meta["state_norm"]
            self._state_mean = torch.tensor(state_norm["mean"], dtype=torch.float, device=self.device)
            state_std = torch.tensor(state_norm["std"], dtype=torch.float, device=self.device)
            # VisualBCPolicy clamps zero-variance training dimensions to 1e-6 before
            # exporting. Live simulator noise or a different reset pose in those
            # dimensions would otherwise become an enormous normalized input even
            # though the model never learned to use that feature.
            self._constant_state_mask = state_std <= 1.0e-5

        self._jit_module: torch.jit.ScriptModule | None = None
        self._mlp: torch.nn.Sequential | None = None
        self._obs_mean: torch.Tensor | None = None
        self._obs_std: torch.Tensor | None = None
        self._act_min: torch.Tensor | None = None
        self._act_max: torch.Tensor | None = None

        self._load_model(config.model_path)
        self.task_description: str | None = None

    # ---------------------- model loading -------------------

    def _load_model(self, model_path: str) -> None:
        """Load the TorchScript module if possible, else reconstruct the MLP from meta."""
        try:
            self._jit_module = torch.jit.load(model_path, map_location=self.device)
            self._jit_module.eval()
            logger.info(f"CCILBCPolicy: loaded TorchScript policy from {model_path}")
            return
        except (RuntimeError, ValueError) as e:
            logger.warning(f"CCILBCPolicy: torch.jit.load failed ({e}); falling back to state_dict + meta.")

        assert self._meta, "state_dict fallback requires --meta_path pointing at ccil_bc_meta.json"
        input_dim = int(self._meta["input_dim"])
        output_dim = int(self._meta["output_dim"])
        hidden_units = list(self._meta["hidden_units"])
        activation = str(self._meta.get("activation", "relu"))
        self._mlp = _build_mlp(input_dim, output_dim, hidden_units, activation).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        # Accept either a raw state_dict or a checkpoint dict wrapping it.
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self._mlp.load_state_dict(state_dict)
        self._mlp.eval()

        obs_norm = self._meta["obs_norm"]
        self._obs_mean = torch.tensor(obs_norm["mean"], dtype=torch.float, device=self.device)
        self._obs_std = torch.tensor(obs_norm["std"], dtype=torch.float, device=self.device)
        act_norm = self._meta["action_norm"]
        self._act_min = torch.tensor(act_norm["min"], dtype=torch.float, device=self.device)
        self._act_max = torch.tensor(act_norm["max"], dtype=torch.float, device=self.device)
        logger.info(f"CCILBCPolicy: reconstructed MLP {input_dim}->{hidden_units}->{output_dim} from {model_path}")

    def _forward(self, obs: torch.Tensor, images: torch.Tensor | None = None) -> torch.Tensor:
        """Map a raw observation batch to a raw action batch."""
        if self._jit_module is not None:
            if images is not None:
                return self._jit_module(obs, images)
            return self._jit_module(obs)
        assert images is None, "state_dict fallback only supports state-only CCIL BC policies"
        # Plain-MLP fallback: standard-scale obs, run MLP, inverse min-max scale action.
        scaled_obs = (obs - self._obs_mean) / self._obs_std
        y = self._mlp(scaled_obs)  # network output in [-1, 1] (min-max scaled action space)
        return (y + 1.0) / 2.0 * (self._act_max - self._act_min) + self._act_min

    def _prepare_camera_tensor(self, observation: GymSpacesDict, batch_size: int) -> torch.Tensor:
        """Stack configured camera observations into a float NCHW tensor."""
        if self._video_source is not None:
            frames = self._video_source.read(list(self.replay_video_keys), num_envs=batch_size)
            camera_obs = {
                image_key: torch.as_tensor(frame, device=self.device)
                for image_key, frame in zip(self.image_keys, frames)
            }
            observation = {self.camera_group_key: camera_obs}

        assert self.camera_group_key in observation, f"observation missing '{self.camera_group_key}'"
        camera_obs = observation[self.camera_group_key]
        images = []
        for key in self.image_keys:
            assert key in camera_obs, f"observation missing '{self.camera_group_key}/{key}'"
            image = camera_obs[key].to(device=self.device)
            if image.ndim == 3:
                image = image.unsqueeze(0)
            assert image.ndim == 4, f"{self.camera_group_key}/{key} must be NHWC or NCHW, got {tuple(image.shape)}"
            assert (
                image.shape[0] == batch_size
            ), f"{self.camera_group_key}/{key} batch {image.shape[0]} does not match state batch {batch_size}"
            if image.shape[-1] == 3:
                image = image.permute(0, 3, 1, 2)
            assert (
                image.shape[1] == 3
            ), f"{self.camera_group_key}/{key} must have 3 RGB channels, got {tuple(image.shape)}"
            image = image.to(dtype=torch.float)
            if image.numel() > 0 and float(image.max()) > 1.0:
                image = image / 255.0
            if tuple(image.shape[-2:]) != self.image_size:
                image = F.interpolate(image, size=self.image_size, mode="bilinear", align_corners=False)
            images.append(image)
        return torch.cat(images, dim=1)

    # ---------------------- inference -------------------

    def _joint_values_by_name(self, env: gym.Env, obs: torch.Tensor, names: list[str]) -> torch.Tensor:
        joint_names = list(env.unwrapped.scene["robot"].joint_names)
        indices = []
        for name in names:
            assert name in joint_names, f"joint '{name}' missing from robot joint_names"
            indices.append(joint_names.index(name))
        return obs[:, indices]

    def _body_quat_xyzw(self, env: gym.Env, body_candidates: list[str], batch_size: int) -> torch.Tensor:
        robot = env.unwrapped.scene["robot"]
        body_names = list(robot.body_names)
        body_name = next((name for name in body_candidates if name in body_names), None)
        if body_name is None:
            return torch.zeros(batch_size, 4, device=self.device, dtype=torch.float)
        quat = torch.as_tensor(
            robot.data.body_quat_w[:, body_names.index(body_name)],
            device=self.device,
            dtype=torch.float,
        )
        return quat

    def _test_obs_new_wrist_state(self, env: gym.Env, observation: GymSpacesDict) -> tuple[torch.Tensor, torch.Tensor]:
        from isaaclab_arena_gr00t.embodiments.alex.alex_lever_eef_frame import convert_sim_eef_state_to_dataset

        policy_obs = observation["policy"]
        if isinstance(policy_obs, torch.Tensor):
            # Alex ability-hands concatenated policy layout:
            # actions34 | robot_joint_pos49 | root_pos3 | root_quat4 |
            # left_eef_pos3 | left_eef_quat4 | right_eef_pos3 | right_eef_quat4.
            left = torch.cat([policy_obs[:, 90:93], policy_obs[:, 93:97]], dim=1)
            right = torch.cat([policy_obs[:, 97:100], policy_obs[:, 100:104]], dim=1)
        else:
            left = torch.cat([policy_obs["left_eef_pos"], policy_obs["left_eef_quat"]], dim=1)
            right = torch.cat([policy_obs["right_eef_pos"], policy_obs["right_eef_quat"]], dim=1)
        converted = convert_sim_eef_state_to_dataset(
            {
                "left_wrist_pose": left.detach().cpu().numpy(),
                "right_wrist_pose": right.detach().cpu().numpy(),
            },
            env,
        )
        return (
            torch.as_tensor(converted["left_wrist_pose"], device=self.device, dtype=torch.float),
            torch.as_tensor(converted["right_wrist_pose"], device=self.device, dtype=torch.float),
        )

    def _adapt_test_obs_new_state(
        self, env: gym.Env, observation: GymSpacesDict, raw_obs: torch.Tensor
    ) -> torch.Tensor:
        policy_obs = observation["policy"]
        if not isinstance(policy_obs, torch.Tensor):
            required = ["left_eef_pos", "left_eef_quat", "right_eef_pos", "right_eef_quat"]
            for key in required:
                assert key in policy_obs, f"test_obs_new state adapter requires observation['policy/{key}']"

        left_hand_names = [
            "left_ability_hand_index_q1",
            "left_ability_hand_index_q2",
            "left_ability_hand_middle_q1",
            "left_ability_hand_middle_q2",
            "left_ability_hand_ring_q1",
            "left_ability_hand_ring_q2",
            "left_ability_hand_pinky_q1",
            "left_ability_hand_pinky_q2",
            "left_ability_hand_thumb_q1",
            "left_ability_hand_thumb_q2",
        ]
        right_hand_names = [name.replace("left_", "right_", 1) for name in left_hand_names]
        spine_names = ["SPINE_Z", "NECK_Y"]
        batch_size = raw_obs.shape[0]
        left_wrist_pose, right_wrist_pose = self._test_obs_new_wrist_state(env, observation)
        parts = [
            left_wrist_pose,
            right_wrist_pose,
            self._body_quat_xyzw(
                env,
                ["LEFT_FOREARM_LINK", "LEFT_FOREARM", "LEFT_ELBOW_Y_LINK"],
                batch_size,
            ),
            self._body_quat_xyzw(
                env,
                ["RIGHT_FOREARM_LINK", "RIGHT_FOREARM", "RIGHT_ELBOW_Y_LINK"],
                batch_size,
            ),
            self._body_quat_xyzw(env, ["HEAD_LINK"], batch_size),
            self._joint_values_by_name(env, raw_obs, left_hand_names),
            self._joint_values_by_name(env, raw_obs, right_hand_names),
            self._joint_values_by_name(env, raw_obs, spine_names),
        ]
        adapted = torch.cat(parts, dim=1)
        assert adapted.shape[1] == 48, f"test_obs_new state adapter produced {adapted.shape[1]} dims, expected 48"
        return adapted

    def _adapt_state(self, env: gym.Env, observation: GymSpacesDict, raw_obs: torch.Tensor) -> torch.Tensor:
        if self.state_adapter is None:
            return raw_obs
        assert self.state_adapter == "test_obs_new", f"Unknown CCIL state_adapter '{self.state_adapter}'"
        if raw_obs.shape[1] == 48:
            return raw_obs
        return self._adapt_test_obs_new_state(env, observation, raw_obs)

    def _stabilize_state(self, obs: torch.Tensor) -> torch.Tensor:
        """Replace constant training features with their recorded values."""
        if self._constant_state_mask is None or not bool(self._constant_state_mask.any()):
            return obs
        assert self._state_mean is not None
        assert (
            obs.shape[-1] == self._state_mean.shape[0]
        ), f"state metadata has {self._state_mean.shape[0]} dims, observation has {obs.shape[-1]}"
        return torch.where(self._constant_state_mask.unsqueeze(0), self._state_mean.unsqueeze(0), obs)

    def _adapt_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.state_adapter == "test_obs_new" and action.shape[1] == 46:
            # test_obs_new action is [wrists14 | forearms8 | neck4 | hands20].
            # Its hand block is [left10 | right10], while Arena's Pink IK action
            # expects ABILITY_HAND_TELEOP_JOINT_ORDER (left/right interleaved).
            dataset_hands = action[:, 26:46]
            hand_block = dataset_hands[:, list(_TEST_OBS_NEW_HAND_TO_TELEOP)]
            return torch.cat([action[:, :14], hand_block], dim=1)
        return action

    def _adapt_action_for_env(self, action: torch.Tensor, env: gym.Env) -> torch.Tensor:
        action_manager = getattr(env.unwrapped, "action_manager", None)
        if (
            self.state_adapter == "test_obs_new"
            and action.shape[1] == 46
            and action_manager is not None
            and getattr(action_manager, "action", None) is not None
            and action_manager.action.shape[1] == 46
        ):
            return action
        action = self._adapt_action(action)
        if self.state_adapter == "test_obs_new" and action.shape[1] == 34:
            from isaaclab_arena_gr00t.embodiments.alex.alex_lever_eef_frame import convert_dataset_eef_action_to_sim

            converted = convert_dataset_eef_action_to_sim(action.detach().cpu().numpy(), env)
            action = torch.as_tensor(converted, device=self.device, dtype=torch.float)
        return action

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        assert "policy" in observation, "observation missing 'policy'"
        policy_obs = observation["policy"]
        if isinstance(policy_obs, torch.Tensor):
            if self.state_adapter == "test_obs_new":
                if policy_obs.shape[1] == 48:
                    raw_obs = policy_obs.to(device=self.device, dtype=torch.float)
                else:
                    assert policy_obs.shape[1] >= 83, (
                        "test_obs_new adapter expected concatenated Alex policy observation with "
                        f"at least 83 dims or native test_obs_new 48 dims, got {policy_obs.shape}"
                    )
                    raw_obs = policy_obs[:, 34:83].to(device=self.device, dtype=torch.float)
            else:
                raw_obs = policy_obs.to(device=self.device, dtype=torch.float)
        else:
            assert self.state_key in policy_obs, f"observation missing 'policy/{self.state_key}'"
            raw_obs = policy_obs[self.state_key].to(device=self.device, dtype=torch.float)
        if self._video_source is not None and self.state_adapter == "test_obs_new":
            obs = torch.as_tensor(
                self._video_source.read_state(num_envs=raw_obs.shape[0]),
                device=self.device,
                dtype=torch.float,
            )
        else:
            obs = self._stabilize_state(self._adapt_state(env, observation, raw_obs))
        with torch.inference_mode():
            images = self._prepare_camera_tensor(observation, obs.shape[0]) if self.use_images else None
            action = self._adapt_action_for_env(self._forward(obs, images), env)
        if not torch.isfinite(action).all():
            logger.warning("CCILBCPolicy: non-finite action; substituting zeros for affected entries.")
            action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        return action.to(device=env.unwrapped.device, dtype=torch.float)

    # ---------------------- CLI helpers -------------------

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("CCIL BC Policy", "Arguments for the CCIL behavioral-cloning policy.")
        group.add_argument(
            "--model_path",
            type=str,
            required=True,
            help="CCIL policy artifact (TorchScript or state_dict .pt).",
        )
        group.add_argument(
            "--meta_path",
            type=str,
            default=None,
            help="ccil_bc_meta.json (state_dict fallback / checks).",
        )
        group.add_argument(
            "--state_key",
            type=str,
            default="robot_joint_pos",
            help="observation['policy'] key fed to the policy.",
        )
        group.add_argument(
            "--state_adapter",
            type=str,
            default=None,
            help="Optional dataset-specific state/action adapter, e.g. test_obs_new.",
        )
        group.add_argument(
            "--policy_device",
            type=str,
            default="cuda",
            help="Device for policy tensor operations.",
        )
        group.add_argument(
            "--use_images",
            action="store_true",
            help="Feed camera observations to a visual policy.",
        )
        group.add_argument(
            "--camera_group_key",
            type=str,
            default="camera_obs",
            help="Top-level observation key for cameras.",
        )
        group.add_argument(
            "--image_keys",
            nargs="+",
            default=("zed_left_cam_rgb", "zed_right_cam_rgb"),
            help="Camera keys under --camera_group_key to feed to the visual policy.",
        )
        group.add_argument(
            "--image_size",
            nargs=2,
            type=int,
            default=(128, 128),
            metavar=("HEIGHT", "WIDTH"),
            help="Visual policy image size.",
        )
        group.add_argument(
            "--replay_video_dataset_path",
            type=str,
            default=None,
            help="Optional LeRobot dataset root whose recorded videos replace sim camera observations.",
        )
        group.add_argument(
            "--replay_video_episode",
            type=int,
            default=0,
            help="Episode for replay video override.",
        )
        group.add_argument(
            "--replay_video_stride",
            type=int,
            default=1,
            help="Dataset video frames to advance per policy query.",
        )
        group.add_argument(
            "--replay_video_loop",
            action="store_true",
            help="Loop replay video when exhausted.",
        )
        group.add_argument(
            "--show_replay_video",
            action="store_true",
            help="Show dockable Kit windows for the recorded LeRobot replay videos.",
        )
        group.add_argument(
            "--replay_video_keys",
            nargs="+",
            default=("cam_zed_left", "cam_zed_right"),
            help="LeRobot video modality keys to read from replay dataset.",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> CCILBCPolicy:
        return CCILBCPolicy(CCILBCPolicyConfig.from_cli_args(args))
