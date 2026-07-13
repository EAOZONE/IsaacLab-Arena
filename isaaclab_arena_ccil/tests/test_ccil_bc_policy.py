# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CCIL BC integration (converter, export round-trip, policy load paths).

These are pure-PyTorch/numpy/h5py tests: no SimulationApp, no d3rlpy. They validate the
Arena-side artifact handling against small synthetic fixtures.
"""

from __future__ import annotations

import h5py
import json
import numpy as np
import pickle
import torch
from types import SimpleNamespace

import pandas as pd

from isaaclab_arena_ccil.data.convert_hdf5_to_ccil import convert
from isaaclab_arena_ccil.data.convert_lerobot_to_ccil import convert as convert_lerobot
from isaaclab_arena_ccil.policy.ccil_bc_policy import _TEST_OBS_NEW_HAND_TO_TELEOP, CCILBCPolicy, _build_mlp
from isaaclab_arena_ccil.policy.config.ccil_bc_policy_config import CCILBCPolicyConfig

OBS_DIM = 49
ACT_DIM = 34
IMAGE_KEYS = ("zed_left_cam_rgb", "zed_right_cam_rgb")
IMAGE_SIZE = (128, 128)


def _write_synthetic_hdf5(
    path: str,
    num_demos: int = 3,
    include_images: bool = False,
    image_extra_frame: bool = False,
) -> None:
    """Write a minimal Arena-style HDF5 with obs/robot_joint_pos plus the two action streams.

    Real recordings contain both the raw Pink IK ``actions`` (EE poses + fingers, the
    default for training) and the post-IK ``processed_actions`` (joint targets).
    """
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i in range(num_demos):
            t = int(rng.integers(10, 20))
            g = data.create_group(f"demo_{i}")
            obs = g.create_group("obs")
            obs.create_dataset(
                "robot_joint_pos",
                data=rng.standard_normal((t, OBS_DIM)).astype(np.float32),
            )
            g.create_dataset("actions", data=rng.standard_normal((t, ACT_DIM)).astype(np.float32))
            g.create_dataset(
                "processed_actions",
                data=rng.standard_normal((t, ACT_DIM)).astype(np.float32),
            )
            if include_images:
                camera_obs = g.create_group("camera_obs")
                image_t = t + 1 if image_extra_frame else t
                for key in IMAGE_KEYS:
                    frames = rng.integers(0, 255, size=(image_t, 32, 48, 3), dtype=np.uint8)
                    camera_obs.create_dataset(key, data=frames)


def _fake_env(device: str = "cpu"):
    return SimpleNamespace(unwrapped=SimpleNamespace(device=device))


def _fake_test_obs_new_env(body_quat_w: np.ndarray):
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
    joint_names = [f"unused_{i}" for i in range(17)] + left_hand_names + right_hand_names + ["SPINE_Z", "NECK_Y"]
    joint_names.extend([f"unused_tail_{i}" for i in range(OBS_DIM - len(joint_names))])
    robot = SimpleNamespace(
        joint_names=joint_names,
        body_names=["LEFT_FOREARM_LINK", "RIGHT_FOREARM_LINK", "HEAD_LINK"],
        data=SimpleNamespace(body_quat_w=body_quat_w),
    )
    return SimpleNamespace(unwrapped=SimpleNamespace(scene={"robot": robot}))


def test_converter_shapes(tmp_path):
    hdf5 = str(tmp_path / "demo.hdf5")
    pkl = str(tmp_path / "ccil" / "demo.pkl")
    _write_synthetic_hdf5(hdf5, num_demos=4)

    trajs = convert(hdf5, pkl, state_key="robot_joint_pos", action_key="actions")
    assert len(trajs) == 4
    with open(pkl, "rb") as f:
        loaded = pickle.load(f)
    assert len(loaded) == 4
    for t in loaded:
        assert set(t.keys()) == {"observations", "actions"}
        assert t["observations"].shape[1] == OBS_DIM
        assert t["actions"].shape[1] == ACT_DIM
        assert t["observations"].shape[0] == t["actions"].shape[0]
        assert np.isfinite(t["observations"]).all() and np.isfinite(t["actions"]).all()


def test_converter_with_images(tmp_path):
    hdf5 = str(tmp_path / "demo_images.hdf5")
    pkl = str(tmp_path / "ccil" / "demo_images.pkl")
    _write_synthetic_hdf5(hdf5, num_demos=2, include_images=True, image_extra_frame=True)

    trajs = convert(
        hdf5,
        pkl,
        state_key="robot_joint_pos",
        action_key="actions",
        image_keys=list(IMAGE_KEYS),
        image_size=IMAGE_SIZE,
    )
    with open(pkl, "rb") as f:
        loaded = pickle.load(f)
    assert len(loaded) == len(trajs) == 2
    for t in loaded:
        assert set(t.keys()) == {"observations", "actions", "images"}
        assert set(t["images"].keys()) == set(IMAGE_KEYS)
        for key in IMAGE_KEYS:
            image = t["images"][key]
            assert image.shape == (
                t["observations"].shape[0],
                3,
                IMAGE_SIZE[0],
                IMAGE_SIZE[1],
            )
            assert image.dtype == np.uint8


def test_lerobot_converter_shapes(tmp_path):
    lerobot_dir = tmp_path / "lerobot"
    data_dir = lerobot_dir / "data" / "chunk-000"
    meta_dir = lerobot_dir / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    rows = []
    for episode_index, length in enumerate([3, 2]):
        for frame_index in range(length):
            rows.append({
                "observation.state": np.full(OBS_DIM, episode_index + frame_index, dtype=np.float32),
                "action": np.full(ACT_DIM, frame_index, dtype=np.float32),
                "episode_index": episode_index,
                "frame_index": frame_index,
                "index": len(rows),
            })
    pd.DataFrame(rows).to_parquet(data_dir / "file-000.parquet")
    with open(meta_dir / "info.json", "w") as f:
        json.dump({"features": {}, "fps": 30}, f)

    pkl = str(tmp_path / "ccil" / "from_lerobot.pkl")
    trajs = convert_lerobot(str(lerobot_dir), None, pkl)

    assert len(trajs) == 2
    assert [t["observations"].shape[0] for t in trajs] == [3, 2]
    assert trajs[0]["observations"].shape[1] == OBS_DIM
    assert trajs[0]["actions"].shape[1] == ACT_DIM
    with open(pkl, "rb") as f:
        loaded = pickle.load(f)
    assert len(loaded) == 2


def _save_torchscript_policy(path: str) -> torch.nn.Module:
    """Create and save a tiny TorchScript 'policy' mapping (B,49)->(B,34)."""
    torch.manual_seed(0)
    net = _build_mlp(OBS_DIM, ACT_DIM, [32, 32], "relu")
    net.eval()
    scripted = torch.jit.trace(net, torch.zeros(1, OBS_DIM))
    scripted.save(path)
    return net


class _TinyVisualPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.state = torch.nn.Linear(OBS_DIM, ACT_DIM)
        self.image = torch.nn.Linear(6, ACT_DIM)

    def forward(self, obs: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(2, 3))
        return self.state(obs) + self.image(pooled)


def _save_torchscript_visual_policy(path: str) -> torch.nn.Module:
    torch.manual_seed(0)
    net = _TinyVisualPolicy()
    net.eval()
    scripted = torch.jit.trace(net, (torch.zeros(2, OBS_DIM), torch.zeros(2, 6, IMAGE_SIZE[0], IMAGE_SIZE[1])))
    scripted.save(path)
    return net


class _DummyVideoSource:
    def __init__(self, left: np.ndarray, right: np.ndarray, state: np.ndarray | None = None):
        self.left = left
        self.right = right
        self.state = state

    def read(self, video_keys: list[str], num_envs: int = 1) -> list[np.ndarray]:
        assert video_keys == ["cam_zed_left", "cam_zed_right"]
        return [
            np.broadcast_to(self.left[None, ...], (num_envs, *self.left.shape)).copy(),
            np.broadcast_to(self.right[None, ...], (num_envs, *self.right.shape)).copy(),
        ]

    def read_state(self, num_envs: int = 1) -> np.ndarray:
        assert self.state is not None
        return np.broadcast_to(self.state[None, ...], (num_envs, self.state.shape[0])).copy()


def test_torchscript_policy_load_and_infer(tmp_path):
    model_path = str(tmp_path / "policy.pt")
    ref_net = _save_torchscript_policy(model_path)

    policy = CCILBCPolicy(CCILBCPolicyConfig(model_path=model_path, policy_device="cpu", num_envs=2))
    obs = torch.randn(2, OBS_DIM)
    observation = {"policy": {"robot_joint_pos": obs}}
    action = policy.get_action(_fake_env("cpu"), observation)

    assert action.shape == (2, ACT_DIM)
    assert torch.isfinite(action).all()
    # TorchScript output must match the underlying eager module.
    with torch.inference_mode():
        expected = ref_net(obs)
    assert torch.allclose(action, expected, atol=1e-5)


def test_visual_torchscript_policy_load_and_infer(tmp_path):
    model_path = str(tmp_path / "visual_policy.pt")
    ref_net = _save_torchscript_visual_policy(model_path)

    policy = CCILBCPolicy(
        CCILBCPolicyConfig(
            model_path=model_path,
            policy_device="cpu",
            num_envs=2,
            use_images=True,
            image_keys=IMAGE_KEYS,
            image_size=IMAGE_SIZE,
        )
    )
    obs = torch.randn(2, OBS_DIM)
    left = torch.randint(0, 255, (2, 32, 48, 3), dtype=torch.uint8)
    right = torch.randint(0, 255, (2, 32, 48, 3), dtype=torch.uint8)
    observation = {
        "policy": {"robot_joint_pos": obs},
        "camera_obs": {"zed_left_cam_rgb": left, "zed_right_cam_rgb": right},
    }
    action = policy.get_action(_fake_env("cpu"), observation)

    assert action.shape == (2, ACT_DIM)
    assert torch.isfinite(action).all()
    with torch.inference_mode():
        images = policy._prepare_camera_tensor(observation, batch_size=2)
        expected = ref_net(obs, images)
    assert torch.allclose(action, expected, atol=1e-5)


def test_visual_policy_stabilizes_constant_training_state(tmp_path):
    model_path = str(tmp_path / "visual_policy.pt")
    _save_torchscript_visual_policy(model_path)
    state_mean = torch.arange(OBS_DIM, dtype=torch.float)
    state_std = torch.ones(OBS_DIM)
    state_std[[2, 7]] = 1.0e-6
    meta_path = tmp_path / "visual_meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "state_norm": {
                    "mean": state_mean.tolist(),
                    "std": state_std.tolist(),
                }
            },
            f,
        )

    policy = CCILBCPolicy(CCILBCPolicyConfig(model_path=model_path, meta_path=str(meta_path), policy_device="cpu"))
    obs = torch.full((2, OBS_DIM), 1000.0)
    stabilized = policy._stabilize_state(obs)

    assert torch.all(stabilized[:, 2] == state_mean[2])
    assert torch.all(stabilized[:, 7] == state_mean[7])
    assert torch.all(stabilized[:, 3] == 1000.0)


def test_visual_policy_replay_video_override(tmp_path):
    model_path = str(tmp_path / "visual_policy.pt")
    _save_torchscript_visual_policy(model_path)

    policy = CCILBCPolicy(
        CCILBCPolicyConfig(
            model_path=model_path,
            policy_device="cpu",
            num_envs=2,
            use_images=True,
            image_keys=IMAGE_KEYS,
            image_size=IMAGE_SIZE,
        )
    )
    left = np.full((32, 48, 3), 10, dtype=np.uint8)
    right = np.full((32, 48, 3), 20, dtype=np.uint8)
    policy._video_source = _DummyVideoSource(left, right)
    policy.replay_video_keys = ("cam_zed_left", "cam_zed_right")

    obs = torch.randn(2, OBS_DIM)
    action = policy.get_action(_fake_env("cpu"), {"policy": {"robot_joint_pos": obs}})

    assert action.shape == (2, ACT_DIM)
    assert torch.isfinite(action).all()


def test_test_obs_new_replay_uses_dataset_state(tmp_path, monkeypatch):
    class ReplayStatePolicy(torch.nn.Module):
        def forward(self, state: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
            return state[:, :ACT_DIM]

    model_path = str(tmp_path / "replay_state_policy.pt")
    model = ReplayStatePolicy()
    torch.jit.trace(
        model,
        (torch.zeros(1, 48), torch.zeros(1, 6, IMAGE_SIZE[0], IMAGE_SIZE[1])),
    ).save(model_path)
    policy = CCILBCPolicy(
        CCILBCPolicyConfig(
            model_path=model_path,
            policy_device="cpu",
            state_adapter="test_obs_new",
            use_images=True,
            image_keys=IMAGE_KEYS,
            image_size=IMAGE_SIZE,
        )
    )
    dataset_state = np.arange(48, dtype=np.float32)
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    policy._video_source = _DummyVideoSource(frame, frame, dataset_state)
    policy.replay_video_keys = ("cam_zed_left", "cam_zed_right")
    monkeypatch.setattr(policy, "_adapt_action_for_env", lambda action, env: action)

    live_state = torch.full((1, OBS_DIM), -1000.0)
    action = policy.get_action(_fake_env("cpu"), {"policy": {"robot_joint_pos": live_state}})

    assert torch.equal(action, torch.arange(ACT_DIM, dtype=torch.float).unsqueeze(0))


def test_test_obs_new_state_and_action_adapter(tmp_path, monkeypatch):
    model_path = str(tmp_path / "policy.pt")
    _save_torchscript_policy(model_path)
    policy = CCILBCPolicy(CCILBCPolicyConfig(model_path=model_path, policy_device="cpu", state_adapter="test_obs_new"))

    from isaaclab_arena_gr00t.embodiments.alex import alex_lever_eef_frame

    def _fake_sim_eef_state_to_dataset(eef_pose_policy, env):
        return {
            "left_wrist_pose": np.full((1, 7), 1.0, dtype=np.float32),
            "right_wrist_pose": np.full((1, 7), 2.0, dtype=np.float32),
        }

    def _fake_dataset_eef_action_to_sim(action_np, env):
        converted = np.asarray(action_np, dtype=np.float32).copy()
        converted[:, :14] += 100.0
        return converted

    monkeypatch.setattr(
        alex_lever_eef_frame,
        "convert_sim_eef_state_to_dataset",
        _fake_sim_eef_state_to_dataset,
    )
    monkeypatch.setattr(
        alex_lever_eef_frame,
        "convert_dataset_eef_action_to_sim",
        _fake_dataset_eef_action_to_sim,
    )
    body_quat_w = np.array([[[1.0, 0.1, 0.2, 0.3], [0.9, 0.4, 0.5, 0.6], [0.8, 0.7, 0.8, 0.9]]])
    raw_obs = torch.arange(OBS_DIM, dtype=torch.float).unsqueeze(0)
    observation = {
        "policy": {
            "left_eef_pos": torch.ones(1, 3),
            "left_eef_quat": torch.ones(1, 4) * 2,
            "right_eef_pos": torch.ones(1, 3) * 3,
            "right_eef_quat": torch.ones(1, 4) * 4,
            "robot_joint_pos": raw_obs,
        }
    }

    adapted_state = policy._adapt_state(_fake_test_obs_new_env(body_quat_w), observation, raw_obs)
    assert adapted_state.shape == (1, 48)
    assert torch.allclose(adapted_state[:, :7], torch.ones(1, 7))
    assert torch.allclose(adapted_state[:, 7:14], torch.ones(1, 7) * 2)
    assert torch.allclose(adapted_state[:, 14:18], torch.tensor([[1.0, 0.1, 0.2, 0.3]]))
    assert torch.allclose(adapted_state[:, 18:22], torch.tensor([[0.9, 0.4, 0.5, 0.6]]))
    assert torch.allclose(adapted_state[:, 22:26], torch.tensor([[0.8, 0.7, 0.8, 0.9]]))

    raw_action = torch.arange(46, dtype=torch.float).unsqueeze(0)
    adapted_action = policy._adapt_action(raw_action)
    expected_hands = raw_action[:, 26:46][:, list(_TEST_OBS_NEW_HAND_TO_TELEOP)]
    assert adapted_action.shape == (1, 34)
    assert torch.allclose(adapted_action[:, :14], raw_action[:, :14])
    assert torch.allclose(adapted_action[:, 14:], expected_hands)

    sim_action = policy._adapt_action_for_env(raw_action, _fake_test_obs_new_env(body_quat_w))
    assert sim_action.shape == (1, 34)
    assert torch.allclose(sim_action[:, :14], raw_action[:, :14] + 100.0)
    assert torch.allclose(sim_action[:, 14:], expected_hands)


def test_export_meta_ref_pairs_roundtrip(tmp_path):
    """Reference pairs from export must reproduce through a fresh TorchScript load (cross-version check)."""
    model_path = str(tmp_path / "policy.pt")
    _save_torchscript_policy(model_path)
    # Build a pickle to sample observations from.
    pkl = str(tmp_path / "demo.pkl")
    with open(pkl, "wb") as f:
        pickle.dump(
            [{
                "observations": np.random.randn(20, OBS_DIM).astype(np.float32),
                "actions": np.random.randn(20, ACT_DIM).astype(np.float32),
            }],
            f,
        )

    from isaaclab_arena_ccil.training.export_bc_to_torch import _sample_observations

    obs = _sample_observations(pkl, num_ref=8, seed=1)
    module = torch.jit.load(model_path, map_location="cpu")
    with torch.inference_mode():
        actions = module(torch.from_numpy(obs)).numpy()
    meta = {
        "input_dim": OBS_DIM,
        "output_dim": ACT_DIM,
        "ref_pairs": [{"obs": obs[i].tolist(), "action": actions[i].tolist()} for i in range(obs.shape[0])],
    }
    meta_path = str(tmp_path / "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    # Re-verify via the policy (simulates the Arena-side check).
    policy = CCILBCPolicy(CCILBCPolicyConfig(model_path=model_path, meta_path=meta_path, policy_device="cpu"))
    for pair in meta["ref_pairs"]:
        o = torch.tensor([pair["obs"]])
        a = policy.get_action(_fake_env("cpu"), {"policy": {"robot_joint_pos": o}})
        assert torch.allclose(a, torch.tensor([pair["action"]]), atol=1e-4)


def test_state_dict_fallback(tmp_path):
    """When the artifact is a plain state_dict, the policy reconstructs the MLP from meta."""
    torch.manual_seed(1)
    net = _build_mlp(OBS_DIM, ACT_DIM, [16], "relu")
    net.eval()
    sd_path = str(tmp_path / "weights.pt")
    torch.save(net.state_dict(), sd_path)

    obs_mean = np.zeros(OBS_DIM, dtype=np.float32)
    obs_std = np.ones(OBS_DIM, dtype=np.float32)
    act_min = -np.ones(ACT_DIM, dtype=np.float32)
    act_max = np.ones(ACT_DIM, dtype=np.float32)
    meta = {
        "input_dim": OBS_DIM,
        "output_dim": ACT_DIM,
        "hidden_units": [16],
        "activation": "relu",
        "obs_norm": {"mean": obs_mean.tolist(), "std": obs_std.tolist()},
        "action_norm": {"min": act_min.tolist(), "max": act_max.tolist()},
    }
    meta_path = str(tmp_path / "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    policy = CCILBCPolicy(CCILBCPolicyConfig(model_path=sd_path, meta_path=meta_path, policy_device="cpu"))
    assert policy._jit_module is None and policy._mlp is not None  # took the fallback path

    obs = torch.randn(5, OBS_DIM)
    action = policy.get_action(_fake_env("cpu"), {"policy": {"robot_joint_pos": obs}})
    assert action.shape == (5, ACT_DIM)
    # With identity obs scaling and [-1,1] action range, output == (net(obs)+1)/2*2-1 == net(obs).
    with torch.inference_mode():
        expected = net(obs)
    assert torch.allclose(action, expected, atol=1e-5)
