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

from isaaclab_arena_ccil.data.convert_hdf5_to_ccil import convert
from isaaclab_arena_ccil.policy.ccil_bc_policy import CCILBCPolicy, _build_mlp
from isaaclab_arena_ccil.policy.config.ccil_bc_policy_config import CCILBCPolicyConfig

OBS_DIM = 49
ACT_DIM = 34


def _write_synthetic_hdf5(path: str, num_demos: int = 3) -> None:
    """Write a minimal Arena-style HDF5 with obs/robot_joint_pos and processed_actions."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i in range(num_demos):
            t = int(rng.integers(10, 20))
            g = data.create_group(f"demo_{i}")
            obs = g.create_group("obs")
            obs.create_dataset("robot_joint_pos", data=rng.standard_normal((t, OBS_DIM)).astype(np.float32))
            g.create_dataset("processed_actions", data=rng.standard_normal((t, ACT_DIM)).astype(np.float32))


def _fake_env(device: str = "cpu"):
    return SimpleNamespace(unwrapped=SimpleNamespace(device=device))


def test_converter_shapes(tmp_path):
    hdf5 = str(tmp_path / "demo.hdf5")
    pkl = str(tmp_path / "ccil" / "demo.pkl")
    _write_synthetic_hdf5(hdf5, num_demos=4)

    trajs = convert(hdf5, pkl, state_key="robot_joint_pos", action_key="processed_actions")
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


def _save_torchscript_policy(path: str) -> torch.nn.Module:
    """Create and save a tiny TorchScript 'policy' mapping (B,49)->(B,34)."""
    torch.manual_seed(0)
    net = _build_mlp(OBS_DIM, ACT_DIM, [32, 32], "relu")
    net.eval()
    scripted = torch.jit.trace(net, torch.zeros(1, OBS_DIM))
    scripted.save(path)
    return net


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


def test_export_meta_ref_pairs_roundtrip(tmp_path):
    """Reference pairs from export must reproduce through a fresh TorchScript load (cross-version check)."""
    model_path = str(tmp_path / "policy.pt")
    _save_torchscript_policy(model_path)
    # Build a pickle to sample observations from.
    pkl = str(tmp_path / "demo.pkl")
    with open(pkl, "wb") as f:
        pickle.dump([{"observations": np.random.randn(20, OBS_DIM).astype(np.float32),
                      "actions": np.random.randn(20, ACT_DIM).astype(np.float32)}], f)

    from isaaclab_arena_ccil.training.export_bc_to_torch import _sample_observations

    obs = _sample_observations(pkl, num_ref=8, seed=1)
    module = torch.jit.load(model_path, map_location="cpu")
    with torch.inference_mode():
        actions = module(torch.from_numpy(obs)).numpy()
    meta = {"input_dim": OBS_DIM, "output_dim": ACT_DIM,
            "ref_pairs": [{"obs": obs[i].tolist(), "action": actions[i].tolist()} for i in range(obs.shape[0])]}
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
        "input_dim": OBS_DIM, "output_dim": ACT_DIM, "hidden_units": [16], "activation": "relu",
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
