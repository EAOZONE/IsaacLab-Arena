# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone probe for a running GR00T policy server's action space / chunking.

This does NOT need Isaac Sim / Arena. It only needs the ``gr00t`` package, so run
it from the same environment used to launch the server, e.g.::

    cd submodules/Isaac-GR00T
    uv run python ../../isaaclab_arena_gr00t/eval/probe_gr00t_server_action.py \
        --host 127.0.0.1 --port 5555 \
        --modality-json ../../isaaclab_arena_gr00t/embodiments/alex/alex_lever_eef_modality.json

What it reports:
  1. The server's advertised modality config (video / state / action groups and the
     action horizon = len(action delta_indices)).
  2. The action dict returned for a synthetic observation: per-group key, shape,
     dtype, and per-timestep value ranges. This tells you whether:
       - the action groups match what Arena expects (left_wrist_pose, right_wrist_pose,
         left_hand, right_hand, neck),
       - the chunk horizon matches the closed-loop config (action_chunk_length, e.g. 16),
       - the chunk actually varies across the horizon (a chunk that is identical at every
         step, or all-zeros, is a strong signal something is wrong upstream).

It intentionally feeds a constant/dummy observation, so absolute action values are not
meaningful for behavior; the point is to validate *structure* and *chunking*.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from gr00t.policy.server_client import PolicyClient


def _horizon(mod_cfg) -> int:
    """Number of temporal steps advertised by a ModalityConfig."""
    delta = getattr(mod_cfg, "delta_indices", None)
    return len(delta) if delta is not None else -1


def _load_state_dims(modality_json_path: str | None) -> dict[str, int]:
    """Read per-group state widths from an Arena/GR00T modality.json (start/end)."""
    if modality_json_path is None:
        return {}
    with open(modality_json_path) as f:
        modality = json.load(f)
    dims: dict[str, int] = {}
    for group, spec in modality.get("state", {}).items():
        dims[group] = int(spec["end"]) - int(spec["start"])
    return dims


def build_dummy_observation(
    modality_configs: dict,
    state_dims: dict[str, int],
    task_description: str,
    image_hw: tuple[int, int] = (480, 640),
) -> dict:
    """Build a minimal, shape-correct observation for a single env (batch size 1)."""
    video_keys = modality_configs["video"].modality_keys
    state_keys = modality_configs["state"].modality_keys
    language_keys = modality_configs["language"].modality_keys

    video_t = _horizon(modality_configs["video"])
    state_t = _horizon(modality_configs["state"])
    h, w = image_hw

    obs: dict = {"video": {}, "state": {}, "language": {}}
    for key in video_keys:
        obs["video"][key] = np.zeros((1, video_t, h, w, 3), dtype=np.uint8)
    for key in state_keys:
        dim = state_dims.get(key)
        if dim is None:
            raise SystemExit(
                f"Missing state dim for group '{key}'. Pass --modality-json pointing at the "
                f"modality.json that defines state start/end, or --state-dims '{key}=<int>,...'."
            )
        obs["state"][key] = np.zeros((1, state_t, dim), dtype=np.float32)
    obs["language"][language_keys[0]] = [[task_description]]
    return obs


def _summarize_action(action: dict[str, np.ndarray]) -> None:
    print("\n=== ACTION returned by server ===")
    total_dim = 0
    for key, arr in action.items():
        arr = np.asarray(arr)
        total_dim += arr.shape[-1] if arr.ndim >= 1 else 0
        print(f"\n[{key}] shape={arr.shape} dtype={arr.dtype}")
        if arr.ndim == 3:  # (B, T, D)
            steps = arr.shape[1]
            # per-step L2 change vs first step, to see whether the chunk moves.
            first = arr[0, 0]
            last = arr[0, -1]
            step_delta = np.linalg.norm(arr[0] - first[None, :], axis=-1)
            print(f"    horizon(T)={steps}  dim(D)={arr.shape[-1]}")
            print(f"    step0 = {np.array2string(first, precision=4, max_line_width=200)}")
            print(f"    stepN = {np.array2string(last,  precision=4, max_line_width=200)}")
            print(f"    ||step_i - step_0||_2 across horizon = "
                  f"{np.array2string(step_delta, precision=4, max_line_width=200)}")
            if np.allclose(arr[0], first[None, :]):
                print("    WARNING: chunk is CONSTANT across the horizon (every step identical).")
            if np.allclose(arr, 0.0):
                print("    WARNING: chunk is ALL ZEROS.")
    print(f"\nTotal concatenated action dim across groups = {total_dim}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--api-token", default=None)
    parser.add_argument(
        "--modality-json",
        default=None,
        help="Path to modality.json used to size dummy state groups (start/end).",
    )
    parser.add_argument(
        "--state-dims",
        default=None,
        help="Override state dims, e.g. 'left_wrist_pose=7,right_wrist_pose=7,left_hand=10,...'.",
    )
    parser.add_argument("--task", default="Pull the lever.")
    parser.add_argument("--calls", type=int, default=2, help="Number of get_action calls.")
    parser.add_argument("--expected-horizon", type=int, default=16)
    args = parser.parse_args()

    client = PolicyClient(host=args.host, port=args.port, api_token=args.api_token, strict=False)
    if not client.ping():
        raise SystemExit(f"Cannot reach GR00T server at {args.host}:{args.port}")
    print(f"Connected to GR00T server at {args.host}:{args.port}")

    modality_configs = client.get_modality_config()
    print("\n=== MODALITY CONFIG (as advertised by server) ===")
    for name in ("video", "state", "action", "language"):
        cfg = modality_configs.get(name)
        if cfg is None:
            print(f"[{name}] <missing>")
            continue
        print(f"[{name}] horizon={_horizon(cfg)} keys={list(cfg.modality_keys)}")

    action_cfg = modality_configs.get("action")
    if action_cfg is not None:
        h = _horizon(action_cfg)
        status = "OK" if h == args.expected_horizon else "MISMATCH"
        print(
            f"\nAction horizon (chunk length) advertised = {h} "
            f"(expected {args.expected_horizon}) -> {status}"
        )

    state_dims = _load_state_dims(args.modality_json)
    if args.state_dims:
        for item in args.state_dims.split(","):
            k, v = item.split("=")
            state_dims[k.strip()] = int(v)

    obs = build_dummy_observation(modality_configs, state_dims, args.task)
    print("\n=== DUMMY OBSERVATION (fed to server) ===")
    for key, arr in obs["video"].items():
        print(f"video[{key}] shape={arr.shape} dtype={arr.dtype}")
    for key, arr in obs["state"].items():
        print(f"state[{key}] shape={arr.shape} dtype={arr.dtype}")
    print(f"language = {obs['language']}")

    prev = None
    for i in range(args.calls):
        action, info = client.get_action(obs)
        print(f"\n########## get_action call {i + 1}/{args.calls} ##########")
        _summarize_action(action)
        if info:
            print(f"info keys: {list(info.keys())}")
        if prev is not None:
            same = all(
                np.allclose(np.asarray(prev[k]), np.asarray(action[k])) for k in action
            )
            print(f"\nIdentical to previous call for same obs? {same}")
        prev = action


if __name__ == "__main__":
    main()
