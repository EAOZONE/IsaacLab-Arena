# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Feed a running GR00T server REAL frames from the lever_eef dataset and check it.

Purpose: separate "is the checkpoint/server good?" from "is Arena feeding good
observations?". We take a real (image, state) pair straight from the LeRobot dataset
(guaranteed in-distribution), ask the server for an action, and compare the predicted
chunk against the dataset's recorded action at the same frame (open-loop accuracy).

We then re-run the same frame with the STATE corrupted (zeros, and a sim-like hand pose)
to measure how sensitive the policy output is to the state input — this quantifies the
"state could be a big problem" hypothesis.

Run from the server's environment, e.g.::

    cd submodules/Isaac-GR00T
    uv run python ../../isaaclab_arena_gr00t/eval/probe_gr00t_dataset_replay.py \
        --dataset ../../datasets/lever_eef --frame 3400
"""

from __future__ import annotations

import argparse
import glob
import json

import cv2
import numpy as np
import pandas as pd

from gr00t.policy.server_client import PolicyClient

# 36-dim dataset state/action layout (see meta/info.json names).
LWRIST = slice(0, 7)
RWRIST = slice(7, 14)
LHAND = slice(14, 24)
RHAND = slice(24, 34)
NECK = slice(34, 36)


def _read_state(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = sorted(glob.glob(f"{dataset}/data/**/*.parquet", recursive=True))
    df = pd.concat([pd.read_parquet(p) for p in f], ignore_index=True)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(df["action"].to_numpy()).astype(np.float32)
    ep = df["episode_index"].to_numpy()
    return state, action, ep


def _read_frame(video_path: str, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    cap.release()
    assert ok, f"could not read frame {idx} from {video_path}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _build_obs(img_left, img_right, state36, task):
    return {
        "video": {
            "cam_zed_left": img_left[None, None].astype(np.uint8),
            "cam_zed_right": img_right[None, None].astype(np.uint8),
        },
        "state": {
            "left_wrist_pose": state36[LWRIST][None, None].astype(np.float32),
            "right_wrist_pose": state36[RWRIST][None, None].astype(np.float32),
            "left_hand": state36[LHAND][None, None].astype(np.float32),
            "right_hand": state36[RHAND][None, None].astype(np.float32),
            "neck": state36[NECK][None, None].astype(np.float32),
        },
        "language": {"annotation.human.action.task_description": [[task]]},
    }


def _flatten_action(action: dict) -> np.ndarray:
    """Concatenate returned action groups into the 36-dim dataset layout (B,T,36)."""
    order = ["left_wrist_pose", "right_wrist_pose", "left_hand", "right_hand", "neck"]
    return np.concatenate([np.asarray(action[k]) for k in order], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--dataset", default="../../datasets/lever_eef")
    ap.add_argument("--frame", type=int, default=None, help="Global frame index; default = mid of ep0.")
    ap.add_argument("--task", default="lever up and down")
    args = ap.parse_args()

    state, action, ep = _read_state(args.dataset)
    n = state.shape[0]
    idx = args.frame if args.frame is not None else int(np.where(ep == 0)[0].mean())
    idx = max(0, min(idx, n - 17))
    print(f"Dataset frames={n}  using global frame idx={idx} (episode {ep[idx]})")

    left = _read_frame(f"{args.dataset}/videos/observation.images.cam_zed_left/chunk-000/file-000.mp4", idx)
    right = _read_frame(f"{args.dataset}/videos/observation.images.cam_zed_right/chunk-000/file-000.mp4", idx)
    print(f"images: left {left.shape} right {right.shape} (RGB)")

    client = PolicyClient(host=args.host, port=args.port, strict=False)
    if not client.ping():
        raise SystemExit(f"Cannot reach GR00T server at {args.host}:{args.port}")

    gt_state = state[idx]
    gt_chunk = action[idx : idx + 16]  # (16, 36) recorded actions

    def query(state36, label):
        client.reset()
        act, _ = client.get_action(_build_obs(left, right, state36, args.task))
        pred = _flatten_action(act)[0]  # (16, 36)
        # open-loop error vs recorded actions for this frame
        err = np.abs(pred - gt_chunk)
        rw_pos_err = np.linalg.norm(pred[:, 7:10] - gt_chunk[:, 7:10], axis=-1)  # right wrist pos
        rh_err = np.abs(pred[:, RHAND] - gt_chunk[:, RHAND]).mean(axis=-1)  # right hand
        print(f"\n--- {label} ---")
        print(f"  right-wrist-pos L2 err over horizon (m): "
              f"mean={rw_pos_err.mean():.4f} first={rw_pos_err[0]:.4f} last={rw_pos_err[-1]:.4f}")
        print(f"  right-hand mean-abs err over horizon (rad): "
              f"mean={rh_err.mean():.4f} first={rh_err[0]:.4f} last={rh_err[-1]:.4f}")
        print(f"  overall mean-abs action err = {err.mean():.4f}")
        print(f"  pred right-wrist step0 pos = {np.round(pred[0,7:10],4)}  gt = {np.round(gt_chunk[0,7:10],4)}")
        print(f"  pred right-hand  step0     = {np.round(pred[0,RHAND],3)}")
        print(f"  gt   right-hand  step0     = {np.round(gt_chunk[0,RHAND],3)}")
        return pred

    print("\n================ IN-DISTRIBUTION (real image + real state) ================")
    print(f"gt state right-wrist pos = {np.round(gt_state[7:10],4)}")
    pred_real = query(gt_state, "real image + REAL state")

    print("\n================ STATE SENSITIVITY (same real image) ================")
    zero_state = np.zeros_like(gt_state)
    pred_zero = query(zero_state, "real image + ZERO state")

    # sim-like left-hand init pose (from AlexAbilityHand reset defaults), wrists kept real.
    sim_state = gt_state.copy()
    # sim init left hand: idx/mid/ring/pinky q1=0,q2=0.77 ; thumb q1=-1.74,q2=0
    sim_state[LHAND] = np.array([0, 0.77, 0, 0.77, 0, 0.77, 0, 0.77, -1.74, 0], np.float32)
    sim_state[RHAND] = np.array([0, 0.77, 0, 0.77, 0, 0.77, 0, 0.77, -1.74, 0], np.float32)
    pred_sim = query(sim_state, "real image + SIM-INIT hand state")

    print("\n================ SUMMARY ================")
    print(f"real vs zero  action divergence (mean abs) = {np.abs(pred_real - pred_zero).mean():.4f}")
    print(f"real vs sim   action divergence (mean abs) = {np.abs(pred_real - pred_sim).mean():.4f}")
    print("If divergence is small, the policy is vision-dominant and state errors matter little.")
    print("If large, state (frame/units/hand-init) is a real driver of the behavior gap.")


if __name__ == "__main__":
    main()
