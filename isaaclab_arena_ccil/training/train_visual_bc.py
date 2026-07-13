# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Train and export a small stereo-image behavioral-cloning policy.

Input pickle schema is produced by ``convert_hdf5_to_ccil.py --image_keys ...``:

``{"observations": (T, state_dim), "images": {key: uint8 (T,3,H,W)}, "actions": (T, act_dim)}``

The exported TorchScript module maps ``(state, stacked_images)`` to raw actions, where
``stacked_images`` is ``(B, 3 * num_cameras, H, W)`` in ``[0, 1]``. Normalization is baked
into the module so Arena inference only has to resize/stack camera tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F


class VisualBCPolicy(torch.nn.Module):
    """Compact CNN + state MLP for stereo visuomotor BC."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        image_channels: int,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer("state_mean", state_mean)
        self.register_buffer("state_std", state_std)
        self.register_buffer("action_min", action_min)
        self.register_buffer("action_max", action_max)
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(128 + state_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, action_dim),
        )

    def forward(self, state: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        scaled_action = self.forward_scaled(state, images)
        return (scaled_action + 1.0) / 2.0 * (self.action_max - self.action_min) + self.action_min

    def forward_scaled(self, state: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        # tanh-bounded to [-1, 1] (the range scaled_action is trained against) so an
        # out-of-distribution input can extrapolate to a bad prediction but can never blow
        # up past the action_min/action_max workspace bounds once rescaled in forward().
        state = (state - self.state_mean) / self.state_std
        latent = self.encoder(images)
        return torch.tanh(self.head(torch.cat([state, latent], dim=1)))


def _load_visual_pickle(path: str, image_keys: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        trajectories = pickle.load(f)
    states = np.concatenate([t["observations"] for t in trajectories], axis=0).astype(np.float32)
    actions = np.concatenate([t["actions"] for t in trajectories], axis=0).astype(np.float32)
    image_blocks = []
    for key in image_keys:
        image_blocks.append(np.concatenate([t["images"][key] for t in trajectories], axis=0))
    images = np.concatenate(image_blocks, axis=1).astype(np.float32) / 255.0
    assert states.shape[0] == actions.shape[0] == images.shape[0], "state/action/image lengths must match"
    return states, images, actions


def _to_tensors(
    states: np.ndarray,
    images: np.ndarray,
    actions: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = torch.from_numpy(states).to(device)
    image = torch.from_numpy(images).to(device)
    action = torch.from_numpy(actions).to(device)
    state_mean = state.mean(dim=0)
    state_std = state.std(dim=0).clamp_min(1.0e-6)
    action_min = action.min(dim=0).values
    action_max = action.max(dim=0).values
    scaled_action = 2.0 * (action - action_min) / (action_max - action_min).clamp_min(1.0e-6) - 1.0
    return state, image, action, scaled_action, state_mean, state_std, action_min, action_max


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/export stereo visual BC for Arena CCIL.")
    parser.add_argument("--pickle", required=True, help="Visual CCIL pickle from convert_hdf5_to_ccil.py.")
    parser.add_argument("--out_policy", required=True, help="Output TorchScript policy path.")
    parser.add_argument("--out_meta", required=True, help="Output metadata JSON path.")
    parser.add_argument("--image_keys", nargs="+", default=["zed_left_cam_rgb", "zed_right_cam_rgb"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_ref", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    states_np, images_np, actions_np = _load_visual_pickle(args.pickle, args.image_keys)
    state, image, action, scaled_action, state_mean, state_std, action_min, action_max = _to_tensors(
        states_np, images_np, actions_np, args.device
    )

    model = VisualBCPolicy(
        state.shape[1],
        action.shape[1],
        image.shape[1],
        state_mean,
        state_std,
        action_min,
        action_max,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_samples = state.shape[0]
    model.train()
    for epoch in range(args.epochs):
        permutation = torch.randperm(num_samples, device=args.device)
        losses = []
        for start in range(0, num_samples, args.batch_size):
            idx = permutation[start : start + args.batch_size]
            pred = model.forward_scaled(state[idx], image[idx])
            loss = F.mse_loss(pred, scaled_action[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"epoch {epoch + 1:04d}/{args.epochs} loss={np.mean(losses):.6f}")

    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_policy)), exist_ok=True)
    with torch.inference_mode():
        traced = torch.jit.trace(model, (state[:1], image[:1]))
        traced.save(args.out_policy)
        ref_count = min(args.num_ref, num_samples)
        ref_idx = torch.randperm(num_samples, device=args.device)[:ref_count]
        ref_actions = traced(state[ref_idx], image[ref_idx]).detach().cpu().numpy()

    meta = {
        "source": "arena_ccil_visual_bc_torchscript",
        "input_dim": int(state.shape[1]),
        "output_dim": int(action.shape[1]),
        "image_keys": list(args.image_keys),
        "image_size": [int(image.shape[2]), int(image.shape[3])],
        "image_channels": int(image.shape[1]),
        "state_norm": {"mean": state_mean.detach().cpu().tolist(), "std": state_std.detach().cpu().tolist()},
        "action_norm": {"min": action_min.detach().cpu().tolist(), "max": action_max.detach().cpu().tolist()},
        "num_ref": int(ref_count),
        "ref_pairs": [
            {
                "obs": states_np[int(i)].tolist(),
                "action": ref_actions[j].tolist(),
            }
            for j, i in enumerate(ref_idx.detach().cpu().numpy())
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_meta)), exist_ok=True)
    with open(args.out_meta, "w") as f:
        json.dump(meta, f)
    print(f"Wrote visual policy to {args.out_policy}")
    print(f"Wrote metadata to {args.out_meta}")


if __name__ == "__main__":
    main()
