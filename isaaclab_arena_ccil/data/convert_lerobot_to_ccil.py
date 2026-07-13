# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert a LeRobot dataset to the CCIL trajectory-pickle format.

Supports local LeRobot directories and Hugging Face Hub dataset repos. State/action
conversion only requires parquet files; pass ``--image_keys`` to also decode LeRobot
video features into ``trajectory["images"]`` for visuomotor BC.

Example::

    /isaac-sim/python.sh isaaclab_arena_ccil/data/convert_lerobot_to_ccil.py \\
        --repo_id H2Ozone/test_obs_new \\
        --out_file /datasets/test_obs_new/ccil/test_obs_new.pkl \\
        --image_keys observation.images.cam_zed_left observation.images.cam_zed_right \\
        --output_image_keys zed_left_cam_rgb zed_right_cam_rgb
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

DEFAULT_STATE_KEY = "observation.state"
DEFAULT_ACTION_KEY = "action"
DEFAULT_IMAGE_SIZE = (128, 128)


def _resolve_lerobot_dir(lerobot_dir: str | None, repo_id: str | None) -> Path:
    """Return a local LeRobot directory, downloading from HF Hub when requested."""
    assert lerobot_dir or repo_id, "Provide either --lerobot_dir or --repo_id"
    if lerobot_dir:
        path = Path(lerobot_dir).expanduser()
        assert path.exists(), f"{path} does not exist"
        return path

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _read_dataframes(lerobot_dir: Path) -> pd.DataFrame:
    data_files = sorted((lerobot_dir / "data").glob("**/*.parquet"))
    assert data_files, f"No parquet files found under {lerobot_dir / 'data'}"
    frames = [pd.read_parquet(path) for path in data_files]
    data = pd.concat(frames, ignore_index=True)
    assert "episode_index" in data.columns, "LeRobot data must contain episode_index"
    return data.sort_values(["episode_index", "frame_index" if "frame_index" in data.columns else "index"])


def _read_episodes(lerobot_dir: Path) -> pd.DataFrame | None:
    episode_files = sorted((lerobot_dir / "meta" / "episodes").glob("**/*.parquet"))
    if episode_files:
        return pd.concat([pd.read_parquet(path) for path in episode_files], ignore_index=True)
    jsonl_path = lerobot_dir / "meta" / "episodes.jsonl"
    if jsonl_path.exists():
        return pd.read_json(jsonl_path, lines=True)
    return None


def _stack_column(frame: pd.DataFrame, key: str) -> np.ndarray:
    assert key in frame.columns, f"{key} missing from columns {list(frame.columns)}"
    values = frame[key].to_list()
    return np.asarray([np.asarray(v, dtype=np.float32) for v in values], dtype=np.float32)


def _resize_rgb_frames(frames: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    assert frames.ndim == 4 and frames.shape[-1] == 3, f"expected NHWC RGB frames, got {frames.shape}"
    tensor = torch.as_tensor(frames).float().permute(0, 3, 1, 2)
    if tuple(tensor.shape[-2:]) != image_size:
        tensor = F.interpolate(tensor, size=image_size, mode="bilinear", align_corners=False)
    return tensor.clamp(0, 255).round().to(torch.uint8).cpu().numpy()


def _decode_video(path: Path) -> np.ndarray:
    assert path.exists(), f"Video file not found: {path}"
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"Could not open video: {path}"
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    assert frames, f"No frames decoded from {path}"
    return np.asarray(frames, dtype=np.uint8)


def _video_path_from_info(lerobot_dir: Path, info: dict[str, Any], video_key: str, row: pd.Series) -> Path:
    template = info.get("video_path")
    if template:
        values = {
            "video_key": video_key,
            "chunk_index": int(row.get(f"videos/{video_key}/chunk_index", row.get("data/chunk_index", 0))),
            "file_index": int(row.get(f"videos/{video_key}/file_index", row.get("data/file_index", 0))),
            "episode_index": int(row["episode_index"]),
            "episode_chunk": int(row.get(f"videos/{video_key}/chunk_index", row.get("data/chunk_index", 0))),
        }
        path = lerobot_dir / template.format(**values)
        if path.exists():
            return path

    candidates = sorted((lerobot_dir / "videos" / video_key).glob(f"**/episode_{int(row['episode_index']):06d}.mp4"))
    if candidates:
        return candidates[0]
    candidates = sorted((lerobot_dir / "videos" / video_key).glob("**/*.mp4"))
    assert candidates, f"No video files found for {video_key}"
    return candidates[0]


def _episode_video_frames(
    lerobot_dir: Path,
    info: dict[str, Any],
    episode_row: pd.Series,
    video_key: str,
    length: int,
    cache: dict[Path, np.ndarray],
) -> np.ndarray:
    video_path = _video_path_from_info(lerobot_dir, info, video_key, episode_row)
    if video_path not in cache:
        cache[video_path] = _decode_video(video_path)
    frames = cache[video_path]

    from_idx = episode_row.get("dataset_from_index")
    to_idx = episode_row.get("dataset_to_index")
    if from_idx is not None and to_idx is not None and frames.shape[0] >= int(to_idx):
        frames = frames[int(from_idx) : int(to_idx)]
    elif frames.shape[0] != length:
        start_s = episode_row.get(f"videos/{video_key}/from_timestamp")
        fps = info["features"].get(video_key, {}).get("fps", info.get("fps", 30))
        if start_s is not None:
            start = int(round(float(start_s) * float(fps)))
            frames = frames[start : start + length]

    assert frames.shape[0] >= length, f"{video_key} has {frames.shape[0]} frames, need {length}"
    return frames[:length]


def convert(
    lerobot_dir: str | None,
    repo_id: str | None,
    out_file: str,
    state_key: str = DEFAULT_STATE_KEY,
    action_key: str = DEFAULT_ACTION_KEY,
    image_keys: list[str] | None = None,
    output_image_keys: list[str] | None = None,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> list[dict[str, np.ndarray]]:
    """Convert a LeRobot dataset to CCIL trajectories and pickle it."""
    root = _resolve_lerobot_dir(lerobot_dir, repo_id)
    info = _load_json(root / "meta" / "info.json") if (root / "meta" / "info.json").exists() else {"features": {}}
    data = _read_dataframes(root)
    episodes = _read_episodes(root)

    if image_keys:
        output_image_keys = output_image_keys or image_keys
        assert len(output_image_keys) == len(image_keys), "--output_image_keys must match --image_keys length"
    else:
        output_image_keys = []

    trajectories: list[dict[str, np.ndarray]] = []
    video_cache: dict[Path, np.ndarray] = {}
    for episode_index, frame in data.groupby("episode_index", sort=True):
        observations = _stack_column(frame, state_key)
        actions = _stack_column(frame, action_key)
        assert observations.shape[0] == actions.shape[0], (
            f"episode {episode_index} observations/actions length mismatch: "
            f"{observations.shape[0]} != {actions.shape[0]}"
        )
        trajectory = {"observations": observations, "actions": actions}

        if image_keys:
            assert episodes is not None, "image conversion requires LeRobot episode metadata"
            matching = episodes[episodes["episode_index"] == episode_index]
            assert len(matching) == 1, f"episode metadata missing for episode {episode_index}"
            episode_row = matching.iloc[0]
            images = {}
            for input_key, output_key in zip(image_keys, output_image_keys):
                frames = _episode_video_frames(root, info, episode_row, input_key, len(frame), video_cache)
                images[output_key] = _resize_rgb_frames(frames, image_size)
            trajectory["images"] = images

        trajectories.append(trajectory)

    obs_dim = trajectories[0]["observations"].shape[1]
    act_dim = trajectories[0]["actions"].shape[1]
    total_steps = sum(t["observations"].shape[0] for t in trajectories)
    for i, trajectory in enumerate(trajectories):
        assert trajectory["observations"].shape[1] == obs_dim, f"episode {i} obs dim changed"
        assert trajectory["actions"].shape[1] == act_dim, f"episode {i} action dim changed"
        assert np.isfinite(trajectory["observations"]).all(), f"episode {i} has non-finite observations"
        assert np.isfinite(trajectory["actions"]).all(), f"episode {i} has non-finite actions"

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"Wrote {len(trajectories)} trajectories ({total_steps} steps) to {out_file}")
    print(f"  observations: (T, {obs_dim})   actions: (T, {act_dim})")
    if image_keys:
        print(f"  images: {output_image_keys} -> (T, 3, {image_size[0]}, {image_size[1]}) uint8")
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LeRobot dataset to CCIL trajectory pickle.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--lerobot_dir", default=None, help="Local LeRobot dataset directory.")
    source.add_argument("--repo_id", default=None, help="Hugging Face dataset repo id, e.g. H2Ozone/test_obs_new.")
    parser.add_argument("--out_file", required=True, help="Output CCIL pickle path.")
    parser.add_argument("--state_key", default=DEFAULT_STATE_KEY, help="LeRobot state column.")
    parser.add_argument("--action_key", default=DEFAULT_ACTION_KEY, help="LeRobot action column.")
    parser.add_argument("--image_keys", nargs="*", default=None, help="LeRobot video feature keys to decode.")
    parser.add_argument(
        "--output_image_keys", nargs="*", default=None, help="Names to store under trajectory['images']."
    )
    parser.add_argument(
        "--image_size",
        nargs=2,
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        metavar=("HEIGHT", "WIDTH"),
        help="Output image size for --image_keys.",
    )
    args = parser.parse_args()
    convert(
        args.lerobot_dir,
        args.repo_id,
        args.out_file,
        args.state_key,
        args.action_key,
        args.image_keys,
        args.output_image_keys,
        tuple(args.image_size),
    )


if __name__ == "__main__":
    main()
