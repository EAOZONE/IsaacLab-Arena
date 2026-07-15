# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert a GR00T-layout (LeRobot v2.1, episode-per-file) dataset to LeRobot v3.0.

``convert_hdf5_to_lerobot.py`` produces the episode-per-file layout GR00T's
``LeRobotEpisodeLoader`` reads (``data/chunk-000/episode_000000.parquet``,
per-episode mp4s, ``meta/episodes.jsonl``, ``meta/tasks.jsonl``,
``meta/modality.json``) but never writes ``meta/stats.json``, and that layout
isn't loadable by the standard ``lerobot`` package, which requires v3.0
(chunked ``data/chunk-000/file-000.parquet``, consolidated per-camera videos,
``meta/tasks.parquet``, ``meta/episodes/chunk-000/file-000.parquet``,
``meta/stats.json``). This is the reverse of ``convert_lerobot_v3_to_gr00t.py``.

Usage (needs pandas, pyarrow, numpy, opencv-python, ffmpeg)::

    python isaaclab_arena_gr00t/lerobot/convert_gr00t_to_lerobot_v3.py \\
        --input_dir /datasets/alex_lever_sim/quest_alex_empty_mimic_80/lerobot \\
        --output_dir /datasets/alex_lever_sim/quest_alex_empty_mimic_80/lerobot_v3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

V3_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
V3_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
V3_EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def load_gr00t_metadata(input_dir: Path) -> tuple[dict, list[dict], dict[int, str]]:
    with open(input_dir / "meta" / "info.json") as f:
        info = json.load(f)
    version = str(info.get("codebase_version", ""))
    assert version.startswith(
        "v2"
    ), f"Expected a GR00T-layout LeRobot v2.x dataset, got codebase_version={version!r}"

    episodes = []
    with open(input_dir / "meta" / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    episodes.sort(key=lambda ep: ep["episode_index"])

    tasks: dict[int, str] = {}
    with open(input_dir / "meta" / "tasks.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                tasks[int(row["task_index"])] = row["task"]

    return info, episodes, tasks


def _episode_data_path(info: dict, episode_index: int) -> str:
    chunks_size = int(info["chunks_size"])
    return info["data_path"].format(
        episode_chunk=episode_index // chunks_size, episode_index=episode_index
    )


def _episode_video_path(info: dict, video_key: str, episode_index: int) -> str:
    chunks_size = int(info["chunks_size"])
    return info["video_path"].format(
        episode_chunk=episode_index // chunks_size,
        video_key=video_key,
        episode_index=episode_index,
    )


def consolidate_data(
    input_dir: Path, output_dir: Path, info: dict, episodes: list[dict]
) -> pd.DataFrame:
    """Concatenate per-episode parquets (in episode order) into one v3.0 data file."""
    frames = []
    for ep in episodes:
        ep_index = int(ep["episode_index"])
        path = input_dir / _episode_data_path(info, ep_index)
        df = pd.read_parquet(path)
        assert len(df) == int(
            ep["length"]
        ), f"episode {ep_index}: {len(df)} rows in data, {ep['length']} in episodes.jsonl"
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # Per-episode "index" values are already a running count from generation, but
    # recompute defensively now that episodes are concatenated in a known order.
    combined["index"] = np.arange(len(combined), dtype=np.int64)

    out_path = output_dir / V3_DATA_PATH.format(chunk_index=0, file_index=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    return combined


def concat_videos(
    input_dir: Path,
    output_dir: Path,
    info: dict,
    episodes: list[dict],
    video_keys: list[str],
) -> dict[str, list[tuple[float, float]]]:
    """Stream-copy-concat per-episode mp4s into one file per camera.

    Returns, per video key, the (from_timestamp, to_timestamp) span of each episode
    within the concatenated file, in the same order as ``episodes``.
    """
    fps = float(info["fps"])
    timestamps: dict[str, list[tuple[float, float]]] = {key: [] for key in video_keys}

    for video_key in video_keys:
        out_path = output_dir / V3_VIDEO_PATH.format(
            video_key=video_key, chunk_index=0, file_index=0
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as list_file:
            list_path = Path(list_file.name)
            for ep in episodes:
                ep_index = int(ep["episode_index"])
                src = input_dir / _episode_video_path(info, video_key, ep_index)
                list_file.write(f"file '{src.resolve()}'\n")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(list_path),
                    "-c",
                    "copy",
                    str(out_path),
                ],
                check=True,
            )
        finally:
            list_path.unlink(missing_ok=True)

        # Same convention as the reference v3.0 dataset: from/to timestamps are
        # cumulative frame counts / fps, not re-probed from the encoded video.
        cumulative = 0.0
        for ep in episodes:
            length = int(ep["length"])
            from_ts = cumulative
            to_ts = cumulative + length / fps
            timestamps[video_key].append((from_ts, to_ts))
            cumulative = to_ts

    return timestamps


def compute_video_stats(output_dir: Path, video_keys: list[str]) -> dict[str, dict]:
    """Per-channel pixel stats (normalized to [0, 1]) over every frame of each camera."""
    stats = {}
    for video_key in video_keys:
        video_path = output_dir / V3_VIDEO_PATH.format(
            video_key=video_key, chunk_index=0, file_index=0
        )
        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"Could not open {video_path}"

        count = 0
        channel_sum = np.zeros(3, dtype=np.float64)
        channel_sumsq = np.zeros(3, dtype=np.float64)
        channel_min = np.full(3, np.inf)
        channel_max = np.full(3, -np.inf)
        height = width = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
            height, width = rgb.shape[:2]
            channel_sum += rgb.sum(axis=(0, 1))
            channel_sumsq += (rgb**2).sum(axis=(0, 1))
            channel_min = np.minimum(channel_min, rgb.min(axis=(0, 1)))
            channel_max = np.maximum(channel_max, rgb.max(axis=(0, 1)))
            count += 1
        cap.release()
        assert count > 0, f"No frames decoded from {video_path}"

        num_pixels = count * height * width
        mean = channel_sum / num_pixels
        variance = np.maximum(channel_sumsq / num_pixels - mean**2, 0.0)
        std = np.sqrt(variance)

        stats[video_key] = {
            "min": [[[float(channel_min[c])]] for c in range(3)],
            "max": [[[float(channel_max[c])]] for c in range(3)],
            "mean": [[[float(mean[c])]] for c in range(3)],
            "std": [[[float(std[c])]] for c in range(3)],
            "count": [int(num_pixels)],
        }
    return stats


def compute_tabular_stats(combined: pd.DataFrame) -> dict[str, dict]:
    """min/max/mean/std/count for every non-video column, matching the v3.0 stats.json shape."""
    stats = {}
    total = len(combined)
    for column in combined.columns:
        series = combined[column]
        sample = series.iloc[0]
        if isinstance(sample, (list, np.ndarray)):
            arr = np.stack(series.to_numpy()).astype(np.float64)
            stats[column] = {
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "count": [total],
            }
        elif series.dtype == bool:
            arr_bool = series.to_numpy()
            arr_float = arr_bool.astype(np.float64)
            stats[column] = {
                "min": [bool(arr_bool.min())],
                "max": [bool(arr_bool.max())],
                "mean": [float(arr_float.mean())],
                "std": [float(arr_float.std())],
                "count": [total],
            }
        else:
            arr = series.to_numpy().astype(np.float64)
            stats[column] = {
                "min": [float(arr.min())],
                "max": [float(arr.max())],
                "mean": [float(arr.mean())],
                "std": [float(arr.std())],
                "count": [total],
            }
    return stats


def write_meta(
    output_dir: Path,
    info: dict,
    episodes: list[dict],
    tasks: dict[int, str],
    video_keys: list[str],
    video_timestamps: dict[str, list[tuple[float, float]]],
    total_frames: int,
) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    tasks_df = pd.DataFrame(
        {"task_index": list(tasks.keys())},
        index=pd.Index(list(tasks.values()), name="task"),
    )
    tasks_df.to_parquet(meta_dir / "tasks.parquet")

    ep_records = []
    cumulative_frames = 0
    for position, ep in enumerate(episodes):
        length = int(ep["length"])
        record = {
            "episode_index": int(ep["episode_index"]),
            "tasks": list(ep["tasks"]),
            "length": length,
            "dataset_from_index": cumulative_frames,
            "dataset_to_index": cumulative_frames + length,
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        for video_key in video_keys:
            from_ts, to_ts = video_timestamps[video_key][position]
            record[f"videos/{video_key}/chunk_index"] = 0
            record[f"videos/{video_key}/file_index"] = 0
            record[f"videos/{video_key}/from_timestamp"] = from_ts
            record[f"videos/{video_key}/to_timestamp"] = to_ts
        ep_records.append(record)
        cumulative_frames += length

    episodes_df = pd.DataFrame(ep_records)
    ep_out_path = output_dir / V3_EPISODES_PATH.format(chunk_index=0, file_index=0)
    ep_out_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_df.to_parquet(ep_out_path, index=False)

    out_info = dict(info)
    out_info.pop("total_videos", None)
    out_info["codebase_version"] = "v3.0"
    out_info["total_episodes"] = len(episodes)
    out_info["total_frames"] = total_frames
    out_info["total_chunks"] = 1
    out_info["splits"] = {"train": f"0:{len(episodes)}"}
    out_info["data_path"] = V3_DATA_PATH
    out_info["video_path"] = V3_VIDEO_PATH

    features = dict(out_info["features"])
    for key, feature in features.items():
        feature = dict(feature)
        feature["fps"] = float(info["fps"])
        if feature.get("dtype") == "video":
            video_info = feature["video_info"]
            feature["video_info"] = {
                "video.fps": float(video_info.get("video.fps", info["fps"])),
                "video.codec": video_info["video.codec"],
                "video.pix_fmt": video_info["video.pix_fmt"],
                "video.is_depth_map": video_info.get("video.is_depth_map", False),
                "has_audio": video_info.get("has_audio", False),
            }
        features[key] = feature
    out_info["features"] = features

    with open(meta_dir / "info.json", "w") as f:
        json.dump(out_info, f, indent=2)
        f.write("\n")


def convert(input_dir: Path, output_dir: Path) -> None:
    assert input_dir.is_dir(), f"Input dataset not found: {input_dir}"
    output_dir.mkdir(parents=True, exist_ok=True)

    info, episodes, tasks = load_gr00t_metadata(input_dir)
    video_keys = [
        key for key, feat in info["features"].items() if feat.get("dtype") == "video"
    ]
    print(f"Converting {len(episodes)} episodes from {input_dir} -> {output_dir}")

    combined = consolidate_data(input_dir, output_dir, info, episodes)
    video_timestamps = concat_videos(input_dir, output_dir, info, episodes, video_keys)

    stats = compute_video_stats(output_dir, video_keys)
    stats.update(compute_tabular_stats(combined))
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")

    write_meta(
        output_dir, info, episodes, tasks, video_keys, video_timestamps, len(combined)
    )
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="GR00T-layout (LeRobot v2.x) dataset root.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output root for the converted LeRobot v3.0 dataset.",
    )
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
