# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert a LeRobot v3.0 dataset to the layout GR00T N1.6's loader expects.

LeRobot v3.0 packs many episodes into chunked parquet/mp4 files
(``data/chunk-000/file-000.parquet``, ``meta/episodes/…``, ``meta/tasks.parquet``),
but GR00T's ``LeRobotEpisodeLoader`` reads the older episode-per-file layout
(``data/chunk-000/episode_000000.parquet``, per-episode mp4s, ``meta/episodes.jsonl``,
``meta/tasks.jsonl``, ``meta/modality.json``, ``meta/stats.json`` with q01/q99).

This script splits the chunked parquets by ``episode_index``, cuts the
concatenated videos at the per-episode timestamps with ffmpeg, rewrites the
metadata files, computes GR00T-format stats, and installs a ``modality.json``
from the given template.

Usage (host or training container; needs pandas, pyarrow, numpy, ffmpeg)::

    python isaaclab_arena_gr00t/lerobot/convert_lerobot_v3_to_gr00t.py \\
        --input_dir /datasets/alex_lever \\
        --output_dir /datasets/alex_lever_gr00t \\
        --modality_template isaaclab_arena_gr00t/embodiments/alex/alex_lever_modality.json
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import shutil
import subprocess
from pathlib import Path

import pandas as pd

GR00T_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
GR00T_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CHUNKS_SIZE = 1000

# Columns whose per-dimension statistics GR00T slices for normalization.
STATS_COLUMNS = ["observation.state", "action"]


def load_v3_metadata(input_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    with open(input_dir / "meta" / "info.json") as f:
        info = json.load(f)
    version = str(info.get("codebase_version", ""))
    assert version.startswith("v3"), f"Expected a LeRobot v3.x dataset, got codebase_version={version!r}"

    episode_files = sorted((input_dir / "meta" / "episodes").glob("*/*.parquet"))
    assert episode_files, f"No episode metadata parquets under {input_dir / 'meta' / 'episodes'}"
    episodes = pd.concat([pd.read_parquet(p) for p in episode_files], ignore_index=True)
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)

    tasks = pd.read_parquet(input_dir / "meta" / "tasks.parquet")
    return info, episodes, tasks


def overwrite_action_from_state(frames: pd.DataFrame, dims: tuple[int, int]) -> None:
    """Replace ``action[:, lo:hi]`` with the same-frame ``observation.state[:, lo:hi]``.

    For datasets whose command stream was not logged for some joints (e.g. the
    all-zero ability-hand columns in alex_lever), the measured positions stand in
    as position targets. Same-frame (not next-frame) matches the recorded teleop
    convention, where logged actions equal the measured positions of the frame.
    """
    lo, hi = dims
    states = np.stack(frames["observation.state"].to_numpy())
    actions = np.stack(frames["action"].to_numpy())
    assert 0 <= lo < hi <= min(states.shape[1], actions.shape[1]), (
        f"Dims [{lo}:{hi}] out of range for state ({states.shape[1]}) / action ({actions.shape[1]})"
    )
    actions[:, lo:hi] = states[:, lo:hi]
    frames["action"] = list(actions.astype(np.float32))


def split_data_parquets(
    input_dir: Path, output_dir: Path, episodes: pd.DataFrame, action_from_state_dims: tuple[int, int] | None
) -> None:
    data_files = sorted((input_dir / "data").glob("*/*.parquet"))
    assert data_files, f"No data parquets under {input_dir / 'data'}"
    frames = pd.concat([pd.read_parquet(p) for p in data_files], ignore_index=True)

    if action_from_state_dims is not None:
        overwrite_action_from_state(frames, action_from_state_dims)

    for ep_meta in episodes.itertuples():
        ep_index = int(ep_meta.episode_index)
        ep_frames = frames[frames["episode_index"] == ep_index].sort_values("frame_index")
        assert len(ep_frames) == int(
            ep_meta.length
        ), f"Episode {ep_index}: {len(ep_frames)} frames in data, {ep_meta.length} in metadata"
        out_path = output_dir / GR00T_DATA_PATH.format(episode_chunk=ep_index // CHUNKS_SIZE, episode_index=ep_index)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ep_frames.reset_index(drop=True).to_parquet(out_path)


def count_video_frames(video_path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def split_videos(input_dir: Path, output_dir: Path, info: dict, episodes: pd.DataFrame) -> None:
    video_keys = [key for key, feat in info["features"].items() if feat.get("dtype") == "video"]
    fps = float(info["fps"])

    for video_key in video_keys:
        for _, ep in episodes.iterrows():
            ep_index = int(ep["episode_index"])
            chunk_index = int(ep[f"videos/{video_key}/chunk_index"])
            file_index = int(ep[f"videos/{video_key}/file_index"])
            from_ts = float(ep[f"videos/{video_key}/from_timestamp"])
            to_ts = float(ep[f"videos/{video_key}/to_timestamp"])

            src = input_dir / info["video_path"].format(
                video_key=video_key, chunk_index=chunk_index, file_index=file_index
            )
            dst = output_dir / GR00T_VIDEO_PATH.format(
                episode_chunk=ep_index // CHUNKS_SIZE, video_key=video_key, episode_index=ep_index
            )
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Re-encode (not stream-copy) so cuts are frame-accurate, not keyframe-snapped.
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(src),
                    "-ss",
                    f"{from_ts:.6f}",
                    "-to",
                    f"{to_ts:.6f}",
                    "-vf",
                    f"fps={fps}",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-g",
                    "2",
                    "-crf",
                    "18",
                    "-an",
                    str(dst),
                ],
                check=True,
            )

            n_frames = count_video_frames(dst)
            expected = int(ep["length"])
            assert abs(n_frames - expected) <= 1, f"{dst}: cut {n_frames} frames, episode length is {expected}"


def write_meta(
    output_dir: Path,
    info: dict,
    episodes: pd.DataFrame,
    tasks: pd.DataFrame,
    modality_template: Path,
) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_meta in episodes.itertuples(index=False):
            record = {
                "episode_index": int(ep_meta.episode_index),
                "tasks": list(ep_meta.tasks),
                "length": int(ep_meta.length),
            }
            f.write(json.dumps(record) + "\n")

    # v3 tasks.parquet is indexed by the task string with a task_index column.
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task, row in tasks.iterrows():
            f.write(json.dumps({"task_index": int(row["task_index"]), "task": task}) + "\n")

    video_keys = [key for key, feat in info["features"].items() if feat.get("dtype") == "video"]
    total_episodes = len(episodes)
    out_info = dict(info)
    out_info.update(
        codebase_version="v2.1",
        total_episodes=total_episodes,
        total_frames=int(episodes["length"].sum()),
        total_videos=total_episodes * len(video_keys),
        total_chunks=(total_episodes + CHUNKS_SIZE - 1) // CHUNKS_SIZE,
        chunks_size=CHUNKS_SIZE,
        splits={"train": f"0:{total_episodes}"},
        data_path=GR00T_DATA_PATH,
        video_path=GR00T_VIDEO_PATH,
    )
    with open(meta_dir / "info.json", "w") as f:
        json.dump(out_info, f, indent=4)

    shutil.copy(modality_template, meta_dir / "modality.json")


def write_stats(output_dir: Path) -> None:
    parquet_paths = sorted((output_dir / "data").glob("*/*.parquet"))
    frames = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)

    stats = {}
    for column in STATS_COLUMNS:
        data = np.vstack([np.asarray(x, dtype=np.float32) for x in frames[column]])
        stats[column] = {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }

    with open(output_dir / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=4)


def convert(
    input_dir: Path,
    output_dir: Path,
    modality_template: Path,
    action_from_state_dims: tuple[int, int] | None = None,
) -> None:
    assert input_dir.is_dir(), f"Input dataset not found: {input_dir}"
    assert modality_template.is_file(), f"Modality template not found: {modality_template}"
    output_dir.mkdir(parents=True, exist_ok=True)

    info, episodes, tasks = load_v3_metadata(input_dir)
    print(f"Converting {len(episodes)} episodes from {input_dir} -> {output_dir}")
    if action_from_state_dims is not None:
        print(f"Overwriting action dims [{action_from_state_dims[0]}:{action_from_state_dims[1]}] from state")
    split_data_parquets(input_dir, output_dir, episodes, action_from_state_dims)
    split_videos(input_dir, output_dir, info, episodes)
    write_meta(output_dir, info, episodes, tasks, modality_template)
    write_stats(output_dir)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=Path, required=True, help="LeRobot v3.0 dataset root.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output root for the GR00T-layout dataset.")
    parser.add_argument(
        "--modality_template",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "embodiments" / "alex" / "alex_lever_modality.json",
        help="modality.json to install into the converted dataset's meta/.",
    )
    parser.add_argument(
        "--action_from_state_dims",
        type=lambda arg: tuple(int(part) for part in arg.split(":")),
        default=None,
        help=(
            "Optional 'start:end' slice of action dims to overwrite with the same-frame state, for joints whose"
            " command stream was not recorded (alex_lever hands: 13:33). Also fixes the degenerate all-zero"
            " normalization stats those columns would otherwise produce."
        ),
    )
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir, args.modality_template, args.action_from_state_dims)


if __name__ == "__main__":
    main()
