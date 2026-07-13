# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""LeRobot-v3 dataset video source for closed-loop policy debugging.

Feeds recorded episode frames (e.g. ``H2Ozone/lever_eef``) into a GR00T closed-loop
policy in place of the live sim cameras, so a rollout can be driven by
training-distribution images while sim state/actions stay live. Useful for isolating
whether "stuck" behavior comes from the visual sim-to-real gap versus state/action
decoding.

Unlike ``DatasetVideoReader`` in ``lerobot/playback_lerobot_eef_dataset.py`` this
reader can run headless by default and advances by an arbitrary frame stride per query
to match the closed-loop server-query cadence. When ``show_ui`` is enabled, it mirrors
the recorded frames into dockable Kit windows.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path


class LerobotDatasetVideoSource:
    """Sequentially reads one episode's videos from a LeRobot-v3 dataset.

    Frames are returned per GR00T video modality key (``cam_zed_left`` ->
    ``observation.images.cam_zed_left``). Each :meth:`read` advances by ``stride``
    dataset frames. When the episode is exhausted the last frame is held (or the
    episode restarts if ``loop`` is set).
    """

    def __init__(
        self,
        dataset_root: str | Path,
        episode_index: int,
        stride: int = 1,
        loop: bool = False,
        show_ui: bool = False,
    ):
        import cv2
        import pandas as pd

        self._cv2 = cv2
        self._pd = pd
        self._root = Path(dataset_root)
        self._episode_index = int(episode_index)
        self._stride = max(1, int(stride))
        self._loop = bool(loop)
        self._show_ui = bool(show_ui)

        info_path = self._root / "meta" / "info.json"
        assert info_path.exists(), f"LeRobot info.json not found at {info_path}"
        with open(info_path) as f:
            info = json.load(f)
        self._data_path_tpl = info["data_path"]
        self._video_path_tpl = info["video_path"]
        self._video_keys = [k for k, feat in info["features"].items() if feat.get("dtype") == "video"]
        assert self._video_keys, f"No video features in {info_path}"

        ep_paths = sorted((self._root / "meta" / "episodes").glob("*/*.parquet"))
        assert ep_paths, f"No episode metadata under {self._root / 'meta' / 'episodes'}"
        episodes_meta = self._pd.concat([self._pd.read_parquet(p) for p in ep_paths], ignore_index=True).set_index(
            "episode_index"
        )
        assert (
            self._episode_index in episodes_meta.index
        ), f"Episode {self._episode_index} not in dataset {self._root} (available: {list(episodes_meta.index)[:10]}...)"
        self._meta = episodes_meta.loc[self._episode_index]
        length_col = self._meta.get("length") if hasattr(self._meta, "get") else None
        self.num_frames = int(length_col) if length_col is not None else -1

        data_path = self._root / self._data_path_tpl.format(
            chunk_index=int(self._meta["data/chunk_index"]),
            file_index=int(self._meta["data/file_index"]),
        )
        assert data_path.exists(), f"Dataset parquet file not found: {data_path}"
        episode_data = self._pd.read_parquet(data_path)
        episode_data = episode_data[episode_data["episode_index"] == self._episode_index]
        sort_key = "frame_index" if "frame_index" in episode_data else "index"
        episode_data = episode_data.sort_values(sort_key)
        assert (
            len(episode_data) == self.num_frames
        ), f"Episode {self._episode_index} metadata has {self.num_frames} frames, parquet has {len(episode_data)}"
        self._states = np.stack([np.asarray(value, dtype=np.float32) for value in episode_data["observation.state"]])

        self._captures: dict[str, object] = {}
        self._last_rgb: dict[str, np.ndarray] = {}
        self._providers: dict[str, object] = {}
        self._windows: dict[str, object] = {}
        if self._show_ui:
            import omni.ui as ui

            for index, key in enumerate(self._video_keys):
                provider = ui.ByteImageProvider()
                window = ui.Window(f"Dataset {key.split('.')[-1]}", width=480, height=380)
                window.position_x = 520 * index
                window.position_y = 120
                with window.frame:
                    ui.ImageWithProvider(provider)
                self._providers[key] = provider
                self._windows[key] = window
        self._frame_pos = 0
        self._exhausted = False
        self.start_episode()

    def _open_captures(self) -> None:
        for cap in self._captures.values():
            cap.release()
        self._captures = {}
        for key in self._video_keys:
            path = self._root / self._video_path_tpl.format(
                video_key=key,
                chunk_index=int(self._meta[f"videos/{key}/chunk_index"]),
                file_index=int(self._meta[f"videos/{key}/file_index"]),
            )
            assert path.exists(), f"Dataset video file not found: {path}"
            cap = self._cv2.VideoCapture(str(path))
            cap.set(
                self._cv2.CAP_PROP_POS_MSEC,
                float(self._meta[f"videos/{key}/from_timestamp"]) * 1000.0,
            )
            self._captures[key] = cap

    def start_episode(self) -> None:
        """(Re)open the episode videos and prime the first frame of each camera."""
        self._open_captures()
        self._frame_pos = 0
        self._exhausted = False
        for key, cap in self._captures.items():
            ok, frame = cap.read()
            assert ok, f"Could not read first frame for {key} in episode {self._episode_index}"
            self._last_rgb[key] = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        self._update_ui()

    def _update_ui(self) -> None:
        """Mirror current recorded RGB frames into Kit windows when UI display is enabled."""
        for key, provider in self._providers.items():
            frame = self._last_rgb.get(key)
            if frame is None:
                continue
            rgba = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2RGBA)
            height, width = rgba.shape[:2]
            provider.set_bytes_data(rgba.flatten().tolist(), [width, height])

    def reset(self) -> None:
        self.start_episode()

    def _advance(self) -> None:
        for key, cap in self._captures.items():
            frame = None
            for _ in range(self._stride):
                ok, raw = cap.read()
                if not ok:
                    frame = None
                    break
                frame = raw
            if frame is not None:
                self._last_rgb[key] = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            else:
                self._exhausted = True
        self._frame_pos += self._stride
        if self._exhausted and self._loop:
            self.start_episode()

    def read(self, video_keys: list[str], num_envs: int = 1) -> list[np.ndarray]:
        """Return current frames for ``video_keys`` as (num_envs, H, W, C) uint8 arrays.

        Ordered to match ``video_keys`` (the GR00T video modality key order), then
        advances the reader by ``stride`` frames for the next query.
        """
        rgb_list: list[np.ndarray] = []
        self._update_ui()
        for modality_key in video_keys:
            original_key = f"observation.images.{modality_key}"
            assert (
                original_key in self._last_rgb
            ), f"Video key '{original_key}' missing; dataset provides {list(self._last_rgb)}"
            frame = self._last_rgb[original_key]
            rgb_list.append(np.broadcast_to(frame[None, ...], (num_envs, *frame.shape)).copy())
        self._advance()
        return rgb_list

    def read_state(self, num_envs: int = 1) -> np.ndarray:
        """Return the state synchronized with the next video frame without advancing."""
        frame_index = min(self._frame_pos, len(self._states) - 1)
        state = self._states[frame_index]
        return np.broadcast_to(state[None, ...], (num_envs, state.shape[0])).copy()

    @property
    def frame_pos(self) -> int:
        return self._frame_pos

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()
        self._captures = {}
