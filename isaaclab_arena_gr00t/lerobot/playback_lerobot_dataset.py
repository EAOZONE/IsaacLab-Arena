# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Kinematically replay LeRobot joint-space episodes on Alex for visual inspection.

Plays the recorded joint positions of a LeRobot dataset (v3.0 chunked layout or
the GR00T episode-per-file layout) frame by frame on the Alex robot in a bare
environment, so episodes can be eyeballed before training — e.g. the
https://huggingface.co/datasets/H2Ozone/lever_fingers demos (or alex_lever).

Joints are written directly to sim (no controller in the loop): dataset motor
names map onto Alex joints (``spine_z`` -> ``SPINE_Z``, ability-hand names map
unchanged); joints absent from the dataset (legs, ``*_GRIPPER_Z``) hold their
defaults — ``*_GRIPPER_Z`` is pinned at ``+π/2`` rad to match the Ability Hand
mount (see ``LEVER_FINGERS_ABSENT_JOINT_DEFAULTS``).

Alex ZED cameras are not parented to ``HEAD_LINK`` in USD; pass ``--enable_cameras``
so they spawn and call :func:`~isaaclab_arena.embodiments.alex.alex.sync_alex_zed_cameras`
each frame (playback bypasses ``env.step``, which normally runs that event).

Usage (inside Docker, from repo root)::

    /isaac-sim/python.sh isaaclab_arena_gr00t/lerobot/playback_lerobot_dataset.py \\
        --dataset_path datasets/lever_fingers \\
        --select_episodes 0,1,2 \\
        alex_lever_teleop \\
        --embodiment alex_v2_lever_fingers_joint_pos \\
        --enable_cameras

By default plays the measured joint state (``observation.state``); pass
``--source action`` to play the commanded joint targets instead (in alex_lever
the hand action channels are all-zero, so fingers only move with ``state``).
"""

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
parser.add_argument(
    "--dataset_path",
    type=str,
    default="datasets/lever_fingers",
    help="LeRobot dataset root (contains meta/ and data/).",
)
parser.add_argument(
    "--select_episodes",
    type=lambda arg: [int(part) for part in arg.split(",")],
    default=[],
    help="Comma-separated episode indices to play (e.g. 0,3,7). Empty plays all episodes in order.",
)
parser.add_argument(
    "--source",
    type=str,
    choices=["state", "action"],
    default="state",
    help="Play measured joint state (default) or commanded action targets.",
)
parser.add_argument("--playback_speed", type=float, default=1.0, help="Speed multiplier (2.0 = twice as fast).")
parser.add_argument("--loop", action="store_true", default=False, help="Restart from the first episode when done.")
add_example_environments_cli_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import json
import numpy as np
import torch
from pathlib import Path

import pandas as pd
import warp as wp

SOURCE_FEATURE_KEYS = {"state": "observation.state", "action": "action"}


def load_episodes(dataset_root: Path, feature_key: str) -> tuple[list[tuple[int, np.ndarray]], list[str], float]:
    """Load all episodes' joint tracks plus motor names and fps from a LeRobot dataset."""
    with open(dataset_root / "meta" / "info.json") as f:
        info = json.load(f)
    motor_names = info["features"][feature_key]["names"]["motors"]
    fps = float(info["fps"])

    parquet_paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    assert parquet_paths, f"No parquet files under {dataset_root / 'data'}"
    frames = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)

    episodes = []
    for episode_index, ep_frames in frames.groupby("episode_index"):
        track = np.stack(ep_frames.sort_values("frame_index")[feature_key].to_numpy()).astype(np.float32)
        episodes.append((int(episode_index), track))
    episodes.sort(key=lambda pair: pair[0])
    return episodes, motor_names, fps


def dataset_motor_to_sim_joint(motor_name: str) -> str:
    # Ability-hand joint names match the sim URDF as-is; the actuated body joints
    # (spine/shoulder/elbow/wrist/neck) are uppercase in the Alex URDF.
    if "ability_hand" in motor_name:
        return motor_name
    return motor_name.upper()


LEVER_FINGERS_NECK_MOTOR_NAMES = ("neck_z", "neck_y")


def main():
    dataset_root = Path(args_cli.dataset_path)
    feature_key = SOURCE_FEATURE_KEYS[args_cli.source]
    episodes, motor_names, fps = load_episodes(dataset_root, feature_key)
    print(f"Loaded {len(episodes)} episodes from {dataset_root} (source: {feature_key}, {fps} fps)")

    neck_overlay_episodes: dict[int, np.ndarray] | None = None
    neck_overlay_indices: tuple[int, int] | None = None
    if args_cli.source == "action":
        state_episodes, state_motor_names, _ = load_episodes(dataset_root, "observation.state")
        neck_overlay_episodes = dict(state_episodes)
        neck_overlay_indices = tuple(state_motor_names.index(name) for name in LEVER_FINGERS_NECK_MOTOR_NAMES)
        print("Overlaying neck_z/neck_y from observation.state (omitted from dataset action).")

    selected = args_cli.select_episodes or [index for index, _ in episodes]
    episodes_by_index = dict(episodes)
    for episode_index in selected:
        assert episode_index in episodes_by_index, f"Episode {episode_index} not in dataset"

    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()
    env_cfg.recorders = {}
    env_cfg.terminations = {}

    env = gym.make(env_name, cfg=env_cfg)
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

    reapply_viewer_cfg(env)
    env = env.unwrapped

    robot = env.scene["robot"]
    sim_joint_names = [dataset_motor_to_sim_joint(name) for name in motor_names]
    joint_ids, resolved_names = robot.find_joints(sim_joint_names, preserve_order=True)
    assert (
        list(resolved_names) == sim_joint_names
    ), f"Dataset motors did not resolve 1:1 onto robot joints: {set(sim_joint_names) - set(resolved_names)}"
    # The warp-backed articulation kernels require int32 joint indices.
    joint_ids = torch.tensor(joint_ids, dtype=torch.int32, device=env.device)

    from isaaclab_arena.embodiments.alex.alex import LEVER_FINGERS_ABSENT_JOINT_DEFAULTS, sync_alex_zed_cameras

    env_ids = torch.arange(env.num_envs, device=env.device)

    def _sync_zed_cameras_to_head() -> None:
        if "zed_left_cam" not in env.scene.sensors:
            return
        sync_alex_zed_cameras(env, env_ids)

    absent_joint_names = list(LEVER_FINGERS_ABSENT_JOINT_DEFAULTS.keys())
    absent_joint_ids, absent_resolved_names = robot.find_joints(absent_joint_names, preserve_order=True)
    absent_positions = torch.tensor(
        [[LEVER_FINGERS_ABSENT_JOINT_DEFAULTS[name] for name in absent_resolved_names]],
        device=env.device,
        dtype=torch.float32,
    )
    absent_joint_ids = torch.tensor(absent_joint_ids, dtype=torch.int32, device=env.device)
    absent_zero_velocities = torch.zeros((1, len(absent_joint_ids)), device=env.device)

    neck_joint_ids = None
    neck_positions_template = None
    if neck_overlay_indices is not None:
        neck_sim_names = [dataset_motor_to_sim_joint(name) for name in LEVER_FINGERS_NECK_MOTOR_NAMES]
        neck_joint_ids_list, neck_resolved_names = robot.find_joints(neck_sim_names, preserve_order=True)
        assert list(neck_resolved_names) == neck_sim_names
        neck_joint_ids = torch.tensor(neck_joint_ids_list, dtype=torch.int32, device=env.device)
        neck_positions_template = torch.zeros((1, len(neck_joint_ids_list)), device=env.device, dtype=torch.float32)

    def _write_absent_joint_defaults() -> None:
        robot.write_joint_position_to_sim_index(position=absent_positions, joint_ids=absent_joint_ids)
        robot.write_joint_velocity_to_sim_index(velocity=absent_zero_velocities, joint_ids=absent_joint_ids)
        robot.set_joint_position_target_index(target=absent_positions, joint_ids=absent_joint_ids)

    def _write_neck_overlay(episode_index: int, frame_index: int) -> None:
        if neck_overlay_episodes is None or neck_joint_ids is None or neck_positions_template is None:
            return
        assert neck_overlay_indices is not None
        state_frame = neck_overlay_episodes[episode_index][frame_index]
        neck_positions = neck_positions_template.clone()
        neck_positions[0, 0] = float(state_frame[neck_overlay_indices[0]])
        neck_positions[0, 1] = float(state_frame[neck_overlay_indices[1]])
        robot.write_joint_position_to_sim_index(position=neck_positions, joint_ids=neck_joint_ids)
        robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(neck_positions), joint_ids=neck_joint_ids
        )
        robot.set_joint_position_target_index(target=neck_positions, joint_ids=neck_joint_ids)

    physics_dt = env.sim.get_physics_dt()
    steps_per_frame = max(1, round((1.0 / fps) / physics_dt / args_cli.playback_speed))

    env.reset()
    _write_absent_joint_defaults()
    env.scene.update(dt=physics_dt)
    _sync_zed_cameras_to_head()
    zero_velocities = torch.zeros((1, len(joint_ids)), device=env.device)

    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            for episode_index in selected:
                track = episodes_by_index[episode_index]
                print(f"Playing episode {episode_index} ({len(track)} frames, {len(track) / fps:.1f}s)", flush=True)
                env.reset()
                _write_absent_joint_defaults()
                max_tracking_error = 0.0
                for frame_index, frame in enumerate(track):
                    positions = torch.as_tensor(frame, device=env.device).unsqueeze(0)
                    robot.write_joint_position_to_sim_index(position=positions, joint_ids=joint_ids)
                    robot.write_joint_velocity_to_sim_index(velocity=zero_velocities, joint_ids=joint_ids)
                    robot.set_joint_position_target_index(target=positions, joint_ids=joint_ids)
                    _write_absent_joint_defaults()
                    _write_neck_overlay(episode_index, frame_index)
                    robot.write_data_to_sim()
                    env.scene.update(dt=physics_dt)
                    _sync_zed_cameras_to_head()
                    for _ in range(steps_per_frame):
                        env.sim.step(render=True)
                    env.scene.update(dt=(steps_per_frame - 1) * physics_dt)
                    joint_pos = wp.to_torch(robot.data.joint_pos)[:, joint_ids.long()]
                    max_tracking_error = max(max_tracking_error, (joint_pos - positions).abs().max().item())
                    if not simulation_app.is_running() or simulation_app.is_exiting():
                        break
                # How far physics (joint limits, self-collision) pulled the robot off the
                # written dataset pose — large values mean the data doesn't fit this embodiment.
                print(
                    f"Episode {episode_index} done. Max joint deviation from dataset: {max_tracking_error:.4f} rad",
                    flush=True,
                )
            if not args_cli.loop:
                break

    print("Playback finished.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
