# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena_gr00t.streaming.gr00t_eef_ikstream_bridge import add_ikstreamer_cli_args


def add_policy_runner_arguments(parser: argparse.ArgumentParser) -> None:
    """Add policy runner specific arguments to the parser."""
    parser.add_argument(
        "--policy_type",
        type=str,
        required=True,
        help="Type of policy to use. This is either a registered policy name or a path to a policy class.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of steps to run the policy (if num_episodes is not provided)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes to run the policy (if num_steps is not provided)",
    )
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=None,
        help=(
            "Override the environment's episode length [s] so episodes time out (reset) sooner."
            " Default: use the environment's own value."
        ),
    )
    parser.add_argument(
        "--language_instruction",
        type=str,
        default=None,
        help="Language instruction for the policy. Takes precedence over the task's own description.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=False,
        help="Record an mp4 video of the rollout (uses gymnasium.wrappers.RecordVideo).",
    )
    parser.add_argument(
        "--video_dir",
        "--video-dir",
        type=str,
        default="/eval/videos",
        help="Output directory for recorded videos. Created if missing. Used with --video and/or --camera_video.",
    )
    parser.add_argument(
        "--camera_video",
        "--camera-video",
        action="store_true",
        default=False,
        help=(
            "Record one mp4 per camera in obs['camera_obs'] (what the policy actually sees)."
            " Independent of --video; use either or both."
        ),
    )
    # --- Perturbation ("poke") -------------------------------------------------
    # Apply a constant external force to a robot link for a window of steps to
    # bump the arm off its expected trajectory and test policy robustness.
    parser.add_argument(
        "--poke",
        action="store_true",
        default=False,
        help="Apply an external-force perturbation to a robot link during rollout (robustness test).",
    )
    parser.add_argument(
        "--poke_body",
        type=str,
        default="RIGHT_WRIST_X_LINK",
        help="Body name (or regex) the poke force is applied to. Default: RIGHT_WRIST_X_LINK (Alex right wrist, most distal).",
    )
    parser.add_argument(
        "--poke_force",
        type=float,
        nargs=3,
        metavar=("FX", "FY", "FZ"),
        default=[0.0, 40.0, 0.0],
        help="Poke force [N] as 'FX FY FZ'. Frame set by --poke_frame. Default: 0 40 0.",
    )
    parser.add_argument(
        "--poke_torque",
        type=float,
        nargs=3,
        metavar=("TX", "TY", "TZ"),
        default=[0.0, 0.0, 0.0],
        help="Poke torque [N*m] as 'TX TY TZ'. Default: 0 0 0.",
    )
    parser.add_argument(
        "--poke_frame",
        type=str,
        choices=["world", "body"],
        default="world",
        help="Frame the poke force/torque is expressed in. Default: world.",
    )
    parser.add_argument(
        "--poke_start_step",
        type=int,
        default=60,
        help="Per-episode step at which the poke begins. Default: 60.",
    )
    parser.add_argument(
        "--poke_duration",
        type=int,
        default=5,
        help="Number of control steps the poke force is held. Default: 5.",
    )
    parser.add_argument(
        "--poke_period",
        type=int,
        default=None,
        help="If set, repeat the poke every N per-episode steps (default: single poke per episode).",
    )
    parser.add_argument(
        "--poke_marker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a red arrow at the poked link while the poke is active. Use --no-poke_marker to disable.",
    )
    parser.add_argument(
        "--poke_random",
        action="store_true",
        default=False,
        help=(
            "Randomize the poke into a 'nudge': each env/episode draws a random horizontal"
            " (world xy) direction and a magnitude from --poke_force_range. Overrides --poke_force"
            " and disables the per-episode ramp."
        ),
    )
    parser.add_argument(
        "--poke_force_range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=[20.0, 45.0],
        help="Magnitude range [N] sampled uniformly per env/episode when --poke_random. Default: 20 45.",
    )
    parser.add_argument(
        "--poke_random_seed",
        type=int,
        default=None,
        help="Seed for the random-poke RNG (reproducible nudges). Default: unseeded.",
    )
    parser.add_argument(
        "--viz_action_targets",
        action="store_true",
        default=False,
        help=(
            "Show small spheres (blue=left, orange=right) at the wrist targets the policy's raw"
            " action commands. Only supported for the 34-dim ability-hand EEF action layout."
        ),
    )
    add_ikstreamer_cli_args(parser)
