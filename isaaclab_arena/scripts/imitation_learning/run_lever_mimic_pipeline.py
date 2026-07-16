# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record, annotate, and augment an Alex lever dataset with one command.

This supervisor intentionally does not launch Isaac Sim itself. Each pipeline
stage owns a separate ``SimulationApp`` and is run as a sequential child process.
"""

from __future__ import annotations

import argparse
import contextlib
import h5py
import os
import subprocess
import sys
from pathlib import Path

_REQUIRED_ANNOTATION_PATHS = (
    "obs/datagen_info/object_pose",
    "obs/datagen_info/eef_pose",
    "obs/datagen_info/target_eef_pose",
    "obs/datagen_info/subtask_term_signals",
)
_LEVER_SUCCESS_DEBOUNCE_STEPS = 50


def _sorted_demo_names(data_group: h5py.Group) -> list[str]:
    names = [name for name in data_group if name.startswith("demo_")]
    return sorted(
        names,
        key=lambda name: (
            (
                0,
                int(name.removeprefix("demo_")),
            )
            if name.removeprefix("demo_").isdigit()
            else (1, name)
        ),
    )


def validate_dataset(
    path: Path, expected_count: int, *, require_annotations: bool = False
) -> None:
    """Validate episode count, success flags, and optional Mimic annotations."""
    if not path.is_file():
        raise RuntimeError(f"Expected dataset was not created: {path}")
    with h5py.File(path, "r") as dataset:
        if "data" not in dataset:
            raise RuntimeError(f"{path} is missing its top-level 'data' group")
        demos = _sorted_demo_names(dataset["data"])
        if len(demos) != expected_count:
            raise RuntimeError(
                f"{path} contains {len(demos)} episodes; expected {expected_count}"
            )
        for name in demos:
            demo = dataset["data"][name]
            if not bool(demo.attrs.get("success", False)):
                raise RuntimeError(f"{path}::{name} is not marked successful")
            if "actions" not in demo or demo["actions"].shape[0] == 0:
                raise RuntimeError(f"{path}::{name} has no actions")
            if require_annotations:
                missing = [key for key in _REQUIRED_ANNOTATION_PATHS if key not in demo]
                if missing:
                    raise RuntimeError(
                        f"{path}::{name} is missing Mimic annotations: {missing}"
                    )
                action_count = demo["actions"].shape[0]
                for key in _REQUIRED_ANNOTATION_PATHS[:-1]:
                    group = demo[key]
                    datasets: list[h5py.Dataset] = []
                    group.visititems(
                        lambda _, item: (
                            datasets.append(item)
                            if isinstance(item, h5py.Dataset)
                            else None
                        )
                    )
                    if not datasets:
                        raise RuntimeError(
                            f"{path}::{name}/{key} contains no annotation arrays"
                        )
                    misaligned = [
                        dataset.name
                        for dataset in datasets
                        if not dataset.shape or dataset.shape[0] != action_count
                    ]
                    if misaligned:
                        raise RuntimeError(
                            f"{path}::{name} has annotation arrays not aligned to {action_count} actions: {misaligned}"
                        )
                for eef_group_name in ("eef_pose", "target_eef_pose"):
                    eef_group = demo[f"obs/datagen_info/{eef_group_name}"]
                    missing_eefs = {"left", "right"} - set(eef_group)
                    if missing_eefs:
                        raise RuntimeError(
                            f"{path}::{name} is missing {eef_group_name} for {sorted(missing_eefs)}"
                        )
                signal_group = demo["obs/datagen_info/subtask_term_signals"]
                if "lever_engaged" not in signal_group:
                    raise RuntimeError(
                        f"{path}::{name} is missing the lever_engaged signal"
                    )
                signal = signal_group["lever_engaged"][...].astype(bool).reshape(-1)
                if (
                    signal.size == 0
                    or signal[0]
                    or not signal[-1]
                    or (signal[:-1] & ~signal[1:]).any()
                ):
                    raise RuntimeError(
                        f"{path}::{name} has an invalid non-monotonic lever_engaged signal"
                    )


def trim_successful_dataset(path: Path, target_count: int) -> None:
    """Atomically retain the first ``target_count`` successful episodes."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(path, "r") as source:
        if "data" not in source:
            raise RuntimeError(f"{path} is missing its top-level 'data' group")
        successful = [
            name
            for name in _sorted_demo_names(source["data"])
            if bool(source["data"][name].attrs.get("success", False))
        ]
        if len(successful) < target_count:
            raise RuntimeError(
                f"{path} contains {len(successful)} successful episodes; expected at least {target_count}"
            )
        if len(successful) == target_count and len(successful) == len(
            _sorted_demo_names(source["data"])
        ):
            return

        with h5py.File(tmp_path, "w") as output:
            for key, value in source.attrs.items():
                output.attrs[key] = value
            output_data = output.create_group("data")
            for key, value in source["data"].attrs.items():
                output_data.attrs[key] = value
            total_steps = 0
            for index, source_name in enumerate(successful[:target_count]):
                source.copy(
                    source["data"][source_name], output_data, name=f"demo_{index}"
                )
                copied = output_data[f"demo_{index}"]
                total_steps += int(
                    copied.attrs.get("num_samples", copied["actions"].shape[0])
                )
            output_data.attrs["total"] = total_steps
    os.replace(tmp_path, path)


def _run(command: list[str], cwd: Path) -> None:
    print(f"\nRunning: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _environment_args(args: argparse.Namespace) -> list[str]:
    environment = [
        "alex_lever_turn",
        "--embodiment",
        args.embodiment,
        f"--spawn_pos={args.spawn_pos}",
        "--spawn_yaw",
        str(args.spawn_yaw),
        "--usd",
        args.usd,
        f"--usd_pos={args.usd_pos}",
        "--usd_yaw",
        str(args.usd_yaw),
        "--usd_scale",
        str(args.usd_scale),
        "--table",
        args.table,
        "--episode_length_s",
        str(args.episode_length_s),
        "--success_angle_threshold",
        str(args.success_angle_threshold),
    ]
    if args.lever_dr:
        environment.append("--lever_dr")
    if args.lever_pose_dr:
        environment.append("--lever_pose_dr")
    environment.extend(
        [
            "--lever_pose_dr_xy_jitter",
            str(args.lever_pose_dr_xy_jitter),
            "--lever_pose_dr_yaw_jitter_deg",
            str(args.lever_pose_dr_yaw_jitter_deg),
        ]
    )
    return environment


def build_commands(
    args: argparse.Namespace, repo_root: Path, outputs: dict[str, Path]
) -> list[list[str]]:
    """Build the three child-process commands for unit testing and execution."""
    scripts = repo_root / "isaaclab_arena" / "scripts" / "imitation_learning"
    common = [args.python_executable]
    simulator = ["--device", args.device, "--headless", "--mimic"]
    if args.enable_cameras:
        simulator.append("--enable_cameras")
    environment = _environment_args(args)
    object_name = Path(args.usd).stem.lower().replace("(", "_").replace(")", "_")

    record = [
        *common,
        str(scripts / "record_scripted_lever_demos.py"),
        *simulator,
        "--dataset_file",
        str(outputs["recorded"]),
        "--lever_eef_dataset_file",
        "none",
        "--num_demos",
        str(args.record_count),
        "--object_name",
        object_name,
        f"--push_local_offset={args.push_local_offset}",
        f"--push_wrist_rot_offset={args.push_wrist_rot_offset}",
        "--approach_height",
        str(args.approach_height),
        "--push_target_deg",
        str(args.push_target_deg),
        "--min_push_depth",
        str(args.min_push_depth),
        "--dwell_steps",
        str(args.dwell_steps),
        "--success_hold_steps",
        str(args.success_hold_steps),
        *environment,
    ]
    annotate = [
        *common,
        str(scripts / "annotate_demos.py"),
        *simulator,
        "--auto",
        "--input_file",
        str(outputs["recorded"]),
        "--output_file",
        str(outputs["annotated"]),
        *environment,
    ]
    generate = [
        *common,
        str(scripts / "generate_dataset.py"),
        *simulator,
        "--generation_num_trials",
        str(args.generated_count),
        "--num_envs",
        str(args.generation_num_envs),
        "--input_file",
        str(outputs["annotated"]),
        "--output_file",
        str(outputs["generated"]),
        *environment,
    ]
    return [record, annotate, generate]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work_dir", type=Path, required=True)
    parser.add_argument("--record_count", type=int, default=20)
    parser.add_argument("--generated_count", type=int, default=400)
    parser.add_argument("--generation_num_envs", type=int, default=10)
    parser.add_argument("--python_executable", default="/isaac-sim/python.sh")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--enable_cameras", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--embodiment", default="alex_v2_ability_hands")
    parser.add_argument("--spawn_pos", default="-0.4,-0.48682,0.94296")
    parser.add_argument("--spawn_yaw", type=float, default=0.0)
    parser.add_argument(
        "--usd", default="isaaclab_arena/assets/lever_sim/Lever_revolute.usd"
    )
    parser.add_argument("--usd_pos", default="0.6,0.0,0.9")
    parser.add_argument("--usd_yaw", type=float, default=0.0)
    parser.add_argument("--usd_scale", type=float, default=1.0)
    parser.add_argument("--table", default="none", choices=("none", "seattle_lab"))
    parser.add_argument(
        "--lever_dr",
        action="store_true",
        help="Enable lever pose jitter plus handle-color variation in all pipeline stages.",
    )
    parser.add_argument(
        "--lever_pose_dr",
        action="store_true",
        help="Enable lever pose jitter without enabling visual handle-color variation.",
    )
    parser.add_argument(
        "--lever_pose_dr_xy_jitter",
        type=float,
        default=0.01,
        help="Half-range for lever x/y jitter in meters when lever pose DR is enabled.",
    )
    parser.add_argument(
        "--lever_pose_dr_yaw_jitter_deg",
        type=float,
        default=5.0,
        help="Half-range for lever yaw jitter in degrees when lever pose DR is enabled.",
    )
    parser.add_argument("--episode_length_s", type=float, default=10.0)
    parser.add_argument("--success_angle_threshold", type=float, default=0.35)
    parser.add_argument("--push_local_offset", default="-0.055,0.0,0.0")
    parser.add_argument("--push_wrist_rot_offset", default="0.0,0.0,0.0,1.0")
    parser.add_argument("--approach_height", type=float, default=0.08)
    parser.add_argument("--push_target_deg", type=float, default=70.0)
    parser.add_argument("--min_push_depth", type=float, default=0.03)
    parser.add_argument("--dwell_steps", type=int, default=60)
    parser.add_argument("--success_hold_steps", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    positive_counts = (
        args.record_count,
        args.generated_count,
        args.generation_num_envs,
        args.dwell_steps,
        args.success_hold_steps,
    )
    if any(value <= 0 for value in positive_counts):
        raise ValueError(
            "episode counts, generation_num_envs, dwell_steps, and success_hold_steps must be positive"
        )
    if args.success_hold_steps < _LEVER_SUCCESS_DEBOUNCE_STEPS:
        raise ValueError(
            f"success_hold_steps must be at least {_LEVER_SUCCESS_DEBOUNCE_STEPS} so auto-annotation can verify success"
        )

    repo_root = Path(__file__).resolve().parents[3]
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "recorded": work_dir / "recorded.hdf5",
        "annotated": work_dir / "annotated.hdf5",
        "generated": work_dir / "generated.hdf5",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Pipeline outputs already exist; pass --overwrite to replace them: {existing}"
        )
    if args.overwrite:
        for path in outputs.values():
            with contextlib.suppress(FileNotFoundError):
                path.unlink()

    record, annotate, generate = build_commands(args, repo_root, outputs)
    _run(record, repo_root)
    validate_dataset(outputs["recorded"], args.record_count)
    _run(annotate, repo_root)
    validate_dataset(outputs["annotated"], args.record_count, require_annotations=True)
    _run(generate, repo_root)
    trim_successful_dataset(outputs["generated"], args.generated_count)
    validate_dataset(outputs["generated"], args.generated_count)
    print(
        f"\nPipeline complete: {outputs['generated']} ({args.generated_count} successful Mimic episodes)"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
