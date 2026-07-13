# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Simulation-free tests for the lever Mimic pipeline supervisor."""

import h5py
import numpy as np
import os

import pytest

from isaaclab_arena.scripts.imitation_learning import run_lever_mimic_pipeline as pipeline


def _write_dataset(path, successes, *, annotated=False):
    with h5py.File(path, "w") as dataset:
        dataset.attrs["format_version"] = 1
        data = dataset.create_group("data")
        data.attrs["env_args"] = "{}"
        total = 0
        for index, success in enumerate(successes):
            demo = data.create_group(f"demo_{index}")
            demo.attrs["success"] = success
            demo.attrs["num_samples"] = 3
            demo.create_dataset("actions", data=np.zeros((3, 2), dtype=np.float32))
            total += 3
            if annotated:
                datagen = demo.create_group("obs/datagen_info")
                object_pose = datagen.create_group("object_pose")
                object_pose.create_dataset("lever_revolute", data=np.zeros((3, 4, 4), dtype=np.float32))
                eef_pose = datagen.create_group("eef_pose")
                target_eef_pose = datagen.create_group("target_eef_pose")
                for eef_name in ("left", "right"):
                    eef_pose.create_dataset(eef_name, data=np.zeros((3, 4, 4), dtype=np.float32))
                    target_eef_pose.create_dataset(eef_name, data=np.zeros((3, 4, 4), dtype=np.float32))
                signals = datagen.create_group("subtask_term_signals")
                signals.create_dataset("lever_engaged", data=np.array([False, True, True]))
        data.attrs["total"] = total


def test_trim_successful_dataset_is_exact_and_recomputes_total(tmp_path):
    path = tmp_path / "generated.hdf5"
    _write_dataset(path, [True, False, True, True, True])

    pipeline.trim_successful_dataset(path, 3)
    pipeline.validate_dataset(path, 3)

    with h5py.File(path, "r") as dataset:
        assert list(dataset["data"]) == ["demo_0", "demo_1", "demo_2"]
        assert dataset["data"].attrs["total"] == 9


def test_validate_annotated_dataset_checks_monotonic_signal(tmp_path):
    path = tmp_path / "annotated.hdf5"
    _write_dataset(path, [True, True], annotated=True)
    pipeline.validate_dataset(path, 2, require_annotations=True)

    with h5py.File(path, "r+") as dataset:
        signal = dataset["data/demo_1/obs/datagen_info/subtask_term_signals/lever_engaged"]
        signal[...] = np.array([False, True, False])

    try:
        pipeline.validate_dataset(path, 2, require_annotations=True)
    except RuntimeError as error:
        assert "non-monotonic" in str(error)
    else:
        raise AssertionError("invalid annotation signal should be rejected")


def test_pipeline_commands_target_lever_and_requested_counts(tmp_path):
    args = pipeline._build_parser().parse_args([
        "--work_dir",
        str(tmp_path),
        "--record_count",
        "20",
        "--generated_count",
        "400",
        "--generation_num_envs",
        "8",
    ])
    outputs = {name: tmp_path / f"{name}.hdf5" for name in ("recorded", "annotated", "generated")}
    commands = pipeline.build_commands(args, tmp_path, outputs)

    assert len(commands) == 3
    assert "record_scripted_lever_demos.py" in commands[0][1]
    assert commands[0][commands[0].index("--num_demos") + 1] == "20"
    assert "--auto" in commands[1]
    assert commands[2][commands[2].index("--generation_num_trials") + 1] == "400"
    assert commands[2][commands[2].index("--num_envs") + 1] == "8"
    assert all("alex_lever_turn" in command and "--mimic" in command for command in commands)
    assert "--spawn_pos=-0.4,-0.48682,0.94296" in commands[0]
    assert "--push_local_offset=-0.055,0.0,0.0" in commands[0]


@pytest.mark.with_subprocess
def test_lever_mimic_pipeline_one_episode_smoke(tmp_path):
    from isaaclab_arena.embodiments.alex.alex import _ABILITY_HAND_MODELS_DIR

    left_hand_urdf = os.path.join(_ABILITY_HAND_MODELS_DIR, "urdf", "abilityHand", "ability_hand_left_large.urdf")
    if not os.path.isfile(left_hand_urdf):
        pytest.skip("Ability Hand model mount is required for the lever Mimic simulation smoke test")

    assert (
        pipeline.main([
            "--work_dir",
            str(tmp_path),
            "--record_count",
            "1",
            "--generated_count",
            "1",
            "--generation_num_envs",
            "1",
            "--device",
            "cpu",
        ])
        == 0
    )
