# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: Alex standing balance RL env trains for one iteration."""

import glob
import os
import time

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess


@pytest.mark.with_subprocess
def test_alex_standing_balance_rl_train_one_iter():
    train_script = f"{TestConstants.submodules_dir}/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py"
    args = [
        TestConstants.python_path,
        train_script,
        "--headless",
        "--external_callback",
        "isaaclab_arena.environments.isaaclab_interop.environment_registration_callback",
        "--task",
        "alex_standing_balance",
        "--embodiment",
        "alex_standing_rl",
        "--num_envs",
        "64",
        "--max_iterations",
        "1",
    ]
    t_start = time.time()
    run_subprocess(args)
    log_pattern = os.path.join(TestConstants.repo_root, "logs", "rsl_rl", "alex_standing_balance", "**", "*.pt")
    checkpoints = [f for f in glob.glob(log_pattern, recursive=True) if os.path.getmtime(f) >= t_start]
    assert checkpoints, "Training completed but no checkpoint was found under logs/rsl_rl/alex_standing_balance/"
