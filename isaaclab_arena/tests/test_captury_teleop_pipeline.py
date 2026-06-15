# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Captury teleop integration with the isaacteleop pipeline stack."""

import numpy as np

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


class _FakeCapturyPoseProvider:
    """Pose provider returning a fixed standard-skeleton pose."""

    def __init__(self):
        from isaaclab_arena.teleop.captury.captury_skeleton import (
            DEFAULT_CAPTURY_JOINT_NAMES,
            default_captury_skeleton_map,
        )

        skeleton_map = default_captury_skeleton_map()
        transforms = np.zeros((len(DEFAULT_CAPTURY_JOINT_NAMES), 6))
        # Plausible standing-operator wrist poses (translation in mm).
        transforms[skeleton_map.left.wrist, 0:3] = [300.0, 1100.0, -400.0]
        transforms[skeleton_map.right.wrist, 0:3] = [-300.0, 1100.0, -400.0]
        transforms[skeleton_map.left.wrist, 3:6] = [0.0, 0.0, 0.0]
        transforms[skeleton_map.right.wrist, 3:6] = [0.0, 0.0, 0.0]
        self._transforms = transforms

    def get_latest_transforms(self) -> np.ndarray:
        return self._transforms


def _test_captury_alex_pipeline_executes(simulation_app):
    """Build the Alex PINK pipeline with a Captury source and execute it directly."""
    import torch

    from isaacteleop.retargeting_engine.interface import TensorGroup, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    from isaaclab_arena.assets.retargeter_library import _build_alex_pipeline
    from isaaclab_arena.teleop.captury.captury_hands_source import CapturyHandsSource

    hands_source = CapturyHandsSource(name="captury_hands", pose_provider=_FakeCapturyPoseProvider())
    pipeline = _build_alex_pipeline(hands_source=hands_source)

    # Provide the world_T_anchor external input, as CapturyTeleopDevice does.
    xform_tg = TensorGroup(TransformMatrix())
    xform_tg[0] = np.eye(4, dtype=np.float32)
    external_inputs = {"world_T_anchor": {ValueInput.VALUE: xform_tg}}

    result = pipeline.execute_pipeline(external_inputs)
    action = torch.from_dlpack(result["action"][0])

    assert action.shape == (14,), f"Expected 14-D Alex action, got {tuple(action.shape)}"
    assert torch.isfinite(action).all(), "Pipeline produced non-finite action values"
    # Wrist positions flow through to the EE pose targets (anchor == world here).
    left_pos = action[0:3].numpy()
    assert np.linalg.norm(left_pos) > 0.1, "Left EE position should reflect the streamed wrist pose"
    return True


def test_captury_alex_pipeline_executes():
    result = run_simulation_app_function(_test_captury_alex_pipeline_executes, headless=HEADLESS)
    assert result


def _test_captury_device_and_retargeters_registered(simulation_app):
    """The captury device and its (device, embodiment) retargeters are registered."""
    from isaaclab_arena.assets.registries import DeviceRegistry, RetargeterRegistry

    device_registry = DeviceRegistry()
    device = device_registry.get_device_by_name("captury")
    assert device is not None
    assert device.name == "captury"

    retargeter_registry = RetargeterRegistry()
    keys = retargeter_registry.get_all_keys()
    for embodiment_name in ["alex_pink", "alex_ability_hands", "alex_v2_pink", "alex_v2_ability_hands"]:
        key = retargeter_registry.convert_tuple_to_str(("captury", embodiment_name))
        assert key in keys, f"Missing captury retargeter for embodiment '{embodiment_name}' (have: {keys})"
    return True


def test_captury_device_and_retargeters_registered():
    result = run_simulation_app_function(_test_captury_device_and_retargeters_registered, headless=HEADLESS)
    assert result


def _test_captury_device_cfg_from_registry(simulation_app):
    """The registry builds a CapturyDeviceCfg for the Alex embodiment."""
    from isaaclab_arena.assets.registries import AssetRegistry, DeviceRegistry
    from isaaclab_arena.teleop.captury.captury_teleop_device import CapturyDeviceCfg

    asset_registry = AssetRegistry()
    device_registry = DeviceRegistry()

    embodiment = asset_registry.get_asset_by_name("alex_pink")()
    device = device_registry.get_device_by_name("captury")()
    device_cfg = device_registry.get_teleop_device_cfg(device, embodiment)

    assert isinstance(device_cfg, CapturyDeviceCfg)
    assert callable(device_cfg.pipeline_builder)
    assert device_cfg.target_frame_prim_path == embodiment.get_teleop_target_frame_prim_path()
    return True


def test_captury_device_cfg_from_registry():
    result = run_simulation_app_function(_test_captury_device_cfg_from_registry, headless=HEADLESS)
    assert result


def _test_captury_enables_elbow_tracking(simulation_app):
    """The Captury ability-hand retargeter adds elbow IK tasks without growing the action."""
    from isaaclab.controllers.pink_ik import LocalFrameTaskCfg, NullSpacePostureTaskCfg

    from isaaclab_arena.assets.registries import AssetRegistry, DeviceRegistry

    asset_registry = AssetRegistry()
    device_registry = DeviceRegistry()

    embodiment = asset_registry.get_asset_by_name("alex_ability_hands")()
    controller = embodiment.action_config.upper_body_ik.controller

    # Variable frame tasks (which the action feeds) before enabling: 2 -> 14-D action.
    variable_frame_tasks_before = [t for t in controller.variable_input_tasks if isinstance(t, LocalFrameTaskCfg)]
    fixed_tasks_before = len(controller.fixed_input_tasks)
    assert len(variable_frame_tasks_before) == 2

    # Building the captury device cfg enables elbow tracking on the embodiment.
    device = device_registry.get_device_by_name("captury")()
    device_registry.get_teleop_device_cfg(device, embodiment)

    elbow_frames = {
        t.frame for t in controller.fixed_input_tasks if isinstance(t, LocalFrameTaskCfg) and "ELBOW" in str(t.frame)
    }
    assert elbow_frames == {"LEFT_ELBOW_Y_LINK", "RIGHT_ELBOW_Y_LINK"}, elbow_frames
    assert len(controller.fixed_input_tasks) == fixed_tasks_before + 2

    null_space_tasks = [t for t in controller.variable_input_tasks if isinstance(t, NullSpacePostureTaskCfg)]
    assert len(null_space_tasks) == 1
    assert null_space_tasks[0].cost == 0.0

    # The action-fed (variable) frame tasks are unchanged -> action stays 14-D.
    variable_frame_tasks_after = [t for t in controller.variable_input_tasks if isinstance(t, LocalFrameTaskCfg)]
    assert len(variable_frame_tasks_after) == 2

    # Elbow tasks are position + orientation (forearm bone drives flexion).
    for task in controller.fixed_input_tasks:
        if isinstance(task, LocalFrameTaskCfg) and "ELBOW" in str(task.frame):
            assert task.orientation_cost > 0.0

    # Enabling twice is idempotent.
    embodiment.enable_teleop_elbow_tracking()
    assert len(controller.fixed_input_tasks) == fixed_tasks_before + 2
    return True


def test_captury_enables_elbow_tracking():
    result = run_simulation_app_function(_test_captury_enables_elbow_tracking, headless=HEADLESS)
    assert result


def _test_openxr_ability_hand_has_no_elbow_tasks(simulation_app):
    """Policy/OpenXR path is unaffected: no elbow tasks unless Captury enables them."""
    from isaaclab.controllers.pink_ik import LocalFrameTaskCfg

    from isaaclab_arena.assets.registries import AssetRegistry

    embodiment = AssetRegistry().get_asset_by_name("alex_ability_hands")()
    controller = embodiment.action_config.upper_body_ik.controller
    elbow_tasks = [
        t for t in controller.fixed_input_tasks if isinstance(t, LocalFrameTaskCfg) and "ELBOW" in str(t.frame)
    ]
    assert elbow_tasks == []
    return True


def test_openxr_ability_hand_has_no_elbow_tasks():
    result = run_simulation_app_function(_test_openxr_ability_hand_has_no_elbow_tasks, headless=HEADLESS)
    assert result
