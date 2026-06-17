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


def _make_openxr_hand_positions(curl: float) -> tuple[np.ndarray, np.ndarray]:
    """Build (positions, valid) for a right hand at the given curl (0=open, 1=fist).

    Each finger's tip extends straight out from the wrist->proximal line when
    open (proximal bend angle ~180 deg) and folds back toward the wrist/palm
    when closed (angle ~90 deg), sweeping the range the flexion mapping reads.
    """
    from isaaclab_arena.teleop.captury import captury_skeleton as cs

    pos = np.zeros((cs.NUM_OPENXR_HAND_JOINTS, 3), dtype=np.float32)
    valid = np.zeros(cs.NUM_OPENXR_HAND_JOINTS, dtype=np.uint8)
    pos[cs.OPENXR_WRIST] = (0.0, 0.0, 0.0)
    valid[cs.OPENXR_WRIST] = 1
    chains = {
        (cs.OPENXR_INDEX_PROXIMAL, cs.OPENXR_INDEX_TIP): 0.03,
        (cs.OPENXR_MIDDLE_PROXIMAL, cs.OPENXR_MIDDLE_TIP): 0.01,
        (cs.OPENXR_RING_PROXIMAL, cs.OPENXR_RING_TIP): -0.01,
        (cs.OPENXR_LITTLE_PROXIMAL, cs.OPENXR_LITTLE_TIP): -0.03,
    }
    for (prox, tip), x in chains.items():
        proximal = np.array([x, 0.05, 0.0])
        pos[prox] = proximal
        out_dir = proximal / np.linalg.norm(proximal)  # straight continuation (open)
        fold_dir = np.array([0.0, -0.6, -0.8])  # back toward wrist and into the palm (closed)
        tip_dir = (1.0 - curl) * out_dir + curl * fold_dir
        tip_dir = tip_dir / np.linalg.norm(tip_dir)
        pos[tip] = proximal + 0.09 * tip_dir
        valid[prox] = 1
        valid[tip] = 1
    return pos, valid


def test_captury_finger_flexion_tracks_open_and_closed():
    """Bend-angle flexion maps every finger (incl. ring/pinky) open->0, fist->max.

    DexPilot fingertip retargeting latches the ulnar fingers at a limit on
    markerless data; the direct flexion mapping must instead move all four
    fingers monotonically from open to closed.
    """
    from isaaclab_arena.teleop.captury.captury_skeleton import (
        ABILITY_FINGER_Q1_MAX,
        OPENXR_RING_TIP,
        captury_ability_hand_finger_q1,
    )

    open_pos, valid = _make_openxr_hand_positions(curl=0.0)
    closed_pos, _ = _make_openxr_hand_positions(curl=1.0)

    q_open = captury_ability_hand_finger_q1(open_pos, valid)
    q_closed = captury_ability_hand_finger_q1(closed_pos, valid)

    for finger in ("index", "middle", "ring", "pinky"):
        assert q_open[finger] < 0.3, f"{finger} should be near-open, got {q_open[finger]:.2f}"
        assert q_closed[finger] > 1.0, f"{finger} should curl when closed, got {q_closed[finger]:.2f}"
        assert q_closed[finger] > q_open[finger]
        assert 0.0 <= q_open[finger] <= ABILITY_FINGER_Q1_MAX
        assert 0.0 <= q_closed[finger] <= ABILITY_FINGER_Q1_MAX

    # Missing keypoints -> that finger is simply absent (caller keeps the dex value).
    valid[OPENXR_RING_TIP] = 0
    assert "ring" not in captury_ability_hand_finger_q1(open_pos, valid)


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


def _make_torso_world_matrix(position_xyz: tuple[float, float, float], yaw_deg: float) -> np.ndarray:
    """Build a (4, 4) torso world matrix with Z-up yaw only."""
    from scipy.spatial.transform import Rotation

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = Rotation.from_euler("z", np.radians(yaw_deg)).as_matrix()
    mat[:3, 3] = position_xyz
    return mat


def test_captury_world_T_anchor_follows_robot_spawn_yaw():
    """Anchor rotation must include robot spawn yaw (fridge +90 deg vs microwave 0 deg).

    Captury expresses skeleton joints relative to the operator torso, then
    applies ``world_T_anchor`` to place them in the simulation.  When Alex is
    spawned with a non-zero yaw the anchor rotation must compose that yaw;
    otherwise the operator frame stays at the static OpenXR calibration and
    arm targets appear rotated relative to the robot (the fridge-env bug).
    """
    from scipy.spatial.transform import Rotation

    from isaaclab_teleop import XrCfg

    from isaaclab_arena.embodiments.alex.alex import _ALEX_XR_ANCHOR_TORSO_PRIM_PATH
    from isaaclab_arena.teleop.captury.captury_teleop_device import CapturyDeviceCfg, CapturyTeleopDevice

    xr_cfg = XrCfg(
        anchor_pos=(0.0, 0.0, -1.0),
        anchor_rot=(0.0, 0.0, -0.70711, 0.70711),
        anchor_prim_path=_ALEX_XR_ANCHOR_TORSO_PRIM_PATH,
        fixed_anchor_height=True,
    )
    cfg = CapturyDeviceCfg(xr_cfg=xr_cfg, pipeline_builder=lambda _src: None)
    device = CapturyTeleopDevice(cfg)

    microwave_torso = _make_torso_world_matrix((-0.40, -0.1, 0.93), yaw_deg=0.0)
    fridge_torso = _make_torso_world_matrix((3.943, -1.0, 0.995), yaw_deg=90.0)
    torso_reads = iter([microwave_torso, fridge_torso])

    def _read_torso(_prim_path: str) -> np.ndarray:
        return next(torso_reads)

    device._get_prim_world_matrix = staticmethod(lambda prim_path: _read_torso(prim_path))

    mw_anchor = device._get_world_T_anchor()
    device.reset()
    fr_anchor = device._get_world_T_anchor()

    np.testing.assert_allclose(mw_anchor[:3, 3], microwave_torso[:3, 3], rtol=1e-5)
    np.testing.assert_allclose(fr_anchor[:3, 3], fridge_torso[:3, 3], rtol=1e-5)

    mw_yaw = Rotation.from_matrix(mw_anchor[:3, :3]).as_euler("xyz")[2]
    fr_yaw = Rotation.from_matrix(fr_anchor[:3, :3]).as_euler("xyz")[2]
    yaw_delta_deg = np.degrees(fr_yaw - mw_yaw)
    assert abs(yaw_delta_deg - 90.0) < 1.0, f"Expected ~90 deg yaw delta, got {yaw_delta_deg:.2f}"
