# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for XR anchor configuration on Alex V1 and V2 teleop embodiments."""

import numpy as np

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True

_EXPECTED_ANCHOR_POS = (0.0, 0.0, -1.0)
_EXPECTED_ANCHOR_ROT = (0.0, 0.0, -0.70711, 0.70711)


def _assert_alex_xr_cfg(embodiment_name: str, simulation_app) -> bool:
    from isaaclab_teleop.xr_cfg import XrAnchorRotationMode

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.alex.alex import _ALEX_XR_ANCHOR_TORSO_PRIM_PATH
    from isaaclab_arena.utils.pose import Pose

    asset_registry = AssetRegistry()
    embodiment = asset_registry.get_asset_by_name(embodiment_name)()
    xr_cfg = embodiment.get_xr_cfg()

    np.testing.assert_allclose(
        xr_cfg.anchor_pos,
        _EXPECTED_ANCHOR_POS,
        rtol=1e-5,
        err_msg=f"{embodiment_name}: anchor_pos expected {_EXPECTED_ANCHOR_POS}, got {xr_cfg.anchor_pos}",
    )
    np.testing.assert_allclose(
        xr_cfg.anchor_rot,
        _EXPECTED_ANCHOR_ROT,
        rtol=1e-5,
        err_msg=f"{embodiment_name}: anchor_rot expected {_EXPECTED_ANCHOR_ROT}, got {xr_cfg.anchor_rot}",
    )
    assert xr_cfg.anchor_prim_path == _ALEX_XR_ANCHOR_TORSO_PRIM_PATH, (
        f"{embodiment_name}: anchor_prim_path expected {_ALEX_XR_ANCHOR_TORSO_PRIM_PATH}, "
        f"got {xr_cfg.anchor_prim_path}"
    )
    assert (
        embodiment.get_teleop_target_frame_prim_path() is None
    ), f"{embodiment_name}: teleop should leave hand poses in world frame for Pink IK"
    assert (
        xr_cfg.fixed_anchor_height is False
    ), f"{embodiment_name}: fixed_anchor_height should be False"
    assert (
        xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.CUSTOM
    ), f"{embodiment_name}: anchor_rotation_mode should use custom absolute torso yaw"
    assert (
        xr_cfg.anchor_rotation_custom_func is not None
    ), f"{embodiment_name}: anchor_rotation_custom_func should be configured"
    yaw = np.deg2rad(30.0)
    torso_pose = np.array(
        [0.0, 0.0, 1.2, 0.0, 0.0, np.sin(yaw * 0.5), np.cos(yaw * 0.5)]
    )
    anchored_rot = xr_cfg.anchor_rotation_custom_func(np.zeros(7), torso_pose)
    expected_rot = np.array([0.0, 0.0, -0.5, np.sqrt(3.0) / 2.0])
    np.testing.assert_allclose(
        anchored_rot,
        expected_rot,
        atol=1e-5,
        err_msg=f"{embodiment_name}: custom XR anchor should apply absolute torso yaw",
    )
    bad_rot = xr_cfg.anchor_rotation_custom_func(np.zeros(7), np.zeros(7))
    np.testing.assert_allclose(
        bad_rot,
        _EXPECTED_ANCHOR_ROT,
        rtol=1e-5,
        err_msg=f"{embodiment_name}: custom XR anchor should fall back to fixed anchor rotation",
    )

    robot_pose = Pose(position_xyz=(0.5, 1.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    embodiment.set_initial_pose(robot_pose)
    xr_cfg_after = embodiment.get_xr_cfg()
    np.testing.assert_allclose(
        xr_cfg_after.anchor_pos,
        _EXPECTED_ANCHOR_POS,
        rtol=1e-5,
        err_msg=f"{embodiment_name}: anchor_pos should stay fixed after set_initial_pose",
    )
    np.testing.assert_allclose(
        xr_cfg_after.anchor_rot,
        _EXPECTED_ANCHOR_ROT,
        rtol=1e-5,
        err_msg=f"{embodiment_name}: anchor_rot should stay fixed after set_initial_pose",
    )

    return True


def _test_alex_v1_ability_hands_xr_anchor(simulation_app) -> bool:
    return _assert_alex_xr_cfg("alex_ability_hands", simulation_app)


def _test_alex_v2_ability_hands_xr_anchor(simulation_app) -> bool:
    return _assert_alex_xr_cfg("alex_v2_ability_hands", simulation_app)


def _test_alex_v1_pink_xr_anchor(simulation_app) -> bool:
    return _assert_alex_xr_cfg("alex_pink", simulation_app)


def _test_alex_v2_pink_xr_anchor(simulation_app) -> bool:
    return _assert_alex_xr_cfg("alex_v2_pink", simulation_app)


def test_alex_v1_ability_hands_xr_anchor_pose():
    result = run_simulation_app_function(
        _test_alex_v1_ability_hands_xr_anchor, headless=HEADLESS
    )
    assert result


def test_alex_v2_ability_hands_xr_anchor_pose():
    result = run_simulation_app_function(
        _test_alex_v2_ability_hands_xr_anchor, headless=HEADLESS
    )
    assert result


def test_alex_v1_pink_xr_anchor_pose():
    result = run_simulation_app_function(
        _test_alex_v1_pink_xr_anchor, headless=HEADLESS
    )
    assert result


def test_alex_v2_pink_xr_anchor_pose():
    result = run_simulation_app_function(
        _test_alex_v2_pink_xr_anchor, headless=HEADLESS
    )
    assert result
