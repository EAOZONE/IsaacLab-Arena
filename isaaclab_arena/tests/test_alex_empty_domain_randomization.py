# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import math

import pytest

from isaaclab_arena_environments.alex_empty_environment import (
    AlexEmptyEnvironment,
    _make_env_cfg_callback,
    _lever_safe_background_pool,
    _parse_light_color_palette,
)


class _DummyCfg:
    pass


def test_alex_empty_lever_dr_cli_defaults_to_auto():
    parser = argparse.ArgumentParser()
    AlexEmptyEnvironment.add_cli_args(parser)

    args = parser.parse_args([])

    assert args.teleop_hand_mode == "passthrough"
    assert args.teleop_hand_close_fraction == 0.75
    assert args.lever_dr is None
    assert args.lever_dr_xy_jitter == 0.05
    assert args.lever_dr_z_jitter == 0.04
    assert args.base_lever_pose_dr is False
    assert args.lever_dr_yaw_jitter_deg == 25.0
    assert args.robot_dr_xy_jitter == 0.04
    assert args.robot_dr_z_jitter is None
    assert args.robot_dr_yaw_jitter_deg == 8.0
    assert args.background_dr_pool == "packing_table"
    assert args.table == "none"


def test_lever_background_pool_filters_kitchen_counter_scenes():
    background_pool = _lever_safe_background_pool(
        ["kitchen", "ground_plane", "kitchen_with_open_drawer", "packing_table"]
    )

    assert background_pool == ["packing_table"]


def test_another_try_lever_is_registered_as_base_lever():
    pytest.importorskip("isaaclab")

    from isaaclab_arena_environments import lever_scene_builder

    assert "another_try_lever" in lever_scene_builder.LEVER_USD_STEMS
    assert "another_try_lever" in lever_scene_builder.LEVER_BASE_OBJECT_STEMS


def test_parse_light_color_palette_accepts_presets_and_rgb_literals():
    palette = _parse_light_color_palette("warm,0.1:0.2:0.3")

    assert palette[0] == (1.0, 0.86, 0.68)
    assert palette[1] == (0.1, 0.2, 0.3)


def test_alex_empty_env_cfg_callback_installs_dr_events():
    pytest.importorskip("isaaclab")

    env_cfg = _DummyCfg()
    env_cfg.events = _DummyCfg()
    env_cfg.terminations = _DummyCfg()
    env_cfg.sim = _DummyCfg()

    callback = _make_env_cfg_callback(
        control_hz=None,
        lever_success_object_name=None,
        lever_success_angle_deg=None,
        robot_dr=True,
        robot_position_xyz=(-0.4, -0.48682, 0.94296),
        robot_yaw_rad=0.0,
        robot_xy_jitter=0.04,
        robot_z_jitter=0.03,
        robot_yaw_jitter_rad=math.radians(8.0),
        background_dr_names=[
            "dr_background_00_kitchen",
            "dr_background_01_packing_table",
        ],
    )

    assert callback is not None
    env_cfg = callback(env_cfg)

    assert env_cfg.events.randomize_alex_root_pose.params["asset_cfg"].name == "robot"
    assert env_cfg.events.randomize_alex_root_pose.params["xy_jitter"] == 0.04
    assert env_cfg.events.randomize_alex_root_pose.params["z_jitter"] == 0.03
    assert env_cfg.events.randomize_background_visibility.params[
        "background_names"
    ] == [
        "dr_background_00_kitchen",
        "dr_background_01_packing_table",
    ]


def test_alex_empty_env_cfg_callback_installs_base_lever_pose_dr_event():
    pytest.importorskip("isaaclab")

    from isaaclab_arena.terms.events import randomize_base_lever_pose_and_reset_handle

    env_cfg = _DummyCfg()
    env_cfg.events = _DummyCfg()
    env_cfg.terminations = _DummyCfg()
    env_cfg.sim = _DummyCfg()
    base_lever_dr_params = {
        "object_name": "another_try_lever",
        "body_suffix": "/Handle_1",
        "base_position_xyz": (-0.05, -0.51, 0.75),
        "base_yaw_rad": math.radians(180.0),
        "xy_jitter": 0.06,
        "z_jitter": 0.04,
        "yaw_jitter_rad": math.radians(35.0),
        "object_scale": (0.0254, 0.0254, 0.0254),
        "body_local_pos": (0.0, 0.0, 0.0),
        "body_local_quat_xyzw": (0.0, 0.0, 0.0, 1.0),
    }

    callback = _make_env_cfg_callback(
        control_hz=None,
        lever_success_object_name=None,
        lever_success_angle_deg=None,
        robot_dr=False,
        robot_position_xyz=(-0.4, -0.48682, 0.94296),
        robot_yaw_rad=0.0,
        robot_xy_jitter=0.04,
        robot_z_jitter=0.0,
        robot_yaw_jitter_rad=math.radians(8.0),
        background_dr_names=[],
        base_lever_dr_params=base_lever_dr_params,
    )

    assert callback is not None
    env_cfg = callback(env_cfg)

    event = env_cfg.events.randomize_base_lever_pose
    assert event.func is randomize_base_lever_pose_and_reset_handle
    assert event.params["object_name"] == "another_try_lever"
    assert event.params["xy_jitter"] == 0.06
    assert event.params["z_jitter"] == 0.04
    assert event.params["yaw_jitter_rad"] == math.radians(35.0)


def test_alex_empty_env_cfg_callback_installs_base_lever_handle_reset_event():
    pytest.importorskip("isaaclab")

    from isaaclab_arena.terms.events import reset_internal_rigid_body_to_object_rest
    from isaaclab_arena.utils.pose import Pose

    env_cfg = _DummyCfg()
    env_cfg.events = _DummyCfg()
    env_cfg.terminations = _DummyCfg()
    env_cfg.sim = _DummyCfg()
    base_lever_reset_params = {
        "object_name": "another_try_lever",
        "body_suffix": "/Handle_1",
        "object_pose": Pose(position_xyz=(0.0, 0.0, 0.0)),
        "object_scale": (0.0254, 0.0254, 0.0254),
        "body_local_pos": (0.0, 0.0, 0.0),
        "body_local_quat_xyzw": (0.0, 0.0, 0.0, 1.0),
    }

    callback = _make_env_cfg_callback(
        control_hz=None,
        lever_success_object_name=None,
        lever_success_angle_deg=None,
        robot_dr=False,
        robot_position_xyz=(-0.4, -0.48682, 0.94296),
        robot_yaw_rad=0.0,
        robot_xy_jitter=0.04,
        robot_z_jitter=0.0,
        robot_yaw_jitter_rad=math.radians(8.0),
        background_dr_names=[],
        base_lever_reset_params=base_lever_reset_params,
    )

    assert callback is not None
    env_cfg = callback(env_cfg)

    event = env_cfg.events.reset_base_lever_handle
    assert event.func is reset_internal_rigid_body_to_object_rest
    assert event.params["object_name"] == "another_try_lever"
    assert event.params["body_suffix"] == "/Handle_1"


def test_background_visibility_supports_assets_without_env_ids():
    pytest.importorskip("isaaclab")

    from isaaclab_arena.terms.events import _set_scene_visibility

    class _XformVisibilityAsset:
        def __init__(self):
            self._prims = [object(), object()]
            self.calls = []

        def set_visibility(self, visible, indices=None):
            self.calls.append((visible, indices))

    asset = _XformVisibilityAsset()

    _set_scene_visibility(asset, True, env_ids=[2, 4])

    visibility, indices = asset.calls[0]
    assert visibility.tolist() == [True, True]
    assert indices is None


def test_background_visibility_preserves_env_ids_when_supported():
    pytest.importorskip("isaaclab")

    from isaaclab_arena.terms.events import _set_scene_visibility

    class _PerEnvVisibilityAsset:
        def __init__(self):
            self.calls = []

        def set_visibility(self, visible, env_ids=None):
            self.calls.append((visible, env_ids))

    asset = _PerEnvVisibilityAsset()

    _set_scene_visibility(asset, False, env_ids="env_ids")

    assert asset.calls == [(False, "env_ids")]
