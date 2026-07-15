# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared scene-building for the lever_sim board asset (``Lever.usd`` / ``Lever_revolute.usd``).

Used by both ``alex_empty`` (manual placement / scripted-demo recording) and ``alex_lever_turn``
(RL training) so the tuned pose, the joint-frame-consistent orientation convention, and the
domain-randomization ranges stay single-sourced instead of drifting between call sites.
"""

from __future__ import annotations

import math
from pathlib import Path

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.pose import Pose, PoseRange

# Tuned lever-board pose (see Pictures/Screenshots 2026-07-04). Lever_revolute.usd (2026-07-07)
# shares the same Layout_v9 origin and inches units (mpu=0.0254) as Lever.usd, just with its
# physics authored as a single dynamic rigid body (the handle) jointed straight to the static
# base instead of the fragile ArticulationRootAPI + dummy-link workaround in Lever_physics.usd,
# so the same tuned pose/scale applies.
#
# 2026-07-08: the baked rotateX(-90) on base_link was removed from Lever_revolute.usd so the
# asset matches usdview without an extra hidden offset. The asset's stage-declared upAxis is Y,
# but its authored geometry is already Z-up (confirmed by opening Lever_revolute.usd directly as
# a root stage: the pegboard sits flat with the valve pointing +Z, no compensating rotation
# needed). Spawn therefore uses roll=0 with only usd_yaw about world Z -- a nonzero roll here
# re-tips an already-flat board. The handle's own rest-facing direction is controlled by the
# RevoluteJoint's local frames inside the asset, not by any board-level spawn rotation -- see
# Handle_1's RevoluteJoint localRot0/localRot1 in Lever_revolute.usd.
LEVER_USD_STEMS = (
    "lever",
    "lever_revolute",
    "new_lever",
    "lever_again",
    "another_lever",
    "another_try_lever",
)
LEVER_USD_DEFAULT_POS = (-0.05062, -0.51385, 0.75167)
LEVER_USD_DEFAULT_YAW = 180.0
LEVER_USD_DEFAULT_SCALE = 0.0254
LEVER_AGAIN_STEM = "lever_again"
LEVER_ARTICULATION_STEMS = ("lever",)
LEVER_BASE_OBJECT_STEMS = ("lever_again", "another_lever", "another_try_lever")

# lever_dr (opt-in) reset-time pose jitter, on top of usd_yaw's nominal yaw.
_LEVER_DR_XY_JITTER = 0.02  # +/- meters, x and y independently
_LEVER_DR_YAW_RANGE_DEG = 15.0  # +/- degrees about the nominal yaw

# lever_dr curated handle-color palette (r, g, b in 0-1), one sampled per reset.
_LEVER_DR_COLOR_PALETTE = (
    (0.55, 0.55, 0.58),  # metal grey (close to the native Steel___Satin material)
    (0.03, 0.03, 0.03),  # black
    (0.55, 0.05, 0.05),  # red
    (0.05, 0.08, 0.55),  # blue
)
# Prim-path suffix (relative to the lever Object's own prim_path) of the visible steel handle
# mesh, found via manual USD traversal of Lever_revolute.usd (defaultPrim /World, so its children
# compose directly onto the Object's own prim_path with no extra "World" segment). The only rigid
# body in the file is Handle_1; its Body1/Body1 child mesh is the larger steel body (bound to
# /World/Looks/Steel___Satin) -- Handle_1/Handle/Body2/Body2 is a separate small plastic part
# (ABS__White_3, also the collision mesh) left untouched for this first cut.
_LEVER_HANDLE_MESH_NAME = (
    "/Layout_v9/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/base_link_1/base_link/"
    "Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1/"
    "Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1/Handle_1/Handle/Body1/Body1"
)
LEVER_HANDLE_RIGID_BODY_SUFFIX = (
    "/Layout_v9/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/base_link_1/base_link/"
    "Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1/Handle_1"
)

# Workbench placed under the lever board (visual sim2real: the real lever_eef dataset was
# recorded with the fixture bolted to a wooden bench, not floating over a bare grid floor).
# SeattleLabTable's own prim origin sits ~(0.37, 0.16) away from its mesh center in its local
# xy (measured via UsdGeom.BBoxCache), so it's placed at the lever xy minus that offset to
# actually center the tabletop under the lever. z is tuned so its surface meets the lever base.
_LEVER_TABLE_XY_OFFSET = (0.37025, 0.15521)
_LEVER_TABLE_POS_Z = 0.0


def _handle_rest_pose_in_asset(
    usd_path: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    assert stage is not None, f"Could not open lever USD: {usd_path}"
    handle_prim = stage.GetPrimAtPath("/World" + LEVER_HANDLE_RIGID_BODY_SUFFIX)
    assert (
        handle_prim.IsValid()
    ), f"Lever handle prim not found in {usd_path}: /World{LEVER_HANDLE_RIGID_BODY_SUFFIX}"
    transform = UsdGeom.Xformable(handle_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    pos = transform.ExtractTranslation()
    quat = transform.ExtractRotationQuat()
    quat_imag = quat.GetImaginary()
    return (pos[0], pos[1], pos[2]), (
        quat_imag[0],
        quat_imag[1],
        quat_imag[2],
        quat.GetReal(),
    )


def build_lever_scene_assets(
    usd_path: str,
    usd_pos: tuple[float, float, float],
    usd_yaw: float,
    usd_scale: float,
    lever_dr: bool,
    table: str,
    lever_dr_xy_jitter: float = _LEVER_DR_XY_JITTER,
    lever_dr_yaw_jitter_deg: float = _LEVER_DR_YAW_RANGE_DEG,
) -> tuple[list[Asset], Object]:
    """Build the lever (+ optional table) scene assets for a lever_sim board USD.

    Args:
        usd_path: Path to the lever_sim board USD (``Lever.usd`` / ``Lever_revolute.usd``).
        usd_pos: World position x,y,z for the board (already resolved by the caller -- pass
            ``LEVER_USD_DEFAULT_POS`` for the tuned default).
        usd_yaw: Yaw in degrees about world Z for the lever board (see the rotation comment below).
        usd_scale: Uniform scale (pass ``LEVER_USD_DEFAULT_SCALE`` for the tuned default).
        lever_dr: Enable reset-time xy/yaw pose jitter plus curated handle-color variation.
        table: Workbench asset key placed under the board (``"seattle_lab"`` or ``"none"``).
        lever_dr_xy_jitter: Half-range for xy reset-time jitter, in metres.
        lever_dr_yaw_jitter_deg: Half-range for reset-time yaw jitter, in degrees.

    Returns:
        A ``(extra_scene_assets, lever_object)`` tuple. ``extra_scene_assets`` contains the
        lever object itself (with DR variation attached/enabled if requested) and, if enabled,
        the table. ``lever_object`` is also returned standalone since tasks need it directly for
        reward/observation ``SceneEntityCfg``s.
    """
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.utils.math import quat_from_euler_xyz

    # The asset's own geometry is already Z-up (flat board, handle pointing +Z) -- only usd_yaw
    # (about world Z) is needed; no roll/pitch tilt.
    lever_yaw_rad = math.radians(usd_yaw)
    lever_rotation_xyzw = tuple(
        quat_from_euler_xyz(
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            torch.tensor([lever_yaw_rad]),
        )[0].tolist()
    )
    usd_stem = Path(usd_path).stem.lower()
    use_pose_range_reset = lever_dr and usd_stem not in LEVER_BASE_OBJECT_STEMS
    if use_pose_range_reset:
        half_yaw_jitter_rad = math.radians(lever_dr_yaw_jitter_deg)
        usd_initial_pose = PoseRange(
            position_xyz_min=(
                usd_pos[0] - lever_dr_xy_jitter,
                usd_pos[1] - lever_dr_xy_jitter,
                usd_pos[2],
            ),
            position_xyz_max=(
                usd_pos[0] + lever_dr_xy_jitter,
                usd_pos[1] + lever_dr_xy_jitter,
                usd_pos[2],
            ),
            rpy_min=(0.0, 0.0, lever_yaw_rad - half_yaw_jitter_rad),
            rpy_max=(0.0, 0.0, lever_yaw_rad + half_yaw_jitter_rad),
        )
    else:
        usd_initial_pose = Pose(position_xyz=usd_pos, rotation_xyzw=lever_rotation_xyzw)

    lever_object = Object(
        name=usd_stem.replace("(", "_").replace(")", "_"),
        usd_path=usd_path,
        initial_pose=usd_initial_pose,
        object_type=(
            ObjectType.ARTICULATION
            if usd_stem in LEVER_ARTICULATION_STEMS
            else ObjectType.BASE if usd_stem in LEVER_BASE_OBJECT_STEMS else None
        ),
        scale=(usd_scale, usd_scale, usd_scale),
    )
    if usd_stem in LEVER_ARTICULATION_STEMS:
        lever_object.object_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=100.0,
            max_linear_velocity=1.0,
            max_angular_velocity=5.0,
            max_depenetration_velocity=0.5,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=16,
        )
        lever_object.object_cfg.spawn.articulation_props = (
            sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=16,
                sleep_threshold=5.0e-5,
                stabilization_threshold=1.0e-5,
                fix_root_link=True,
            )
        )
        lever_object.object_cfg.init_state.joint_pos = {
            "RevoluteJoint": math.radians(90.0)
        }
        lever_object.object_cfg.init_state.joint_vel = {"RevoluteJoint": 0.0}
    elif usd_stem in LEVER_BASE_OBJECT_STEMS and lever_dr:
        # ``another_lever`` / ``lever_again`` are base USDs with a nested PhysX
        # rigid handle. Moving the root Xform or nested RigidPrim from a reset
        # event after GPU PhysX/Fabric starts can trip CUDA illegal-address
        # failures. Keep these base-object levers physically fixed by default;
        # the teleop dataset still gets safer visual, lighting, background, and
        # robot-root variation.
        pass
    scene_assets: list[Asset] = [lever_object]

    if lever_dr:
        from isaaclab_arena.variations.visual_color_variation import (
            VisualColorVariation,
            VisualColorVariationCfg,
        )

        lever_object.add_variation(
            VisualColorVariation(
                lever_object.name,
                cfg=VisualColorVariationCfg(
                    palette=list(_LEVER_DR_COLOR_PALETTE),
                    mesh_name=_LEVER_HANDLE_MESH_NAME,
                ),
            )
        )
        lever_object.get_variation(f"{lever_object.name}_color_variation").enable()

    if table != "none":
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

        scene_assets.append(
            Object(
                name="lever_table",
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
                initial_pose=Pose(
                    position_xyz=(
                        usd_pos[0] - _LEVER_TABLE_XY_OFFSET[0],
                        usd_pos[1] - _LEVER_TABLE_XY_OFFSET[1],
                        _LEVER_TABLE_POS_Z,
                    ),
                    rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
                ),
            )
        )

    return scene_assets, lever_object
