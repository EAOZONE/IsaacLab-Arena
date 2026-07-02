# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Author revolute lever joints on the normalized lever practice-board USD."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

_LAYOUT_BASE_NAME = "layout_base"
_LEVER_LINKS_SCOPE = "lever_links"
_LEVER_JOINTS_SCOPE = "lever_joints"
_RESERVED_LAYOUT_CHILDREN = {_LAYOUT_BASE_NAME, _LEVER_LINKS_SCOPE, _LEVER_JOINTS_SCOPE}


@dataclass(frozen=True)
class RevoluteLeverSpec:
    """One rocker/twist lever on the practice board."""

    joint_name: str
    child_suffix: str
    axis: str
    lower_deg: float
    upper_deg: float
    mass_kg: float = 0.05
    stiffness: float = 1.5
    damping: float = 0.15
    max_force: float = 2.0


# Child paths are relative to ``Layout_v9``. Axes are in the layout-root frame (Z-up).
LEVER_REVOLUTE_SPECS: tuple[RevoluteLeverSpec, ...] = (
    RevoluteLeverSpec(
        joint_name="blue_handled_valve_lever",
        child_suffix=(
            "Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/base_link_1/base_link/"
            "Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1/"
            "Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1/Handle_1"
        ),
        axis="Y",
        lower_deg=-55.0,
        upper_deg=55.0,
        mass_kg=0.04,
    ),
    RevoluteLeverSpec(
        joint_name="red_handled_valve_stem",
        child_suffix="Red_Handled_Valve_v4_1/Red_Handled_Valve_v4/Valve_Stem_1",
        axis="Z",
        lower_deg=-90.0,
        upper_deg=90.0,
        mass_kg=0.03,
        stiffness=0.5,
        damping=0.2,
    ),
    RevoluteLeverSpec(
        joint_name="horizontal_donut_lever",
        child_suffix="Horizontal_Donut_v2_1/Horizontal_Donut_v2/Body1",
        axis="Y",
        lower_deg=-45.0,
        upper_deg=45.0,
        mass_kg=0.02,
    ),
    RevoluteLeverSpec(
        joint_name="vertical_donut_lever_1",
        child_suffix="Vertical_Donut_v2_1/Vertical_Donut_v2/Body1",
        axis="X",
        lower_deg=-45.0,
        upper_deg=45.0,
        mass_kg=0.02,
    ),
    RevoluteLeverSpec(
        joint_name="vertical_donut_lever_3",
        child_suffix="Vertical_Donut_v2_3/Vertical_Donut_v2/Body1",
        axis="X",
        lower_deg=-45.0,
        upper_deg=45.0,
        mass_kg=0.02,
    ),
    RevoluteLeverSpec(
        joint_name="vertical_donut_lever_10",
        child_suffix="Vertical_Donut_v2_10/Vertical_Donut_v2/Body1",
        axis="X",
        lower_deg=-45.0,
        upper_deg=45.0,
        mass_kg=0.02,
    ),
    RevoluteLeverSpec(
        joint_name="radiator_cap",
        child_suffix="Radiator_Cap_v5_1/Radiator_Cap_v5/Body6",
        axis="Z",
        lower_deg=-120.0,
        upper_deg=30.0,
        mass_kg=0.02,
        stiffness=0.3,
        damping=0.1,
    ),
    RevoluteLeverSpec(
        joint_name="dipstick_lever",
        child_suffix="Dipstick_Interface_v1_1/Dipstick_Interface_v1/Amazon_Dipstick_1",
        axis="Y",
        lower_deg=-50.0,
        upper_deg=50.0,
        mass_kg=0.015,
    ),
)


def _world_root_path(stage: Usd.Stage) -> str:
    root = stage.GetDefaultPrim()
    assert root.IsValid(), "Stage has no defaultPrim"
    return str(root.GetPath())


def _layout_root_path(stage: Usd.Stage) -> str:
    root = stage.GetDefaultPrim()
    assert root.IsValid(), "Stage has no defaultPrim"
    layout = root.GetChild("Layout_v9")
    assert layout.IsValid(), "Expected /World/Layout_v9 geometry root"
    return str(layout.GetPath())


def _update_joint_transform(stage: Usd.Stage, joint_path: str, body0_path: str, body1_path: str) -> None:
    joint_prim = stage.GetPrimAtPath(joint_path)
    body0 = stage.GetPrimAtPath(body0_path)
    body1 = stage.GetPrimAtPath(body1_path)
    assert joint_prim.IsValid(), joint_path
    assert body0.IsValid(), body0_path
    assert body1.IsValid(), body1_path

    xform_cache = UsdGeom.XformCache()
    body1_to_body0, _ = xform_cache.ComputeRelativeTransform(body1, body0)

    local_pos0_attr = joint_prim.GetAttribute("physics:localPos0")
    local_orient0_attr = joint_prim.GetAttribute("physics:localRot0")
    local_pos1_attr = joint_prim.GetAttribute("physics:localPos1")
    local_orient1_attr = joint_prim.GetAttribute("physics:localRot1")

    body0_to_joint = Gf.Matrix4d()
    body0_to_joint = Gf.Matrix4d.SetTransform(
        body0_to_joint,
        Gf.Rotation(Gf.Quatd(local_orient0_attr.Get())),
        Gf.Vec3d(local_pos0_attr.Get()),
    )
    relative_transform = body0_to_joint * body1_to_body0.GetInverse()
    local_pos1_attr.Set(relative_transform.ExtractTranslation())
    local_orient1_attr.Set(Gf.Quatf(relative_transform.ExtractRotationQuat()))


def _add_rigid_body(stage: Usd.Stage, prim_path: str, *, kinematic: bool) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    assert prim.IsValid(), prim_path
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_api.CreateKinematicEnabledAttr(kinematic)


def _add_mass(stage: Usd.Stage, prim_path: str, mass_kg: float) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass_kg)


def _set_local_matrix(prim: Usd.Prim, local_matrix: Gf.Matrix4d) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(local_matrix.ExtractTranslation()))
    xformable.AddOrientOp().Set(Gf.Quatf(local_matrix.ExtractRotationQuat()))


def _reparent_preserve_world(stage: Usd.Stage, source_path: str, dest_parent_path: str, dest_name: str) -> str:
    source = stage.GetPrimAtPath(source_path)
    dest_parent = stage.GetPrimAtPath(dest_parent_path)
    assert source.IsValid(), source_path
    assert dest_parent.IsValid(), dest_parent_path

    dest_path = f"{dest_parent_path}/{dest_name}"
    if str(source.GetPath()) == dest_path:
        return dest_path
    if stage.GetPrimAtPath(dest_path).IsValid():
        stage.RemovePrim(dest_path)

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_matrix = cache.GetLocalToWorldTransform(source)
    parent_world = cache.GetLocalToWorldTransform(dest_parent)
    local_matrix = world_matrix * parent_world.GetInverse()

    layer = stage.GetRootLayer()
    Sdf.CopySpec(layer, Sdf.Path(source_path), layer, Sdf.Path(dest_path))
    stage.RemovePrim(source_path)
    _set_local_matrix(stage.GetPrimAtPath(dest_path), local_matrix)
    return dest_path


def _ensure_layout_base(stage: Usd.Stage, layout_path: str) -> str:
    world_path = _world_root_path(stage)
    world_prim = stage.GetPrimAtPath(world_path)
    layout_prim = stage.GetPrimAtPath(layout_path)
    base_path = f"{world_path}/{_LAYOUT_BASE_NAME}"
    if not stage.GetPrimAtPath(base_path).IsValid():
        stage.DefinePrim(base_path, "Xform")

    if layout_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        layout_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    base_prim = stage.GetPrimAtPath(base_path)
    if base_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        base_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    if not world_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(world_prim)
        world_prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)

    if base_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rb_api = UsdPhysics.RigidBodyAPI(base_prim)
        rb_api.CreateKinematicEnabledAttr(False)

    _add_rigid_body(stage, base_path, kinematic=False)
    _add_mass(stage, base_path, 5.0)

    for child in list(layout_prim.GetChildren()):
        if child.GetName() in _RESERVED_LAYOUT_CHILDREN:
            continue
        _reparent_preserve_world(stage, str(child.GetPath()), base_path, child.GetName())

    return base_path


def _reparent_lever_link(stage: Usd.Stage, *, layout_path: str, links_scope_path: str, spec: RevoluteLeverSpec) -> str:
    dest_path = f"{links_scope_path}/{spec.joint_name}"
    if stage.GetPrimAtPath(dest_path).IsValid():
        return dest_path

    legacy_dest = f"{layout_path}/{_LEVER_LINKS_SCOPE}/{spec.joint_name}"
    if stage.GetPrimAtPath(legacy_dest).IsValid():
        return _reparent_preserve_world(stage, legacy_dest, links_scope_path, spec.joint_name)

    source_path = f"{layout_path}/{spec.child_suffix}"
    if not stage.GetPrimAtPath(source_path).IsValid():
        source_path = f"{layout_path}/{_LAYOUT_BASE_NAME}/{spec.child_suffix}"
    source = stage.GetPrimAtPath(source_path)
    assert source.IsValid(), f"Lever child prim not found for {spec.joint_name}: tried {spec.child_suffix}"
    return _reparent_preserve_world(stage, source_path, links_scope_path, spec.joint_name)


def _child_anchor_in_parent(stage: Usd.Stage, parent_path: str, child_path: str) -> Gf.Vec3f:
    xform_cache = UsdGeom.XformCache()
    rel, _ = xform_cache.ComputeRelativeTransform(stage.GetPrimAtPath(child_path), stage.GetPrimAtPath(parent_path))
    return Gf.Vec3f(rel.ExtractTranslation())


def _apply_revolute_joint(
    stage: Usd.Stage,
    *,
    base_path: str,
    child_path: str,
    joint_parent_path: str,
    spec: RevoluteLeverSpec,
) -> None:
    child_prim = stage.GetPrimAtPath(child_path)
    assert child_prim.IsValid(), f"Lever child prim not found: {child_path}"

    joint_path = f"{joint_parent_path}/{_LEVER_JOINTS_SCOPE}/{spec.joint_name}"
    if stage.GetPrimAtPath(joint_path).IsValid():
        stage.RemovePrim(joint_path)

    _add_rigid_body(stage, child_path, kinematic=False)
    _add_mass(stage, child_path, spec.mass_kg)

    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([base_path])
    joint.CreateBody1Rel().SetTargets([child_path])
    joint.GetAxisAttr().Set(spec.axis)
    joint.GetLowerLimitAttr().Set(math.radians(spec.lower_deg))
    joint.GetUpperLimitAttr().Set(math.radians(spec.upper_deg))
    joint.CreateLocalPos0Attr().Set(_child_anchor_in_parent(stage, base_path, child_path))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    _update_joint_transform(stage, joint_path, base_path, child_path)

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateStiffnessAttr(spec.stiffness)
    drive.CreateDampingAttr(spec.damping)
    drive.CreateMaxForceAttr(spec.max_force)


def apply_articulation(stage: Usd.Stage, specs: Iterable[RevoluteLeverSpec] = LEVER_REVOLUTE_SPECS) -> int:
    """Bake a fixed-base articulation with revolute levers into ``stage`` (idempotent)."""
    world_path = _world_root_path(stage)
    layout_path = _layout_root_path(stage)

    links_scope_path = f"{world_path}/{_LEVER_LINKS_SCOPE}"
    joints_scope_path = f"{world_path}/{_LEVER_JOINTS_SCOPE}"
    if not stage.GetPrimAtPath(links_scope_path).IsValid():
        stage.DefinePrim(links_scope_path, "Scope")
    if not stage.GetPrimAtPath(joints_scope_path).IsValid():
        stage.DefinePrim(joints_scope_path, "Scope")

    specs_list = list(specs)
    for spec in specs_list:
        _reparent_lever_link(stage, layout_path=layout_path, links_scope_path=links_scope_path, spec=spec)

    base_path = _ensure_layout_base(stage, layout_path)

    for spec in specs_list:
        child_path = f"{links_scope_path}/{spec.joint_name}"
        _apply_revolute_joint(
            stage,
            base_path=base_path,
            child_path=child_path,
            joint_parent_path=world_path,
            spec=spec,
        )

    legacy_links = stage.GetPrimAtPath(f"{layout_path}/{_LEVER_LINKS_SCOPE}")
    if legacy_links.IsValid() and len(list(legacy_links.GetChildren())) == 0:
        stage.RemovePrim(str(legacy_links.GetPath()))
    legacy_joints = stage.GetPrimAtPath(f"{layout_path}/{_LEVER_JOINTS_SCOPE}")
    if legacy_joints.IsValid() and len(list(legacy_joints.GetChildren())) == 0:
        stage.RemovePrim(str(legacy_joints.GetPath()))

    return len(specs_list)
