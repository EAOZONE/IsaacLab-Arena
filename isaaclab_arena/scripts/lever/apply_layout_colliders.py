# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Normalize the lever practice-board USD, bake static colliders, and author lever joints.

``assets/Lever/Levers.usd`` is a raw CAD export: Y-up, inch units
(``metersPerUnit = 0.0254``), repeated parts instanced, and no
``UsdPhysics.CollisionAPI`` on any mesh. Referenced as-is into an Arena stage
(Z-up, meters) it would spawn 39x too large, lying on its side, and with no
collision. This script patches the file in place (idempotent):

1. Bakes a Y-up -> Z-up rotation and inch -> meter scale onto the ``/World``
   root prim and rewrites the stage metadata to Z-up / ``metersPerUnit = 1``.
2. Disables scenegraph instancing so per-mesh physics schemas can be authored.
3. Applies ``CollisionAPI`` + ``convexDecomposition`` approximation to every
   mesh. (Isaac Lab's spawn-time ``collision_props`` cannot do this —
   ``modify_collision_properties`` only touches prims that already carry the
   schema.)
4. Authors a fixed-base articulation on ``Layout_v9`` with revolute joints for
   each practice lever (see ``layout_physics.py``).

Run inside the Arena container (plain ``pxr``, no SimulationApp needed)::

    /isaac-sim/python.sh isaaclab_arena/scripts/lever/apply_layout_colliders.py

The patched USD is consumed by the ``lever_layout`` asset
(see ``isaaclab_arena/assets/object_library.py``).
"""

from __future__ import annotations

import argparse
import math
import os

DEFAULT_LAYOUT_USD = os.path.join(
    os.path.dirname(__file__), "..", "..", "assets", "Lever", "Levers.usd"
)

_INCH_METERS_PER_UNIT = 0.0254


def _normalize_units_and_axis(stage) -> bool:
    """Convert the Y-up / inch-unit export to Z-up / meters, in place.

    The correction must live on the geometry prim *below* the defaultPrim
    (``Layout_v9``): Isaac Lab's USD spawner authors its own translate/orient/scale
    ops on the referencing prim (which composes with the defaultPrim), so any
    correction baked on the defaultPrim itself is silently discarded at spawn.

    The exporter authored ``rotateX(-90)`` on ``Layout_v9`` to present the natively
    Z-up CAD model in a Y-up stage; the Y-up -> Z-up correction (rotateX(+90))
    cancels it exactly, so the normalized prim carries only the inch -> meter scale.

    Returns True if a correction was applied, False if already normalized.
    """
    from pxr import Gf, UsdGeom

    root = stage.GetDefaultPrim()
    assert root.IsValid(), "Stage has no defaultPrim"
    geo_prim = root.GetChild("Layout_v9")
    assert geo_prim.IsValid() and geo_prim.IsA(UsdGeom.Xformable), (
        "Expected Layout_v9 geometry root under defaultPrim"
    )
    geo = UsdGeom.Xformable(geo_prim)

    ops = {op.GetOpName(): op for op in geo.GetOrderedXformOps()}
    scale_op = ops.get("xformOp:scale")
    if scale_op is not None and Gf.IsClose(scale_op.Get(), Gf.Vec3f(_INCH_METERS_PER_UNIT), 1e-6):
        return False

    up_axis = UsdGeom.GetStageUpAxis(stage)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    assert up_axis == UsdGeom.Tokens.y and math.isclose(meters_per_unit, _INCH_METERS_PER_UNIT), (
        f"Unexpected stage conventions (upAxis={up_axis}, metersPerUnit={meters_per_unit}); "
        "this script only handles the raw Y-up inch export or an already-normalized file."
    )
    translate = ops["xformOp:translate"].Get()
    rotate = ops["xformOp:rotateXYZ"].Get()
    assert Gf.IsClose(translate, Gf.Vec3d(0.0, 0.0, 0.0), 1e-6), f"Unexpected translate {translate} on {geo.GetPath()}"
    assert Gf.IsClose(rotate, Gf.Vec3f(-90.0, 0.0, 0.0), 1e-6), f"Unexpected rotation {rotate} on {geo.GetPath()}"

    ops["xformOp:rotateXYZ"].Set(Gf.Vec3f(0.0, 0.0, 0.0))
    ops["xformOp:scale"].Set(Gf.Vec3f(_INCH_METERS_PER_UNIT))

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return True


def _disable_instancing(stage) -> int:
    """Disable scenegraph instancing so physics schemas can be authored per mesh."""
    num_deinstanced = 0
    # Instances can nest; re-traverse until none remain.
    while True:
        instances = [prim for prim in stage.Traverse() if prim.IsInstance()]
        if not instances:
            return num_deinstanced
        for prim in instances:
            prim.SetInstanceable(False)
            num_deinstanced += 1


def apply_colliders(stage) -> int:
    """Apply convex-decomposition colliders to all meshes; return the mesh count."""
    from pxr import UsdGeom, UsdPhysics

    num_meshes = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr().Set("convexDecomposition")
        num_meshes += 1

    assert num_meshes > 0, "No meshes found"
    return num_meshes


def process(usd_path: str) -> None:
    from pxr import Usd

    from isaaclab_arena.scripts.lever.layout_physics import apply_articulation

    stage = Usd.Stage.Open(usd_path)
    assert stage is not None, f"Could not open {usd_path}"

    normalized = _normalize_units_and_axis(stage)
    num_deinstanced = _disable_instancing(stage)
    num_meshes = apply_colliders(stage)
    num_joints = apply_articulation(stage)
    stage.GetRootLayer().Save()

    print(
        f"{usd_path}: normalized={normalized}, de-instanced {num_deinstanced} prims, "
        f"applied convexDecomposition colliders to {num_meshes} meshes, "
        f"authored {num_joints} revolute lever joints"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layout_usd",
        type=str,
        default=os.path.normpath(DEFAULT_LAYOUT_USD),
        help="Path to the lever board USD to patch in place.",
    )
    args = parser.parse_args()
    process(args.layout_usd)


if __name__ == "__main__":
    main()
