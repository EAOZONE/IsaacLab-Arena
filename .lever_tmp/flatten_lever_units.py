"""Convert Lever.usd from inches (metersPerUnit=0.0254) to meters (mpu=1).

Bakes the unit conversion into the data — mesh points, extents, and every
xformOp:translate — so the asset needs no spawn scale. Scaled articulated
references are numerically fragile in PhysX (non-finite broadphase bounds),
so the asset must be natively in meters.

Run on the PRISTINE (physics-free) asset, before add_lever_physics.py:
    /isaac-sim/python.sh .lever_tmp/flatten_lever_units.py
"""

from pxr import Gf, Usd, UsdGeom, Vt

PATH = "/workspaces/isaaclab_arena/isaaclab_arena/assets/lever_sim/Lever.usd"

stage = Usd.Stage.Open(PATH)
mpu = UsdGeom.GetStageMetersPerUnit(stage)
assert abs(mpu - 1.0) > 1e-9, "stage is already in meters"
s = mpu
print(f"scaling by metersPerUnit={s}")

n_meshes, n_translates = 0, 0
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if pts:
            mesh.GetPointsAttr().Set(Vt.Vec3fArray([p * s for p in pts]))
        ext = mesh.GetExtentAttr().Get()
        if ext:
            mesh.GetExtentAttr().Set(Vt.Vec3fArray([e * s for e in ext]))
        n_meshes += 1
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        continue
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            val = op.Get()
            if val is not None:
                op.Set(Gf.Vec3d(val) * s)
                n_translates += 1

UsdGeom.SetStageMetersPerUnit(stage, 1.0)
stage.Save()
print(f"scaled {n_meshes} meshes, {n_translates} translate ops; stage now in meters")
