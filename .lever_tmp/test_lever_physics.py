"""Headless verification that Lever.usd's blue lever physics works.

Consumes the asset the same way Arena does: referenced into a meters, Z-up
stage with a position + yaw offset (what UsdFileCfg produces). The asset is
natively in meters, so spawn scale is 1.0.

Checks:
1. The referenced asset parses as a 1-DOF articulation (lever_joint).
2. At rest the lever holds its authored pose (joint friction beats gravity).
3. Driving the joint rotates the nut about its own local Y axis (no translation).
4. Joint limits (+/- 90 deg) are respected.

Run inside the container:
    /isaac-sim/python.sh .lever_tmp/test_lever_physics.py
"""

import os

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np

SCALE = float(os.environ.get("LEVER_TEST_SCALE", "1.0"))
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ASSET = "/workspaces/isaaclab_arena/isaaclab_arena/assets/lever_sim/Lever.usd"
LEVER = "/World/lever"
BASE_LINK = (
    LEVER + "/Layout_v9/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/base_link_1/base_link"
)
NUT = BASE_LINK + "/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1"

# fresh meters / Z-up stage, asset referenced in like Arena's UsdFileCfg does
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.Scene.Define(stage, "/physicsScene")
lever = UsdGeom.Xform.Define(stage, Sdf.Path(LEVER))
lever.GetPrim().GetReferences().AddReference(ASSET)
# the referenced /World brings identity translate/rotateXYZ/scale ops along;
# overwrite them with the spawn pose (what Arena's spawner effectively does)
ops = {op.GetOpName(): op for op in lever.GetOrderedXformOps()}
ops["xformOp:translate"].Set(Gf.Vec3d(0.6, 0.0, 0.9))
ops["xformOp:rotateXYZ"].Set(Gf.Vec3f(0.0, 0.0, 90.0))
ops["xformOp:scale"].Set(Gf.Vec3f(SCALE))
print("spawn scale:", SCALE, flush=True)

from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import SingleArticulation

sim = SimulationContext(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 120.0)
sim.initialize_physics()
sim.play()

art = SingleArticulation(LEVER)
art.initialize()
print("DOF names:", art.dof_names, flush=True)
assert art.num_dof == 1, f"expected 1 DOF, got {art.num_dof}"

cache = UsdGeom.XformCache(Usd.TimeCode.Default())


def rel_nut_in_base():
    cache.Clear()
    base = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE_LINK))
    nut = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(NUT))
    return nut * base.GetInverse()


rel0 = rel_nut_in_base()
t0 = rel0.ExtractTranslation()
q0 = rel0.ExtractRotationQuat().GetNormalized()

# --- 1. rest stability: 240 steps (2 s), lever must hold its pose ---
for _ in range(240):
    sim.step(render=False)
pos_rest = float(art.get_joint_positions()[0])
rel_rest = rel_nut_in_base()
drift_t = (rel_rest.ExtractTranslation() - t0).GetLength()
print(f"rest: joint={np.degrees(pos_rest):.3f} deg, translation drift={drift_t:.6f}", flush=True)
assert abs(pos_rest) < np.radians(3.0), f"lever moved at rest: {np.degrees(pos_rest):.2f} deg"
assert drift_t < 0.05, f"nut translated at rest: {drift_t}"

# --- 2. drive the joint with sustained torque (like a robot push):
#        nut must rotate about its local Y ---
for _ in range(120):
    art.set_joint_efforts(np.array([1.0]))
    sim.step(render=False)
pos_driven = float(art.get_joint_positions()[0])
rel1 = rel_nut_in_base()
q1 = rel1.ExtractRotationQuat().GetNormalized()
delta = q0.GetInverse() * q1  # rotation in the nut's own initial frame
img = delta.GetImaginary()
angle = 2.0 * np.arctan2(img.GetLength(), delta.GetReal())
if angle > np.pi:  # quaternion double cover: take the short way around
    angle = 2.0 * np.pi - angle
axis = Gf.Vec3d(img).GetNormalized() if img.GetLength() > 1e-9 else Gf.Vec3d(0)
axis_y = abs(Gf.Dot(axis, Gf.Vec3d(0, 1, 0)))
drift_t1 = (rel1.ExtractTranslation() - t0).GetLength()
print(
    f"driven: joint={np.degrees(pos_driven):.2f} deg, rotated {np.degrees(angle):.2f} deg,"
    f" |axis . localY|={axis_y:.4f}, translation drift={drift_t1:.6f}",
    flush=True,
)
assert abs(pos_driven) > np.radians(20.0), f"joint barely moved: {np.degrees(pos_driven):.2f} deg"
assert axis_y > 0.99, f"rotation not about nut local Y (|dot|={axis_y:.3f})"
assert drift_t1 < 0.05, f"nut translated while rotating: {drift_t1}"

# --- 3. joint limit: keep driving, must clamp at +90 deg ---
for _ in range(360):
    art.set_joint_efforts(np.array([1.0]))
    sim.step(render=False)
pos_limit = float(art.get_joint_positions()[0])
print(f"limit: joint={np.degrees(pos_limit):.2f} deg (limits +/- 90)", flush=True)
assert abs(pos_limit) < np.radians(92.0), f"limit violated: {np.degrees(pos_limit):.2f} deg"
assert abs(pos_limit) > np.radians(80.0), f"never reached the limit: {np.degrees(pos_limit):.2f} deg"

# --- 4. release at the limit: friction must hold it there (no swing back) ---
for _ in range(240):
    sim.step(render=False)
pos_hold = float(art.get_joint_positions()[0])
print(f"hold after release: joint={np.degrees(pos_hold):.2f} deg", flush=True)
assert abs(pos_hold - pos_limit) < np.radians(10.0), (
    f"lever swung back after release: {np.degrees(pos_hold):.2f} deg"
)

print("ALL LEVER PHYSICS CHECKS PASSED", flush=True)
simulation_app.close()
