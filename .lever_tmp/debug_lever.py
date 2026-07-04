"""Debug why the lever moves at rest: gravity, early joint states, contacts."""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

PATH = "/workspaces/isaaclab_arena/isaaclab_arena/assets/lever_sim/Lever.usd"

omni.usd.get_context().open_stage(PATH)
stage = omni.usd.get_context().get_stage()
with Usd.EditContext(stage, stage.GetSessionLayer()):
    UsdPhysics.Scene.Define(stage, "/physicsScene")

from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import SingleArticulation

sim = SimulationContext(
    stage_units_in_meters=UsdGeom.GetStageMetersPerUnit(stage),
    physics_dt=1.0 / 120.0,
    rendering_dt=1.0 / 120.0,
)
sim.initialize_physics()

# report gravity
scene = UsdPhysics.Scene(stage.GetPrimAtPath("/physicsScene"))
print("gravity dir:", scene.GetGravityDirectionAttr().Get(), "mag:", scene.GetGravityMagnitudeAttr().Get())
try:
    physics_ctx = sim.get_physics_context()
    print("physics ctx gravity:", physics_ctx.get_gravity())
except Exception as exc:
    print("no physics ctx gravity:", exc)

sim.play()
art = SingleArticulation("/World")
art.initialize()

# check the authored joint friction is visible
jp = stage.GetPrimAtPath("/World/PhysicsBase/lever_joint")
print("joint applied schemas:", list(jp.GetAppliedSchemas()))
print("jointFriction:", jp.GetAttribute("physxJoint:jointFriction").Get())

for i in range(20):
    sim.step(render=False)
    q = float(art.get_joint_positions()[0])
    v = float(art.get_joint_velocities()[0])
    print(f"step {i:3d}: q={np.degrees(q):8.3f} deg  v={np.degrees(v):9.3f} deg/s")

for i in range(40, 241, 40):
    for _ in range(40):
        sim.step(render=False)
    q = float(art.get_joint_positions()[0])
    v = float(art.get_joint_velocities()[0])
    print(f"step {i + 20:3d}: q={np.degrees(q):8.3f} deg  v={np.degrees(v):9.3f} deg/s")

simulation_app.close()
