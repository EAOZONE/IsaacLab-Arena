"""Soften LEVER_AGAIN.usd's RevoluteJoint drive so a hand push can actually move it.

Diagnostic (inspect_lever_again_bbox.py) found stiffness=40000, damping=40000 holding
the handle at targetPosition=90 within limits [85, 180]. That's an enormous restoring
torque (~37700 N*m to reach the 54deg-from-rest success threshold) -- no plausible
hand-contact force will move it. Soften to something in the range that worked for the
older Lever_revolute.usd asset (stiffness=2000, damping=200, per project memory).
"""
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--viz", "none"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
from pxr import UsdPhysics

usd_path = "/workspaces/isaaclab_arena/isaaclab_arena/assets/lever_sim/LEVER_AGAIN.usd"
ctx = omni.usd.get_context()
ctx.open_stage(usd_path)
stage = ctx.get_stage()

joints = [p for p in stage.Traverse() if p.GetTypeName() == "PhysicsRevoluteJoint"]
assert len(joints) == 1, f"expected one RevoluteJoint, found {[str(p.GetPath()) for p in joints]}"
joint_prim = joints[0]
drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
print("Before: stiffness=", drive.GetStiffnessAttr().Get(), "damping=", drive.GetDampingAttr().Get())

drive.GetStiffnessAttr().Set(2000.0)
drive.GetDampingAttr().Set(200.0)

print("After: stiffness=", drive.GetStiffnessAttr().Get(), "damping=", drive.GetDampingAttr().Get())

stage.Save()
print("Saved.")

simulation_app.close()
