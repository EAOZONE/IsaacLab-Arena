"""Find spawn rotation that reproduces the old sim pose after USD -90deg X removal."""

import math

import torch
from isaaclab.utils.math import quat_from_euler_xyz
from pxr import Gf, Usd, UsdGeom

path = "/workspaces/isaaclab_arena/isaaclab_arena/assets/lever_sim/Lever_revolute.usd"
stage = Usd.Stage.Open(path)

BASE = (
    "/World/Layout_v9/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/base_link_1/base_link"
)
HANDLE = (
    BASE
    + "/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1/"
    "Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1/Handle_1"
)
cache = UsdGeom.XformCache()
base_xf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(BASE))
handle_xf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(HANDLE))
layout_xf = cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/World/Layout_v9"))


def rot_to_euler(rot: Gf.Rotation) -> tuple[float, float, float]:
    return rot.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))


def apply_spawn(roll_deg: float, pitch_deg: float, yaw_deg: float, xf: Gf.Matrix4d) -> Gf.Rotation:
    q = quat_from_euler_xyz(
        torch.tensor([math.radians(roll_deg)]),
        torch.tensor([math.radians(pitch_deg)]),
        torch.tensor([math.radians(yaw_deg)]),
    )[0].tolist()
    spawn = Gf.Rotation(Gf.Quatd(q[3], Gf.Vec3d(q[0], q[1], q[2])))
    return spawn * xf.ExtractRotation()


# Old desired sim pose (roll=90, yaw=180): what base/handle looked like with OLD usd (-90 on base)
# Old native base had -90 X; old spawn roll=90 => net base identity in a certain frame
# Target: reproduce old "after sim spawn" handle orientation using NEW usd (identity base)

# Old sim result for handle (from previous inspect run with old USD):
OLD_TARGET_HANDLE_EULER = (-90.0, -1.6, -88.2)  # approximate from previous session with old usd+spawn

print("NEW USD native euler (deg):")
print("  base:", tuple(round(v, 1) for v in rot_to_euler(base_xf.ExtractRotation())))
print("  handle:", tuple(round(v, 1) for v in rot_to_euler(handle_xf.ExtractRotation())))
print("  layout +Z:", tuple(round(v, 1) for v in rot_to_euler(layout_xf.ExtractRotation())))

# Brute force: try roll=0 with old yaw, and yaw-only variants
for roll in [0.0, 90.0, -90.0]:
    for pitch in [0.0, 90.0, -90.0]:
        for yaw in [0.0, 90.0, 180.0, 270.0]:
            h = apply_spawn(roll, pitch, yaw, handle_xf)
            e = rot_to_euler(h)
            # old target had handle roughly vertical: pitch near -90 or similar
            if abs(e[0] + 90) < 5 and abs(e[1]) < 10:
                print(f"Candidate roll={roll} pitch={pitch} yaw={yaw} -> handle euler {tuple(round(v,1) for v in e)}")

print("\nOld spawn (roll=90, yaw=180) on NEW usd:")
h = apply_spawn(90, 0, 180, handle_xf)
print("  handle:", tuple(round(v, 1) for v in rot_to_euler(h)))

print("No-roll spawn (roll=0, yaw=180) on NEW usd:")
h = apply_spawn(0, 0, 180, handle_xf)
print("  handle:", tuple(round(v, 1) for v in rot_to_euler(h)))
print("  base:", tuple(round(v, 1) for v in rot_to_euler(apply_spawn(0, 0, 180, base_xf))))
print("  layout +Z dir:", apply_spawn(0, 0, 180, layout_xf).TransformDir(Gf.Vec3d(0, 0, 1)))

print("\nNo-roll spawn (roll=0, yaw=90 default usd_yaw) on NEW usd:")
h = apply_spawn(0, 0, 90, handle_xf)
print("  handle:", tuple(round(v, 1) for v in rot_to_euler(h)))
print("  base:", tuple(round(v, 1) for v in rot_to_euler(apply_spawn(0, 0, 90, base_xf))))

# What spawn makes pegboard normal point -Z (vertical board facing robot)?
print("\nSearch pegboard normal ~= (0,0,-1) after spawn:")
for roll in [0.0, 90.0, -90.0]:
    for pitch in [0.0, 90.0, -90.0]:
        for yaw in [0.0, 90.0, 180.0, 270.0]:
            n = apply_spawn(roll, pitch, yaw, layout_xf).TransformDir(Gf.Vec3d(0, 0, 1))
            n.Normalize()
            if abs(n[0]) < 0.1 and abs(n[1]) < 0.1 and n[2] < -0.9:
                print(f"  roll={roll} pitch={pitch} yaw={yaw} normal=({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f})")
