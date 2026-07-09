"""Recompute RevoluteJoint localRot0/localPos0 so joint angle = 0 at the authored pose; zero the drive target."""
from pxr import Usd, UsdGeom, Gf, Sdf

path = "/workspaces/isaaclab_arena/isaaclab_arena/assets/lever_sim/Lever_revolute.usd"
stage = Usd.Stage.Open(path)
BASE = "/World/Layout_v9/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/base_link_1/base_link"
HEX = BASE + "/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1"
HANDLE = HEX + "/Handle_1/Handle"
JOINT = BASE + "/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1/RevoluteJoint"

cache = UsdGeom.XformCache()
X0 = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(HEX))
X1 = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(HANDLE))

j = stage.GetPrimAtPath(JOINT)
lp1 = Gf.Vec3d(j.GetAttribute("physics:localPos1").Get())
lr1 = Gf.Quatd(j.GetAttribute("physics:localRot1").Get())

m1 = Gf.Matrix4d(); m1.SetTransform(Gf.Rotation(lr1), lp1)
F1 = m1 * X1                     # joint frame1 in asset coords
m0 = F1 * X0.GetInverse()        # required frame0 local transform
rot0 = m0.ExtractRotation().GetQuat()
pos0 = m0.ExtractTranslation()
rot0.Normalize()
print("new localPos0:", pos0)
print("new localRot0 (wxyz):", rot0)

j.GetAttribute("physics:localPos0").Set(Gf.Vec3f(pos0))
j.GetAttribute("physics:localRot0").Set(Gf.Quatf(rot0))
j.GetAttribute("drive:angular:physics:targetPosition").Set(0.0)

# sanity: recompute residual
lp0 = Gf.Vec3d(j.GetAttribute("physics:localPos0").Get())
lr0 = Gf.Quatd(j.GetAttribute("physics:localRot0").Get())
m0c = Gf.Matrix4d(); m0c.SetTransform(Gf.Rotation(lr0), lp0)
F0 = m0c * X0
rel = (F1 * F0.GetInverse()).ExtractRotation()
print("residual after fix: angle", rel.GetAngle(), "deg")

stage.GetRootLayer().Save()
print("saved", path)
