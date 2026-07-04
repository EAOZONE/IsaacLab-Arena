"""Author physics on Lever.usd.

- Static (triangle-mesh) colliders on all layout meshes.
- Rigid body + convex-hull colliders on the blue lever assembly
  (Hex_Nut_..._v1_1: hex nut + handle).
- Revolute joint about the nut's local Y axis, anchored to the valve base_link.
- ArticulationRootAPI on /World so Arena detects the asset as an articulation.

Run inside the container:
    /isaac-sim/python.sh .lever_tmp/add_lever_physics.py
"""

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

PATH = "/workspaces/isaaclab_arena/isaaclab_arena/assets/lever_sim/Lever.usd"

WORLD = "/World"
LAYOUT = WORLD + "/Layout_v9"
BLUE_VALVE = LAYOUT + "/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3"
BASE_LINK = BLUE_VALVE + "/base_link_1/base_link"
NUT = BASE_LINK + "/Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1"
VALVE_BODY = BASE_LINK + "/Body1"
PHYSICS_BASE = WORLD + "/PhysicsBase"
JOINT_PATH = PHYSICS_BASE + "/lever_joint"

stage = Usd.Stage.Open(PATH)
nut_prim = stage.GetPrimAtPath(NUT)
base_prim = stage.GetPrimAtPath(BASE_LINK)
assert nut_prim and base_prim, "expected prims not found"

# --- colliders on all render meshes ---
num_static, num_dynamic = 0, 0
for prim in stage.Traverse():
    if prim.GetTypeName() != "Mesh":
        continue
    path = str(prim.GetPath())
    if path.startswith(WORLD + "/Looks") or path.startswith("/Render"):
        continue
    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    if path.startswith(NUT):
        # dynamic bodies need convex collision shapes
        mesh_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
        num_dynamic += 1
    else:
        mesh_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.none)
        num_static += 1
print(f"colliders: {num_static} static (trimesh), {num_dynamic} dynamic (convex hull)")

# --- rigid body on the nut + handle assembly ---
UsdPhysics.RigidBodyAPI.Apply(nut_prim)
UsdPhysics.MassAPI.Apply(nut_prim).CreateMassAttr().Set(0.2)

# --- articulation root on the default prim ---
UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(WORLD))

# --- dummy base link at the asset origin ---
# A joint whose body0 is a static (non-rigid-body) prim parses as an external
# constraint, not an articulation DOF. So the articulation needs a rigid base
# link of its own; it is welded to the static layout by the fixed joint below.
base_link_prim = UsdGeom.Xform.Define(stage, Sdf.Path(PHYSICS_BASE)).GetPrim()
UsdPhysics.RigidBodyAPI.Apply(base_link_prim)
UsdPhysics.MassAPI.Apply(base_link_prim).CreateMassAttr().Set(1.0)

cache = UsdGeom.XformCache()
static_l2w = cache.GetLocalToWorldTransform(base_prim)
dummy_l2w = cache.GetLocalToWorldTransform(base_link_prim)
nut_l2w = cache.GetLocalToWorldTransform(nut_prim)

# fixed joint: static valve base_link -> dummy base link (all poses relative,
# so the anchor follows the asset wherever it is spawned)
fix_rel = dummy_l2w * static_l2w.GetInverse()
fixed = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(PHYSICS_BASE + "/root_joint"))
fixed.CreateBody0Rel().SetTargets([Sdf.Path(BASE_LINK)])
fixed.CreateBody1Rel().SetTargets([Sdf.Path(PHYSICS_BASE)])
fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(fix_rel.ExtractTranslation()))
fixed.CreateLocalRot0Attr().Set(Gf.Quatf(fix_rel.ExtractRotationQuat().GetNormalized()))
fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
fixed.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

# --- revolute joint dummy base link -> nut, axis = nut local Y ---
rel = nut_l2w * dummy_l2w.GetInverse()  # nut frame expressed in dummy-base frame
rel_t = rel.ExtractTranslation()
rel_q = rel.ExtractRotationQuat().GetNormalized()

joint = UsdPhysics.RevoluteJoint.Define(stage, Sdf.Path(JOINT_PATH))
joint.CreateBody0Rel().SetTargets([Sdf.Path(PHYSICS_BASE)])
joint.CreateBody1Rel().SetTargets([Sdf.Path(NUT)])
joint.CreateAxisAttr().Set(UsdPhysics.Tokens.y)
joint.CreateLocalPos0Attr().Set(Gf.Vec3f(rel_t))
joint.CreateLocalRot0Attr().Set(Gf.Quatf(rel_q))
joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
joint.CreateLowerLimitAttr().Set(-90.0)
joint.CreateUpperLimitAttr().Set(90.0)
# enough friction that the handle holds its position against gravity until
# pushed: gravity torque on the 0.2 kg handle is ~0.075 N*m, so 0.2 N*m holds
# it with margin while staying far below what a robot push exerts.
# (asset must already be flattened to meters — see flatten_lever_units.py)
# (PhysxSchema is not importable outside kit; author the applied schema directly)
joint.GetPrim().AddAppliedSchema("PhysxJointAPI")
joint.GetPrim().CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(0.2)
print(f"joint localPos0={Gf.Vec3f(rel_t)} localRot0={Gf.Quatf(rel_q)}")

# --- the nut/handle is assembled through the valve body: never collide with it ---
UsdPhysics.FilteredPairsAPI.Apply(nut_prim).CreateFilteredPairsRel().SetTargets([Sdf.Path(VALVE_BODY)])

stage.Save()
print("saved", PATH)
