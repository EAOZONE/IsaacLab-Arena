"""Which meshes under the live (spawned, scaled) Handle_1 actually have collision, and where do
they sit in the same world frame record_scripted_lever_demos.py's debug telemetry uses?
"""
import sys

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
add_example_environments_cli_args(parser)
args_cli = parser.parse_args(
    [
        "--viz",
        "none",
        "alex_empty",
        "--embodiment",
        "alex_v2_ability_hands",
        "--usd",
        "isaaclab_arena/assets/lever_sim/LEVER_AGAIN.usd",
    ]
)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

arena_builder = get_arena_builder_from_cli(args_cli)
env_name, env_cfg = arena_builder.build_registered()
env = gym.make(env_name, cfg=env_cfg).unwrapped
env.reset()

stage = omni.usd.get_context().get_stage()
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if "lever_again" not in path or "Handle_1" not in path:
        continue
    has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
    has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    if prim.GetTypeName() in ("Mesh", "Xform") and (has_collision or has_rigid or prim.GetTypeName() == "Mesh"):
        xform = UsdGeom.Xformable(prim)
        world_tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_tf.ExtractTranslation()
        bbox = bbox_cache.ComputeWorldBound(prim)
        rng = bbox.ComputeAlignedRange()
        print(
            f"{path} type={prim.GetTypeName()} collision={has_collision} rigid={has_rigid}\n"
            f"    world_translation={tuple(translation)}\n"
            f"    world_bbox_min={tuple(rng.GetMin())} world_bbox_max={tuple(rng.GetMax())}"
        )

env.close()
simulation_app.close()
