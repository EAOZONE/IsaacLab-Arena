"""Spawn lever via alex_empty defaults, dump transforms, save screenshot.

Run inside the container:
    /isaac-sim/python.sh .lever_tmp/debug_lever_spawn.py --viz none \\
        alex_empty --embodiment alex_v2_ability_hands \\
        --usd isaaclab_arena/assets/lever_sim/Lever_revolute.usd --table none
"""

from __future__ import annotations

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

parser = get_isaaclab_arena_cli_parser()
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import omni.usd
import torch
import warp as wp
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, UsdGeom

HANDLE_SUFFIX = (
    "/Layout_v9/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/base_link_1/base_link/"
    "Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1_1/"
    "Hex_Nut_ANSI_B18_2_2___5_16_24_Steel_Grade_2H_Plain_v1/Handle_1"
)
BASE_SUFFIX = "/Layout_v9/Blue_Handled_Valve_v3_1/Blue_Handled_Valve_v3/base_link_1/base_link"
OUT_PATH = "/workspaces/isaaclab_arena/.lever_tmp/lever_spawn_debug.png"


def _rot_to_euler_deg(rot: Gf.Rotation) -> tuple[float, float, float]:
    d = rot.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
    return (d[0], d[1], d[2])


def _dir_str(v: Gf.Vec3d) -> str:
    v.Normalize()
    return f"({v[0]:+.2f}, {v[1]:+.2f}, {v[2]:+.2f})"


def main() -> None:
    from isaaclab_arena_environments import lever_scene_builder

    print("=== configured spawn ===")
    print("  pos:", lever_scene_builder.LEVER_USD_DEFAULT_POS)
    print("  yaw:", lever_scene_builder.LEVER_USD_DEFAULT_YAW)
    print("  scale:", lever_scene_builder.LEVER_USD_DEFAULT_SCALE)

    arena_builder = get_arena_builder_from_cli(args_cli)
    env = arena_builder.make_registered()
    env.reset()

    lever = env.unwrapped.scene["lever_revolute"]
    root_pos = wp.to_torch(lever.data.root_pos_w)[0].tolist()
    root_quat = wp.to_torch(lever.data.root_quat_w)[0].tolist()
    print("\n=== IsaacLab RigidObject root (env 0) ===")
    print("  pos:", [round(v, 5) for v in root_pos])
    print("  quat xyzw:", [round(v, 5) for v in root_quat])

    stage = omni.usd.get_context().get_stage()
    lever_prim_path = "/World/envs/env_0/lever_revolute"
    cache = UsdGeom.XformCache()
    for label, suffix in [("object root", ""), ("base_link", BASE_SUFFIX), ("Handle_1", HANDLE_SUFFIX)]:
        p = stage.GetPrimAtPath(lever_prim_path + suffix)
        if not p.IsValid():
            print(f"MISSING {label}: {lever_prim_path + suffix}")
            continue
        xf = cache.GetLocalToWorldTransform(p)
        rot = xf.ExtractRotation()
        print(f"\n=== USD world {label} ===")
        print("  path:", p.GetPath())
        print("  pos:", [round(v, 5) for v in xf.ExtractTranslation()])
        print("  euler XYZ deg:", tuple(round(v, 1) for v in _rot_to_euler_deg(rot)))
        print("  local +Z dir:", _dir_str(rot.TransformDir(Gf.Vec3d(0, 0, 1))))

    base_p = stage.GetPrimAtPath(lever_prim_path + BASE_SUFFIX)
    handle_p = stage.GetPrimAtPath(lever_prim_path + HANDLE_SUFFIX)
    if base_p.IsValid() and handle_p.IsValid():
        bxf = cache.GetLocalToWorldTransform(base_p)
        hxf = cache.GetLocalToWorldTransform(handle_p)
        off = hxf.ExtractTranslation() - bxf.ExtractTranslation()
        print("\n=== handle offset from base (world) ===")
        print("  dir:", _dir_str(Gf.Vec3d(off)))
        print("  distance:", round((hxf.ExtractTranslation() - bxf.ExtractTranslation()).GetLength(), 5))

    action = torch.zeros(env.action_space.shape[-1], device=env.unwrapped.device)
    for _ in range(120):
        env.step(action.unsqueeze(0))

    cache.Clear()
    if handle_p.IsValid():
        hxf2 = cache.GetLocalToWorldTransform(handle_p)
        rot2 = hxf2.ExtractRotation()
        print("\n=== Handle_1 after 120 steps ===")
        print("  euler XYZ deg:", tuple(round(v, 1) for v in _rot_to_euler_deg(rot2)))
        print("  local +Z dir:", _dir_str(rot2.TransformDir(Gf.Vec3d(0, 0, 1))))

    look_at = np.array(root_pos)
    set_camera_view(eye=look_at + np.array([0.8, -0.8, 0.5]), target=look_at)
    for _ in range(60):
        simulation_app.update()

    import omni.kit.viewport.utility as vp_util

    vp = vp_util.get_active_viewport()
    vp_util.capture_viewport_to_file(vp, OUT_PATH)
    for _ in range(60):
        simulation_app.update()
    print(f"\nSaved screenshot: {OUT_PATH}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
