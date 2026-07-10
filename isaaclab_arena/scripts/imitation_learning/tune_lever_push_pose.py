# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Interactively jog Alex's wrist with the keyboard to find a lever push point.

``record_scripted_lever_demos.py``'s ``--push_local_offset`` (and, for
``lever_again``, the extra ``_LEVER_AGAIN_RIGHT_FIST_CONTACT_OFFSET`` hack) are
untuned guesses at where, in the lever handle's own rest-pose frame, the closed
fist should press down. Guessing those numbers blind and re-running the
recorder headless is slow. This script instead spawns the same scene, holds
one arm's hand in the push (fist) pose, and lets you drive the wrist around
with the keyboard while watching it live in the Kit viewport -- so you can
visually park the fist right on the handle before reading off the offset to
paste back into the recorder.

Keys:
    W/S             move wrist +/- world X
    A/D             move wrist +/- world Y
    Q/E             move wrist +/- world Z (up/down)
    K               toggle hand open / closed thumbs-up fist
    R               reset the wrist target back to the arm's home pose
    P               print the current offset (also printed periodically)

Orientation is not jangled -- the wrist keeps its starting (home) orientation
the whole time, matching the recorder's own "press straight down, no wrist
rotation" design. If you need ``--push_wrist_rot_offset`` too, tune it
separately; this tool is position-only.

Run inside the container, with a GUI so you can see the viewport::

    /isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/tune_lever_push_pose.py \\
        --viz kit --object_name lever_again --arm right \\
        alex_empty --embodiment alex_v2_ability_hands \\
        --usd isaaclab_arena/assets/lever_sim/LEVER_AGAIN.usd
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)

parser = get_isaaclab_arena_cli_parser()
parser.add_argument(
    "--object_name",
    type=str,
    default="lever_again",
    help="Scene key of the lever rigid object to reach for (matches the --usd asset's prim name).",
)
parser.add_argument(
    "--arm",
    type=str,
    choices=["left", "right"],
    default="right",
    help="Which arm to jog.",
)
parser.add_argument(
    "--pos_sensitivity",
    type=float,
    default=0.005,
    help="Wrist target movement [m] per keypress (held keys repeat at the OS key-repeat rate).",
)
parser.add_argument(
    "--close_fraction",
    type=float,
    default=1.0,
    help="How far the four fingers close into a fist when toggled closed (K); the thumb stays extended.",
)
parser.add_argument(
    "--print_every",
    type=int,
    default=30,
    help="Print the live offset every N sim steps, in addition to on-demand (P).",
)
# NOTE(alexmillane, 2025.09.04): This has to be added last, because
# of the app specific flags being parsed after the global flags.
add_example_environments_cli_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import warp as wp
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_arena.embodiments.alex.alex import (
    ALEX_ABILITY_HAND_WRIST_ACTION_DIM,
    build_ability_hand_joint_action,
    build_ability_hand_thumbs_up_action,
)


def _create_environment():
    """Build the (unwrapped) env, matching record_scripted_lever_demos.py minus the recorder."""
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    import gymnasium as gym

    from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    reapply_viewer_cfg(env)
    return env


def _body_pose(env, link_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Live (pos, quat_xyzw) of ``link_name`` on the robot, relative to its env origin."""
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies([link_name])
    idx = int(body_ids[0])
    pos = wp.to_torch(robot.data.body_pos_w)[0, idx] - env.scene.env_origins[0]
    quat = wp.to_torch(robot.data.body_quat_w)[0, idx]
    return pos.clone(), quat.clone()


def _lever_handle_prim_pose(env, object_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Live Handle_1 prim pose for lever USDs spawned as static/base assets.

    Uses ``isaacsim.core.prims.RigidPrim`` (Fabric-synced), not raw
    ``UsdGeom.Xformable.ComputeLocalToWorldTransform`` -- the latter silently returns the
    USD-authored (design-time) transform and never reflects PhysX simulation results at all
    (confirmed: 300 steps under active gravity produced a bit-identical pose via the Xformable
    read, while RigidPrim showed continuous drift over the same run).
    """
    from isaacsim.core.prims import RigidPrim

    from isaaclab_arena_environments.lever_scene_builder import LEVER_HANDLE_RIGID_BODY_SUFFIX

    prim_path = f"/World/envs/env_0/{object_name}{LEVER_HANDLE_RIGID_BODY_SUFFIX}"
    pos_w, quat_wxyz = RigidPrim(prim_path).get_world_poses()
    pos = pos_w[0].to(device=env.device, dtype=env.scene.env_origins.dtype)
    # RigidPrim is scalar-first (w, x, y, z); the rest of this codebase uses xyzw.
    quat = quat_wxyz[0][[1, 2, 3, 0]].to(device=env.device, dtype=pos.dtype)
    return pos - env.scene.env_origins[0], quat


def _object_pose(env, object_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Live (pos, quat_xyzw) of the lever handle, relative to its env origin."""
    assert object_name in env.scene.keys(), (
        f"'{object_name}' not found in the scene (available: {list(env.scene.keys())}). Pass --object_name to"
        " match the --usd asset's prim name."
    )
    data = getattr(env.scene[object_name], "data", None)
    if data is None or not hasattr(data, "root_pos_w"):
        return _lever_handle_prim_pose(env, object_name)

    pos = wp.to_torch(data.root_pos_w)[0] - env.scene.env_origins[0]
    quat = wp.to_torch(data.root_quat_w)[0]
    return pos.clone(), quat.clone()


def main() -> None:
    env = _create_environment()
    assert env.num_envs == 1, f"Interactive lever tuning only supports --num_envs 1, got {env.num_envs}"
    assert env.action_manager.total_action_dim >= ALEX_ABILITY_HAND_WRIST_ACTION_DIM, (
        "This script targets the ability-hands (Pink IK, EE-pose action) embodiments."
    )

    env.reset()

    active_arm = args_cli.arm
    idle_arm = "left" if active_arm == "right" else "right"
    active_link = f"{active_arm.upper()}_GRIPPER_Z_LINK"
    idle_link = f"{idle_arm.upper()}_GRIPPER_Z_LINK"

    home_pos, home_quat = _body_pose(env, active_link)
    idle_pos, idle_quat = _body_pose(env, idle_link)
    rest_handle_pos, rest_handle_quat = _object_pose(env, args_cli.object_name)

    open_hand = build_ability_hand_joint_action(0.0, 0.0, device=env.device)
    closed_hand = build_ability_hand_thumbs_up_action(
        args_cli.close_fraction if active_arm == "left" else 0.0,
        args_cli.close_fraction if active_arm == "right" else 0.0,
        device=env.device,
    )

    state = {
        "target_pos": home_pos.clone(),
        "hand_closed": True,
        "reset_requested": False,
        "print_requested": True,
    }

    def _print_offset() -> None:
        local_offset = quat_apply_inverse(
            rest_handle_quat.unsqueeze(0), (state["target_pos"] - rest_handle_pos).unsqueeze(0)
        ).squeeze(0)
        pos = state["target_pos"]
        print(
            f"[tune_lever_push_pose] wrist world pos = ({pos[0]:.5f}, {pos[1]:.5f}, {pos[2]:.5f})  "
            f"hand={'closed' if state['hand_closed'] else 'open'}\n"
            f"    --push_local_offset {local_offset[0]:.5f},{local_offset[1]:.5f},{local_offset[2]:.5f}"
        )

    def _request_reset() -> None:
        state["reset_requested"] = True

    def _request_print() -> None:
        state["print_requested"] = True

    teleop_interface = Se3Keyboard(
        Se3KeyboardCfg(
            pos_sensitivity=args_cli.pos_sensitivity,
            rot_sensitivity=0.0,
            gripper_term=True,
            sim_device=str(env.device),
        )
    )
    teleop_interface.add_callback("R", _request_reset)
    teleop_interface.add_callback("P", _request_print)

    print(f"Using teleop device: {teleop_interface}")
    print(
        "[tune_lever_push_pose] W/S=+-X  A/D=+-Y  Q/E=+-Z(up/down)  K=toggle fist  R=reset wrist  "
        "P=print offset now"
    )

    step_index = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            command = teleop_interface.advance()
            delta_pos = command[:3]
            gripper_cmd = float(command[6])
            state["hand_closed"] = gripper_cmd < 0.0

            if state["reset_requested"]:
                state["target_pos"] = home_pos.clone()
                state["reset_requested"] = False
                print("[tune_lever_push_pose] wrist target reset to home pose.")
            else:
                state["target_pos"] = state["target_pos"] + delta_pos

            hand = closed_hand if state["hand_closed"] else open_hand
            targets = {
                active_arm: (state["target_pos"], home_quat),
                idle_arm: (idle_pos, idle_quat),
            }
            action = torch.cat([*targets["left"], *targets["right"], hand]).unsqueeze(0)
            env.step(action)

            step_index += 1
            if state["print_requested"] or step_index % args_cli.print_every == 0:
                _print_offset()
                state["print_requested"] = False

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
