# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test lever articulation: spawn env and print joint names / positions."""

from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args(["--headless"])
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab_arena_environments.alex_lever_teleop_environment  # noqa: F401
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

p = get_isaaclab_arena_cli_parser()
add_example_environments_cli_args(p)
a = p.parse_args(["alex_lever_teleop", "--embodiment", "alex_v2_lever_fingers_joint_pos", "--seed", "0"])
builder = get_arena_builder_from_cli(a)
name, cfg = builder.build_registered()
cfg.recorders = {}
cfg.terminations = {}
env = gym.make(name, cfg=cfg).unwrapped
env.reset()

lever = env.scene["lever_layout"]
print("joint names:", lever.joint_names, flush=True)
print("initial joint pos:", lever.data.joint_pos[0].tolist(), flush=True)

for _ in range(120):
    env.sim.step(render=False)
env.scene.update(dt=env.sim.get_physics_dt())
print("after settle joint pos:", lever.data.joint_pos[0].tolist(), flush=True)

# Nudge the blue T-handle downward with an external force via joint velocity impulse on its DOF.
if "blue_handled_valve_lever" in lever.joint_names:
    idx = lever.joint_names.index("blue_handled_valve_lever")
    vel = torch.zeros_like(lever.data.joint_vel)
    vel[0, idx] = 3.0
    lever.write_joint_velocity_to_sim(vel)
    for _ in range(60):
        env.sim.step(render=False)
    env.scene.update(dt=env.sim.get_physics_dt())
    print("after impulse joint pos:", lever.data.joint_pos[0, idx].item(), flush=True)

env.close()
app.close()
