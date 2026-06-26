# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Alex standing balance RL training

Train a lower-body policy that keeps Alex upright while the upper body is held at a
nominal pose with random arm disturbances.

## Train (inside the container)

**Recommended for ability-hands teleop deploy:** `alex_wbc_standing_rl` (same URDF/mass as
`alex_wbc_ability_hands`). Nubs (`alex_standing_rl`) is faster to iterate but does not match
the hands-equipped robot.

```bash
/isaac-sim/python.sh submodules/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --headless \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task alex_standing_balance \
  --embodiment alex_wbc_standing_rl \
  --num_envs 1024 \
  --max_iterations 3000
```

Checkpoints land under `logs/rsl_rl/alex_standing_balance/<timestamp>/` with
`params/agent.yaml` saved alongside each `model_*.pt`.

Monitor with TensorBoard:

```bash
/isaac-sim/python.sh -m tensorboard.main --logdir logs/rsl_rl/alex_standing_balance
```

## Deploy on a WBC teleop embodiment

Use `alex_wbc_ability_hands` (or `alex_v2_wbc_ability_hands`) with teleop — arms via Pink IK,
legs via the RL checkpoint:

```bash
/isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/teleop.py \
  --device cuda --viz kit \
  alex_teleop_sandbox \
  --teleop_device openxr \
  --embodiment alex_wbc_ability_hands \
  --standing_model_path logs/rsl_rl/alex_standing_balance/<timestamp>/model_2999.pt
```

`--standing_model_path` implies `--standing_wbc_version rl`. For classical PD instead,
omit the checkpoint or pass `--standing_wbc_version standing_pd`.

Door / doorman teleop envs accept the same flags.

## Notes

- Training uses **nubs** Alex (`alex_standing_rl` / `alex_v2_standing_rl`) with 13-D
  lower-body actions (12 legs + `SPINE_Z`).
- Observations: `base_ang_vel`, `projected_gravity`, lower-body `joint_pos_rel`,
  `joint_vel_rel`, `last_action` (45-D total).
- Actions are scaled joint deltas: `q_target = q_default + 0.25 * policy_output`.
