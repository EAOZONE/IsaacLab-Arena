# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Alex embodiment for in-place standing balance RL (lower-body actions only)."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.alex.alex import (
    ALEX_V1,
    ALEX_V2,
    AlexSceneCfg,
    CONTROL_DT,
    _configure_wbc_ability_hand_robot,
    _alex_arena_urdf_paths,
    _default_nubs_cfg,
    merge_urdfs,
    _resolve_mesh_paths,
    ALEX_NUBFOREARMS_PARTS,
)
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_alex.alex_env.mdp import alex_standing_rl_mdp
from isaaclab_arena_alex.alex_whole_body_controller.wbc_policy.policy.alex_constants import (
    ALEX_LOWER_BODY_JOINT_NAMES,
    ALEX_STANDING_FULL_JOINT_POS,
    ALEX_STANDING_RL_ACTION_SCALE,
    ALEX_STANDING_TARGET_HEIGHT,
)


def _configure_standing_rl_robot(robot_version: str):
    """Floating-base nubs Alex with standing nominal pose baked into defaults."""
    paths = _alex_arena_urdf_paths(robot_version)
    merged_urdf = merge_urdfs(
        robot_version,
        ALEX_NUBFOREARMS_PARTS,
        output_name=f"alex_{robot_version.lower()}_nubs_arena",
    )
    resolved_urdf = _resolve_mesh_paths(merged_urdf, paths["nubs_resolved"], robot_version)

    robot_cfg = _default_nubs_cfg(robot_version)
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    robot_cfg.spawn.asset_path = resolved_urdf
    robot_cfg.spawn.fix_base = False
    robot_cfg.soft_joint_pos_limit_factor = 1.0
    robot_cfg.init_state.pos = (0.0, 0.0, ALEX_STANDING_TARGET_HEIGHT)
    robot_cfg.init_state.joint_pos = {name: value for name, value in ALEX_STANDING_FULL_JOINT_POS.items()}
    return robot_cfg


def _configure_wbc_standing_rl_robot(robot_version: str):
    """Floating-base ability-hands Alex for standing RL (matches WBC teleop deploy)."""
    robot_cfg, _, _ = _configure_wbc_ability_hand_robot(robot_version)
    robot_cfg.init_state.pos = (0.0, 0.0, ALEX_STANDING_TARGET_HEIGHT)
    return robot_cfg


_LOWER_BODY_SCENE = SceneEntityCfg("robot", joint_names=list(ALEX_LOWER_BODY_JOINT_NAMES))


@configclass
class AlexStandingRLActionsCfg:
    """Lower-body joint-position actions for standing RL."""

    lower_body_joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ALEX_LOWER_BODY_JOINT_NAMES),
        scale=ALEX_STANDING_RL_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        # Bound the raw action so a diverging policy (action std ran away to ~30)
        # cannot command absurd joint targets and blow up the physics step.
        clip={".*": (-5.0, 5.0)},
    )


@configclass
class AlexStandingRLObservationsCfg:
    """Observations aligned with :class:`AlexRLStandingPolicy` inference."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Clips bound each term so a single diverging env cannot feed a near-inf
        # value into the policy/critic (and into the running obs normalizer),
        # which is what corrupts the PPO value target and produces NaNs.
        base_ang_vel = ObsTerm(
            func=mdp_isaac_lab.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-20.0, 20.0),
        )
        projected_gravity = ObsTerm(func=mdp_isaac_lab.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_pos = ObsTerm(
            func=mdp_isaac_lab.joint_pos_rel,
            params={"asset_cfg": _LOWER_BODY_SCENE},
            clip=(-10.0, 10.0),
        )
        joint_vel = ObsTerm(
            func=mdp_isaac_lab.joint_vel_rel,
            params={"asset_cfg": _LOWER_BODY_SCENE},
            clip=(-50.0, 50.0),
        )
        actions = ObsTerm(func=mdp_isaac_lab.last_action, clip=(-10.0, 10.0))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class AlexStandingRLEventCfg:
    """Reset, pose initialization, upper-body hold, and arm disturbances."""

    reset_standing_pose = EventTerm(
        func=alex_standing_rl_mdp.reset_alex_standing_pose,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "joint_positions": ALEX_STANDING_FULL_JOINT_POS},
    )
    reset_arm_noise = EventTerm(
        func=alex_standing_rl_mdp.reset_alex_standing_arm_noise,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    hold_upper_body = EventTerm(
        func=alex_standing_rl_mdp.hold_alex_upper_body_joints,
        mode="interval",
        interval_range_s=(CONTROL_DT, CONTROL_DT),
        params={"asset_cfg": SceneEntityCfg("robot"), "nominal_positions": ALEX_STANDING_FULL_JOINT_POS},
    )
    resample_arm_disturbance = EventTerm(
        func=alex_standing_rl_mdp.resample_alex_standing_arm_noise,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={"asset_cfg": SceneEntityCfg("robot"), "noise_std": 0.15},
    )


class AlexStandingRLEmbodimentBase(EmbodimentBase):
    """Base class for Alex nubs standing-balance RL training."""

    default_arm_mode = ArmMode.DUAL_ARM
    robot_version = ALEX_V1

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = True,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms=concatenate_observation_terms)

        self.scene_config = AlexSceneCfg()
        self.scene_config.robot = _configure_standing_rl_robot(self.robot_version)

        self.action_config = AlexStandingRLActionsCfg()
        self.observation_config = AlexStandingRLObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.event_config = AlexStandingRLEventCfg()


@register_asset
class AlexStandingRLEmbodiment(AlexStandingRLEmbodimentBase):
    """Alex V1 nubs — RL training for in-place standing (13-D lower-body actions)."""

    name = "alex_standing_rl"


@register_asset
class AlexV2StandingRLEmbodiment(AlexStandingRLEmbodimentBase):
    """Alex V2 nubs — RL training for in-place standing."""

    name = "alex_v2_standing_rl"
    robot_version = ALEX_V2


class AlexWBCStandingRLEmbodimentBase(AlexStandingRLEmbodimentBase):
    """Base class for ability-hands standing RL (same MDP, deploy-matched URDF)."""

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = True,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms=concatenate_observation_terms)
        self.scene_config.robot = _configure_wbc_standing_rl_robot(self.robot_version)


@register_asset
class AlexWBCStandingRLEmbodiment(AlexWBCStandingRLEmbodimentBase):
    """Alex V1 ability-hands — standing RL on the WBC teleop URDF."""

    name = "alex_wbc_standing_rl"


@register_asset
class AlexV2WBCStandingRLEmbodiment(AlexWBCStandingRLEmbodimentBase):
    """Alex V2 ability-hands — standing RL on the WBC teleop URDF."""

    name = "alex_v2_wbc_standing_rl"
    robot_version = ALEX_V2
