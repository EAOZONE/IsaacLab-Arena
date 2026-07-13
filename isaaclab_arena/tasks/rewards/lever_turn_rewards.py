# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence

import warp as wp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.utils.math import quat_error_magnitude


class HingeAngleFromRest(ManagerTermBase):
    """Unsigned rotation (rad) of a raw (non-articulated) rigid-body hinge from its settled rest pose.

    The lever handle is spawned as a single dynamic ``RigidObject`` connected to a static base by a
    raw ``UsdPhysics.RevoluteJoint`` authored directly in the USD -- it is not an IsaacLab
    ``Articulation``, so there is no ``joint_pos`` to read (see ``isaaclab_arena/utils/joint_utils.py``,
    which only works for real articulations). Since the joint has exactly one DOF, the full geodesic
    rotation between the handle's current world orientation and its rest orientation *is* the hinge
    angle, regardless of the joint's own axis/frame conventions -- no need to know the axis or the
    static base's world pose.

    The handle's authored USD pose is not itself physically stable: the joint carries a baked-in
    angular drive (see ``Lever_revolute.usd``'s ``RevoluteJoint`` ``DriveAPI``) that pulls it to the
    correct visual rest pose from wherever it's spawned, taking on the order of 10-15 physics steps to
    settle (observed: ang. speed decays from ~54 rad/s to <0.05 rad/s by step 9, see
    ``.lever_tmp/diag_lever_rl5.py``). Freezing "rest" too early (e.g. after a single fixed step) reads
    that settle motion itself as hinge rotation -- which previously caused ``LeverTurnSuccess`` to fire
    on essentially every episode within 1-2 steps regardless of policy behavior. So instead of a single
    capture, ``_rest_quat_w`` is re-synced to the current pose on every step while the handle is still
    moving fast, and only frozen once its angular speed decays below ``_SETTLE_ANG_SPEED_THRESHOLD``
    (with ``_SETTLE_TIMEOUT_STEPS`` as a fallback in case gains ever change enough that it never fully
    settles). Used as both an observation term and (via its own ``RewardTermCfg``) a reward term; each
    manager instantiates and resets its own copy independently.
    """

    _SETTLE_ANG_SPEED_THRESHOLD = 0.05
    """Angular speed (rad/s) below which the drive-settle transient is considered done."""

    _SETTLE_TIMEOUT_STEPS = 100
    """Steps after which rest is frozen regardless of angular speed, so a non-settling handle can't
    keep "rest" tracking the live pose (and hence reward/success permanently at zero) all episode."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg: SceneEntityCfg = cfg.params["object_cfg"]
        self._object_name = object_cfg.name
        # Identity quaternion in Isaac Lab's (x, y, z, w) layout; overwritten during settle tracking.
        self._rest_quat_w = torch.zeros(env.num_envs, 4, device=env.device)
        self._rest_quat_w[:, 3] = 1.0
        # Isaac Lab defers ManagerTermBase instantiation until sim play, so reward/termination
        # managers may never register this term in _class_term_cfgs and thus never call reset().
        # Track per-env settle state explicitly and also re-arm on episode_length_buf == 1.
        self._tracking = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._tracking[env_ids] = True

    def _sync_rest_quat_if_needed(self, env: ManagerBasedRLEnv, current_quat_w: torch.Tensor) -> None:
        self._tracking |= env.episode_length_buf == 1
        if not self._tracking.any():
            return
        env_ids = self._tracking.nonzero(as_tuple=True)[0]
        self._rest_quat_w[env_ids] = current_quat_w[env_ids].clone()

        object: RigidObject = env.scene[self._object_name]
        ang_speed = torch.norm(wp.to_torch(object.data.root_ang_vel_w), dim=-1)
        settled = ang_speed < self._SETTLE_ANG_SPEED_THRESHOLD
        timed_out = env.episode_length_buf >= self._SETTLE_TIMEOUT_STEPS
        self._tracking &= ~(settled | timed_out)

    def __call__(self, env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
        object: RigidObject = env.scene[object_cfg.name]
        current_quat_w = wp.to_torch(object.data.root_quat_w)
        self._sync_rest_quat_if_needed(env, current_quat_w)
        return quat_error_magnitude(current_quat_w, self._rest_quat_w)


class HingeAngleFromRestObs(HingeAngleFromRest):
    """Like :class:`HingeAngleFromRest`, but returns shape ``(num_envs, 1)`` for obs concatenation."""

    def __call__(self, env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
        return super().__call__(env, object_cfg).unsqueeze(-1)


def hand_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward the agent for reaching the lever handle using a tanh-kernel.

    Same shape as ``isaaclab_arena.tasks.rewards.rewards.object_ee_distance``, but reads the hand
    position via a body-index lookup on the robot articulation (``robot_cfg.body_ids``) rather than a
    ``FrameTransformer`` scene entity -- Alex embodiments don't set one up (see
    ``AlexAbilityHandObservationsCfg`` in ``isaaclab_arena/embodiments/alex/alex.py``, which reads
    end-effector poses the same way via ``mdp.get_eef_pos``).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    hand_pos_w = wp.to_torch(robot.data.body_pos_w)[:, robot_cfg.body_ids[0]]
    object_pos_w = wp.to_torch(object.data.root_pos_w)
    distance = torch.norm(hand_pos_w - object_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


# Right ability-hand joints, in the order declared by ``ABILITY_HAND_TELEOP_JOINT_ORDER`` in
# ``isaaclab_arena/embodiments/alex/alex.py`` (only the ``right_*`` half). For every one of these
# joints the URDF's upper limit is the fully-closed pose and the lower limit is fully-open (see
# ``_ABILITY_HAND_JOINT_LIMITS_BY_SUFFIX`` in that file) -- including ``thumb_q1``, whose limits
# happen to be negative-to-zero rather than positive, but the same upper=closed convention holds.
RIGHT_ABILITY_HAND_JOINT_NAMES = [
    "right_ability_hand_index_q1",
    "right_ability_hand_middle_q1",
    "right_ability_hand_ring_q1",
    "right_ability_hand_pinky_q1",
    "right_ability_hand_thumb_q1",
    "right_ability_hand_index_q2",
    "right_ability_hand_middle_q2",
    "right_ability_hand_ring_q2",
    "right_ability_hand_pinky_q2",
    "right_ability_hand_thumb_q2",
]


def right_hand_closedness(env: ManagerBasedRLEnv, hand_joint_cfg: SceneEntityCfg) -> torch.Tensor:
    """Worst-case (min) normalized closedness (0 = fully open, 1 = fully closed) of the right ability hand.

    Reads live ``joint_pos_limits`` from the articulation rather than a hardcoded copy, since
    ``(pos - lower) / (upper - lower)`` already equals 0/1 at open/closed for every joint in
    ``RIGHT_ABILITY_HAND_JOINT_NAMES`` per the upper=closed convention documented there.

    Uses ``.min()`` rather than ``.mean()`` across joints so a full-fist grasp is required to earn
    the reward -- a mean lets curling a single finger (e.g. just the index, to hook the handle)
    pay off almost as well as closing the whole hand, which is what the policy learned when this
    used ``.mean()``.
    """
    robot: Articulation = env.scene[hand_joint_cfg.name]
    joint_pos = wp.to_torch(robot.data.joint_pos)[:, hand_joint_cfg.joint_ids]
    limits = wp.to_torch(robot.data.joint_pos_limits)[:, hand_joint_cfg.joint_ids, :]
    closedness = (joint_pos - limits[..., 0]) / (limits[..., 1] - limits[..., 0])
    return closedness.min(dim=-1).values


def grasp_readiness(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    hand_joint_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward closing the right hand, gated by proximity to the lever handle.

    ``hand_object_distance``'s proximity kernel times :func:`right_hand_closedness`, so closing
    the hand only pays off near the handle -- not an unconditional incentive to clench the fist
    from across the scene -- while still giving a dense, differentiable signal that ramps up as
    the policy learns to reach in, mirroring how ``hand_object_distance`` itself shapes reaching.
    """
    proximity = hand_object_distance(env, std, object_cfg, robot_cfg)
    return proximity * right_hand_closedness(env, hand_joint_cfg)


class LeverTurnSuccess(HingeAngleFromRest):
    """Termination: True once the hinge has stayed past ``angle_threshold`` rad for a sustained window.

    A single-step angle check fires on a momentary, incidental knock from the arm/hand passing near
    the handle just as easily as on a deliberate turn -- the handle's weak drive damping (see
    :class:`HingeAngleFromRest`) makes it cheap to nudge past threshold without actually grasping it.
    Requiring ``_DEBOUNCE_STEPS`` consecutive steps above threshold filters out those transient bumps
    while still counting a real, held turn as success promptly (``_DEBOUNCE_STEPS`` * step_dt is small
    relative to ``episode_length_s``).
    """

    _DEBOUNCE_STEPS = 50

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._steps_above_threshold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._last_processed_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._steps_above_threshold[env_ids] = 0
        self._last_processed_step[env_ids] = -1

    def __call__(self, env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, angle_threshold: float) -> torch.Tensor:
        angle = super().__call__(env, object_cfg)
        above = angle > angle_threshold
        episode_step = env.episode_length_buf
        reset_ids = episode_step < self._last_processed_step
        self._steps_above_threshold[reset_ids] = 0
        new_step = episode_step != self._last_processed_step
        self._steps_above_threshold[new_step] = torch.where(
            above[new_step],
            self._steps_above_threshold[new_step] + 1,
            torch.zeros_like(self._steps_above_threshold[new_step]),
        )
        self._last_processed_step = episode_step.clone()
        return self._steps_above_threshold >= self._DEBOUNCE_STEPS


class LeverEngaged(HingeAngleFromRest):
    """Monotonic per-episode signal that latches once the lever starts moving."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._engaged = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._last_episode_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._engaged[env_ids] = False
        self._last_episode_step[env_ids] = -1

    def __call__(self, env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, angle_threshold: float) -> torch.Tensor:
        angle = super().__call__(env, object_cfg)
        episode_step = env.episode_length_buf
        reset_ids = episode_step < self._last_episode_step
        self._engaged[reset_ids] = False
        self._engaged |= angle > angle_threshold
        self._last_episode_step = episode_step.clone()
        return self._engaged


class LeverTurnedBonus(HingeAngleFromRest):
    """Sparse 1.0 bonus once the hinge has rotated past ``angle_threshold`` (rad) from rest.

    Mirrors the sparse+dense combination in ``lift_object_rewards.object_is_lifted`` /
    ``object_goal_distance``. Subclasses :class:`HingeAngleFromRest` purely to reuse its per-env
    rest-quat buffer/reset logic -- the reward manager instantiates this as its own independent copy,
    separate from any ``HingeAngleFromRest`` term also configured as an observation or reward.
    """

    def __call__(self, env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, angle_threshold: float) -> torch.Tensor:
        angle = super().__call__(env, object_cfg)
        return (angle > angle_threshold).float()
