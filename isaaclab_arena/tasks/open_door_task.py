# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING

import torch

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.tasks.common.open_close_door_mimic import RotateDoorMimicEnvCfg, make_open_door_subtask_obs_cfg
from isaaclab_arena.tasks.rotate_revolute_joint_task import RotateRevoluteJointTask


def _make_ik_aware_is_open(openable_object: Openable, ik_term_name: str):
    """Build a success termination func: door open AND no IK-solver failure this episode.

    Reads the per-env ``ik_failed`` latch from the Pink IK action term (set by
    :class:`IKFailureTrackingPinkInverseKinematicsAction`). Falls back to plain ``is_open`` if
    the term is missing or lacks the latch (e.g. a non-IK embodiment).
    """

    def is_open_and_ik_ok(env, asset_cfg=None, threshold=None):
        opened = openable_object.is_open(env, asset_cfg=asset_cfg, threshold=threshold)
        try:
            term = env.action_manager.get_term(ik_term_name)
        except (KeyError, AttributeError, ValueError):
            return opened
        ik_failed = getattr(term, "ik_failed", None)
        if ik_failed is None:
            return opened
        return opened & ~ik_failed

    return is_open_and_ik_ok


def _make_ik_failed_term(ik_term_name: str):
    """Build a termination func that fires the moment an env's IK solver fails.

    Reads the per-env ``ik_failed`` latch from the Pink IK action term so a failed episode ends
    immediately (rather than running to timeout). Returns all-False if the term/latch is absent.
    """

    def ik_failed(env):
        try:
            term = env.action_manager.get_term(ik_term_name)
        except (KeyError, AttributeError, ValueError):
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        failed = getattr(term, "ik_failed", None)
        if failed is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return failed

    return ik_failed


@register_task
class OpenDoorTask(RotateRevoluteJointTask):
    def __init__(
        self,
        openable_object: Openable,
        openness_threshold: float | None = None,
        reset_openness: float | None = 0.0,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        fail_on_ik_error: bool = False,
        ik_term_name: str = "upper_body_ik",
    ):
        super().__init__(
            openable_object=openable_object,
            target_joint_percentage_threshold=openness_threshold,
            reset_joint_percentage=reset_openness,
            episode_length_s=episode_length_s,
            task_description=task_description,
        )

        self.fail_on_ik_error = fail_on_ik_error
        """If True, an IK-solver failure during an episode prevents it from being scored a success."""
        self.ik_term_name = ik_term_name
        """Name of the Pink IK action term whose per-env ``ik_failed`` latch gates success."""

        self.termination_cfg = self.make_termination_cfg()
        self.task_description = (
            f"Reach out to the {openable_object.name} and open it." if task_description is None else task_description
        )

    def make_termination_cfg(self):
        params = {}
        if self.target_joint_percentage_threshold is not None:
            params["threshold"] = self.target_joint_percentage_threshold
        if self.fail_on_ik_error:
            success_func = _make_ik_aware_is_open(self.openable_object, self.ik_term_name)
        else:
            success_func = self.openable_object.is_open
        success = TerminationTermCfg(
            func=success_func,
            params=params,
        )
        cfg = TerminationsCfg(success=success)
        if self.fail_on_ik_error:
            # End the episode the instant the IK solver fails (don't wait for timeout).
            cfg.ik_failure = TerminationTermCfg(func=_make_ik_failed_term(self.ik_term_name))
        return cfg

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        return RotateDoorMimicEnvCfg(
            arm_mode=arm_mode,
            openable_object_name=self.openable_object.name,
        )

    def get_mimic_subtask_obs_cfg(self):
        return make_open_door_subtask_obs_cfg(self.openable_object)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)

    # Dependent on the openable object, so this is passed in from the task at
    # construction time.
    success: TerminationTermCfg = MISSING

    ik_failure: TerminationTermCfg | None = None
    """Set by the task when ``fail_on_ik_error`` is enabled; ends the episode on an IK-solver failure."""
