# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the CCIL behavioral-cloning policy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field


@dataclass
class CCILBCPolicyConfig:
    """Configuration for :class:`~isaaclab_arena_ccil.policy.ccil_bc_policy.CCILBCPolicy`.

    The CCIL BC model is produced offline (Python 3.8 + custom d3rlpy) and brought into
    Arena as a dependency-free artifact: either a TorchScript module saved by d3rlpy's
    ``save_policy`` (preferred, scalers baked in) or a plain ``state_dict`` + ``meta.json``
    describing the MLP and normalization (cross-version fallback).
    """

    model_path: str = field(metadata={"help": "Path to the CCIL policy artifact (TorchScript .pt or state_dict .pt)."})
    """Path to the CCIL policy artifact (TorchScript .pt or state_dict .pt)."""

    meta_path: str | None = field(
        default=None,
        metadata={"help": "Path to ccil_bc_meta.json (required only for the state_dict fallback / ref-pair checks)."},
    )
    """Path to ``ccil_bc_meta.json``; required only for the plain-MLP fallback and ref-pair verification."""

    state_key: str = field(
        default="robot_joint_pos",
        metadata={"help": "Observation key under observation['policy'] fed to the policy."},
    )
    """Observation key under ``observation['policy']`` fed to the policy (matches the converter)."""

    num_envs: int = field(default=1, metadata={"help": "Number of environments to simulate."})
    """Number of environments to simulate."""

    policy_device: str = field(default="cuda", metadata={"help": "Device for policy tensor operations."})
    """Device for policy tensor operations."""

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> CCILBCPolicyConfig:
        """Create configuration from parsed CLI arguments."""
        return cls(
            model_path=args.model_path,
            meta_path=getattr(args, "meta_path", None),
            state_key=getattr(args, "state_key", "robot_joint_pos"),
            num_envs=getattr(args, "num_envs", 1),
            policy_device=getattr(args, "policy_device", "cuda"),
        )
