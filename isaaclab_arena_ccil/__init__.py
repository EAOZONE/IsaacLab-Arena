# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab Arena integration for the CCIL behavioral-cloning policy.

CCIL (https://github.com/personalrobotics/CCIL) is a state-based imitation-learning
method built on a custom ``d3rlpy`` fork (Python 3.8). This package keeps the heavy,
version-incompatible training/export offline and brings only a dependency-free
TorchScript / weight artifact into the Arena interpreter for closed-loop evaluation.
"""
