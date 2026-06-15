# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Captury markerless mocap teleoperation for Isaac Lab Arena.

This package streams upper-body skeleton data from a Captury Live system
(https://captury.com/) and converts it into the same OpenXR-style ``HandInput``
tensors used by Arena's existing IsaacTeleop retargeting pipelines.  Any
embodiment that supports OpenXR hand-tracking teleop can therefore be driven
from Captury by swapping the hands source node — the per-robot retargeters
(SE3 wrist tracking, dex-hand retargeting, PINK IK action spaces) are reused
unchanged.

Modules:
    * :mod:`~isaaclab_arena.teleop.captury.captury_skeleton` — pure-numpy
      conversion from Captury global joint transforms to OpenXR hand arrays.
    * :mod:`~isaaclab_arena.teleop.captury.captury_client` — network client
      wrapping the ``remotecaptury`` streaming library.
    * :mod:`~isaaclab_arena.teleop.captury.captury_hands_source` — isaacteleop
      pipeline source node (requires ``isaacteleop``).
    * :mod:`~isaaclab_arena.teleop.captury.captury_teleop_device` — Isaac Lab
      teleop device that executes a retargeting pipeline directly from Captury
      data, without an OpenXR session.
"""

from isaaclab_arena.teleop.captury.captury_client import CapturyClient
from isaaclab_arena.teleop.captury.captury_skeleton import (
    CapturyHandJointMap,
    CapturySkeletonMap,
    build_skeleton_map_from_joint_names,
    captury_hand_to_openxr_arrays,
    captury_transforms_to_matrices,
    captury_upper_arm_directions,
)

__all__ = [
    "CapturyClient",
    "CapturyHandJointMap",
    "CapturySkeletonMap",
    "build_skeleton_map_from_joint_names",
    "captury_hand_to_openxr_arrays",
    "captury_transforms_to_matrices",
    "captury_upper_arm_directions",
]
