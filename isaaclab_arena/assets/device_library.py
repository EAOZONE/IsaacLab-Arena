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

import os
from abc import ABC, abstractmethod
from collections.abc import Callable

from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab_teleop import IsaacTeleopCfg, XrCfg

from isaaclab_arena.assets.register import register_device


class TeleopDeviceBase(ABC):

    name: str | None = None

    def __init__(self, sim_device: str | None = None):
        self.sim_device = sim_device

    @abstractmethod
    def get_device_cfg(self, pipeline_builder: Callable | None = None, embodiment: object | None = None):
        raise NotImplementedError


@register_device
class OpenXRCfg(TeleopDeviceBase):
    name = "openxr"

    def __init__(self, sim_device: str | None = None):
        super().__init__(sim_device=sim_device)

    def get_device_cfg(
        self,
        pipeline_builder: Callable | None = None,
        embodiment: object | None = None,
        retargeters_to_tune: Callable | None = None,
    ) -> IsaacTeleopCfg:
        if pipeline_builder is None:
            raise ValueError("OpenXRCfg requires a pipeline_builder (got None)")
        xr_cfg = embodiment.get_xr_cfg() if embodiment is not None else XrCfg()
        target_frame_prim_path = embodiment.get_teleop_target_frame_prim_path()
        return IsaacTeleopCfg(
            pipeline_builder=pipeline_builder,
            sim_device=self.sim_device,
            xr_cfg=xr_cfg,
            target_frame_prim_path=target_frame_prim_path,
            retargeters_to_tune=retargeters_to_tune,
        )


@register_device
class CapturyCfg(TeleopDeviceBase):
    """Captury Live markerless mocap teleop device.

    The Captury Live server address can be set via the ``CAPTURY_HOST`` and
    ``CAPTURY_PORT`` environment variables (the device is constructed without
    arguments by the environment CLI plumbing). Set ``CAPTURY_VISUALIZE_SKELETON=0``
    to hide the tracked-skeleton debug overlay and show only the robot.
    """

    name = "captury"

    def __init__(
        self,
        sim_device: str | None = None,
        host: str | None = None,
        port: int | None = None,
        actor_id: int | None = None,
        joint_names: list[str] | None = None,
        visualize_skeleton: bool | None = None,
    ):
        super().__init__(sim_device=sim_device)
        self.host = host if host is not None else os.environ.get("CAPTURY_HOST", "127.0.0.1")
        self.port = port if port is not None else int(os.environ.get("CAPTURY_PORT", "2101"))
        self.actor_id = actor_id
        self.joint_names = joint_names
        self.visualize_skeleton = (
            visualize_skeleton
            if visualize_skeleton is not None
            else os.environ.get("CAPTURY_VISUALIZE_SKELETON", "1").lower() not in ("0", "false", "no")
        )

    def get_device_cfg(
        self,
        pipeline_builder: Callable | None = None,
        embodiment: object | None = None,
        retargeters_to_tune: Callable | None = None,
    ):
        from isaaclab_arena.teleop.captury.captury_teleop_device import CapturyDeviceCfg

        if pipeline_builder is None:
            raise ValueError("CapturyCfg requires a pipeline_builder (got None)")
        xr_cfg = embodiment.get_xr_cfg() if embodiment is not None else XrCfg()
        target_frame_prim_path = embodiment.get_teleop_target_frame_prim_path()
        return CapturyDeviceCfg(
            pipeline_builder=pipeline_builder,
            sim_device=self.sim_device,
            xr_cfg=xr_cfg,
            target_frame_prim_path=target_frame_prim_path,
            captury_host=self.host,
            captury_port=self.port,
            captury_actor_id=self.actor_id,
            captury_joint_names=self.joint_names,
            captury_visualize_skeleton=self.visualize_skeleton,
        )


@register_device
class KeyboardCfg(TeleopDeviceBase):
    name = "keyboard"

    def __init__(self, sim_device: str | None = None, pos_sensitivity: float = 0.05, rot_sensitivity: float = 0.05):
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

    def get_device_cfg(
        self, pipeline_builder: Callable | None = None, embodiment: object | None = None
    ) -> Se3KeyboardCfg:
        return Se3KeyboardCfg(
            pos_sensitivity=self.pos_sensitivity,
            rot_sensitivity=self.rot_sensitivity,
        )


@register_device
class SpaceMouseCfg(TeleopDeviceBase):
    name = "spacemouse"

    def __init__(self, sim_device: str | None = None, pos_sensitivity: float = 0.05, rot_sensitivity: float = 0.05):
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

    def get_device_cfg(
        self, pipeline_builder: Callable | None = None, embodiment: object | None = None
    ) -> Se3SpaceMouseCfg:
        return Se3SpaceMouseCfg(
            pos_sensitivity=self.pos_sensitivity,
            rot_sensitivity=self.rot_sensitivity,
        )
