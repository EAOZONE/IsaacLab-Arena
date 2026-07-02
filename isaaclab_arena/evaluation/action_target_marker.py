# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Debug-overlay spheres at the wrist targets a 34-dim ability-hand EEF action commands."""

from __future__ import annotations

import torch

# Alex ability-hand EEF action layout: [left_wrist_pose(7) | right_wrist_pose(7) | hand_joints(20)],
# each wrist pose = pos(3) + quat(4). Poses are in the env/world frame — the Pink IK action term
# converts world -> PELVIS_LINK internally (pink_task_space_actions.py::process_actions) — so the
# position columns can be read directly with no frame conversion.
EEF_ACTION_DIM = 34
LEFT_POS_SLICE = slice(0, 3)
RIGHT_POS_SLICE = slice(7, 10)


class ActionTargetMarker:
    """Renders a sphere at each env's left/right wrist target (blue/orange) each step."""

    def __init__(self, radius: float = 0.02):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/action_targets",
            markers={
                "left": sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 1.0)),
                ),
                "right": sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.3, 0.1)),
                ),
            },
        )
        self._markers = VisualizationMarkers(cfg)
        self._warned = False

    def update(self, actions: torch.Tensor) -> None:
        """Move the markers to the latest action's wrist targets.

        No-op (with a one-time warning) if ``actions`` isn't the 34-dim ability-hand EEF layout.
        """
        if actions.shape[-1] != EEF_ACTION_DIM:
            if not self._warned:
                print(
                    f"[ActionTargetMarker] action dim {actions.shape[-1]} != {EEF_ACTION_DIM};"
                    " not visualizing (only the ability-hand EEF action layout is supported)."
                )
                self._warned = True
            return
        num_envs = actions.shape[0]
        translations = torch.cat([actions[:, LEFT_POS_SLICE], actions[:, RIGHT_POS_SLICE]], dim=0)
        marker_indices = torch.cat(
            [
                torch.zeros(num_envs, dtype=torch.int64, device=actions.device),
                torch.ones(num_envs, dtype=torch.int64, device=actions.device),
            ]
        )
        self._markers.visualize(translations=translations, marker_indices=marker_indices)
