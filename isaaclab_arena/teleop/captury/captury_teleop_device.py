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

"""Isaac Lab teleop device driven by a Captury Live mocap stream."""

from __future__ import annotations

import logging
import numpy as np
import torch
from collections.abc import Callable
from scipy.spatial.transform import Rotation

from isaaclab.utils import configclass
from isaaclab_teleop import IsaacTeleopCfg

logger = logging.getLogger(__name__)

# Rotation from the teleop anchor convention (Y-up, +X right, +Z back — same
# as OpenXR) to the Isaac Lab world convention (Z-up). Matches
# ``XrAnchorManager._OXR_TO_USD_ROTATION`` so Captury-driven pipelines see the
# same anchor frame as OpenXR-driven ones.
_ANCHOR_TO_USD_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@configclass
class CapturyDeviceCfg(IsaacTeleopCfg):
    """Configuration for Captury Live mocap teleoperation.

    Extends :class:`~isaaclab_teleop.IsaacTeleopCfg` so it can be stored in
    the same ``env_cfg.isaac_teleop`` slot; scripts must check for this
    subclass *before* the ``IsaacTeleopCfg`` branch and construct a
    :class:`CapturyTeleopDevice` instead of the OpenXR-based device.

    Note:
        Unlike the OpenXR device, :attr:`pipeline_builder` here is called with
        a single argument — the Captury hands source node — and must return an
        ``OutputCombiner`` with an ``"action"`` output:

        .. code-block:: python

            def pipeline_builder(hands_source):
                ...
    """

    captury_host: str = "127.0.0.1"
    """IP address or hostname of the Captury Live server."""

    captury_port: int = 2101
    """RemoteCaptury streaming port on the Captury Live server."""

    captury_actor_id: int | None = None
    """Captury actor to follow; ``None`` follows the first tracked actor."""

    captury_stale_timeout_s: float = 0.5
    """Poses older than this [s] are treated as tracking loss."""

    captury_joint_names: list[str] | None = None
    """Joint names of the streamed Captury skeleton, in streaming order.

    ``None`` assumes the standard Captury Live skeleton. Provide the actual
    joint list (visible in Captury Live) when using a skeleton with fingers
    or a non-default joint order.
    """

    captury_anchor_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation [m] re-basing the Captury world origin in the anchor frame."""

    captury_anchor_yaw_deg: float = 0.0
    """Rotation about the Captury up axis [deg] applied before the translation."""

    captury_wrist_rotation_offset_xyzw: tuple[float, float, float, float] | None = None
    """Fixed wrist rotation offset (XYZW quaternion), composed on the right of
    the wrist frame.

    Calibrates out a constant rotation between the Captury wrist frame and the
    frame the downstream Se3 retargeter expects. Applies to both the
    geometrically synthesized frame (finger-tracked skeletons) and the raw
    Captury wrist bone frame (finger-less skeletons). Measure the residual
    wrist rotation in the viewport and set it here; ``None`` is identity.
    """

    captury_euler_degrees: bool = True
    """Whether streamed Captury Euler angles are in degrees."""

    captury_visualize_skeleton: bool = True
    """Overlay the tracked Captury skeleton as debug markers in the viewport.

    Joints render as small spheres in the simulation world frame (the same
    frame the hands drive), with the retargeted joints (wrists, elbows,
    shoulders, fingers) highlighted. Useful for checking tracking and
    anchor alignment; set ``False`` to disable.
    """

    captury_skeleton_marker_radius: float = 0.015
    """Radius [m] of the skeleton debug-overlay joint markers."""

    captury_torso_tracking: bool = True
    """Map the operator torso onto the robot torso each step.

    When enabled (default), streamed poses are expressed relative to the
    operator's torso joint and ``world_T_anchor`` follows
    :attr:`~isaaclab_teleop.XrCfg.anchor_prim_path` (e.g. Alex ``TORSO_LINK``)
    so arm motion is relative to the robot instead of the Captury studio origin.
    """


class CapturyTeleopDevice:
    """Teleop device that drives a retargeting pipeline from Captury mocap.

    Mirrors the public interface of
    :class:`~isaaclab_teleop.IsaacTeleopDevice` (context manager,
    :meth:`advance`, :meth:`add_callback`, :meth:`reset`) but requires no
    OpenXR session: the isaacteleop retargeting graph is executed directly
    with the Captury hands source as its only data source.

    Keyboard commands (when running with a Kit app window):
        * ``R`` — fire the ``"R"``/``"RESET"`` callbacks (reset environment).
        * ``Enter`` — fire the ``"START"`` callback.
        * ``Backspace`` — fire the ``"STOP"`` callback.

    Example:
        .. code-block:: python

            cfg = CapturyDeviceCfg(
                captury_host="192.168.1.10",
                pipeline_builder=my_builder,  # takes the hands source node
                sim_device="cuda:0",
            )
            with CapturyTeleopDevice(cfg) as device:
                while running:
                    action = device.advance()
                    if action is not None:
                        env.step(action.repeat(num_envs, 1))
    """

    WORLD_T_ANCHOR_INPUT_NAME = "world_T_anchor"
    """Well-known name of the ValueInput leaf receiving the anchor transform."""

    def __init__(self, cfg: CapturyDeviceCfg):
        """Initialize the device.

        Args:
            cfg: Captury teleoperation configuration.
        """
        self._cfg = cfg
        self._device = torch.device(cfg.sim_device)
        self._client = None
        self._pipeline = None
        self._external_leaf_names: set[str] = set()
        self._callbacks: dict[str, Callable] = {}
        self._waiting_logged = False
        self._keyboard_sub = None
        self._skeleton_map = None
        self._anchor_T_captury = None
        self._elbow_directions_world: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self._arm_tracking_hints: dict[str, object | None] = {"left": None, "right": None}
        self._skeleton_markers = None
        self._skeleton_marker_indices: np.ndarray | None = None
        self._skeleton_key_joints: set[int] = set()
        self._anchor_prim_initial_height: float | None = None
        self._anchor_prim_initial_yaw: float | None = None
        self._anchor_torso_world_override: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> CapturyTeleopDevice:
        """Connect to Captury Live and build the retargeting pipeline."""
        from isaaclab_arena.teleop.captury.captury_client import CapturyClient
        from isaaclab_arena.teleop.captury.captury_hands_source import CapturyHandsSource, make_anchor_transform
        from isaaclab_arena.teleop.captury.captury_skeleton import (
            build_skeleton_map_from_joint_names,
            default_captury_skeleton_map,
        )

        assert self._cfg.pipeline_builder is not None, "CapturyDeviceCfg requires a pipeline_builder"

        self._client = CapturyClient(
            host=self._cfg.captury_host,
            port=self._cfg.captury_port,
            actor_id=self._cfg.captury_actor_id,
            stale_timeout_s=self._cfg.captury_stale_timeout_s,
        )
        self._client.start()

        # Joint order resolution: explicit config > auto-detected from the live
        # skeleton > the finger-less default.
        joint_names = self._cfg.captury_joint_names or self._client.joint_names
        if joint_names is not None:
            skeleton_map = build_skeleton_map_from_joint_names(joint_names)
        else:
            skeleton_map = default_captury_skeleton_map()
        self._skeleton_map = skeleton_map
        self._anchor_T_captury = make_anchor_transform(
            translation_m=self._cfg.captury_anchor_translation,
            yaw_deg=self._cfg.captury_anchor_yaw_deg,
        )

        hands_source = CapturyHandsSource(
            name="captury_hands",
            pose_provider=self._client,
            skeleton_map=skeleton_map,
            anchor_T_captury=self._anchor_T_captury,
            wrist_rotation_offset_xyzw=self._cfg.captury_wrist_rotation_offset_xyzw,
            euler_degrees=self._cfg.captury_euler_degrees,
            torso_relative=self._cfg.captury_torso_tracking,
        )

        self._pipeline = self._cfg.pipeline_builder(hands_source)
        self._external_leaf_names = {
            node.name for node in self._pipeline.get_leaf_nodes() if node.input_spec() and node is not hands_source
        }
        self._setup_keyboard()
        if self._cfg.captury_visualize_skeleton:
            self._setup_skeleton_visualization(skeleton_map)
        logger.info(f"Captury teleop session started ({self._cfg.captury_host}:{self._cfg.captury_port})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Disconnect from Captury Live and release the pipeline."""
        if self._keyboard_sub is not None:
            self._teardown_keyboard()
        self._skeleton_markers = None
        if self._client is not None:
            self._client.stop()
            self._client = None
        self._pipeline = None
        logger.info("Captury teleop session ended")
        return False

    def __str__(self) -> str:
        msg = f"Captury Teleop Device: {self.__class__.__name__}\n"
        msg += f"\tServer: {self._cfg.captury_host}:{self._cfg.captury_port}\n"
        msg += f"\tActor: {self._cfg.captury_actor_id if self._cfg.captury_actor_id is not None else 'auto'}\n"
        msg += f"\tAnchor Position: {self._cfg.xr_cfg.anchor_pos}\n"
        msg += f"\tAnchor Rotation: {self._cfg.xr_cfg.anchor_rot}\n"
        msg += f"\tSim Device: {self._cfg.sim_device}\n"
        msg += "\tKeyboard: R = reset, Enter = start, Backspace = stop\n"
        return msg

    # ------------------------------------------------------------------
    # Device interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset hook for device-interface compatibility.

        Clears cached torso anchor height/yaw so the next :meth:`advance` after
        an environment reset re-reads the robot spawn pose (important when the
        anchor prim was briefly invalid or stale on the previous episode).
        """
        self._anchor_prim_initial_height = None
        self._anchor_prim_initial_yaw = None
        self._anchor_torso_world_override = None

    def add_callback(self, key: str, func: Callable) -> None:
        """Add a callback function for teleop commands.

        Args:
            key: The command type to bind to: ``"START"``, ``"STOP"``,
                ``"RESET"``, or ``"R"``.
            func: The function to call when the command is received.
        """
        self._callbacks[key] = func

    def advance(
        self,
        target_T_world: np.ndarray | torch.Tensor | None = None,
        anchor_torso_world: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Execute one retargeting step and return the action tensor.

        Args:
            target_T_world: Optional (4, 4) transform re-basing output poses
                into a target frame. When ``None`` and
                :attr:`~isaaclab_teleop.IsaacTeleopCfg.target_frame_prim_path`
                is set, the transform is read from the prim automatically.
            anchor_torso_world: Optional (4, 4) world transform of the robot
                torso link. When provided (recommended for Alex), this overrides
                the USD prim read used for ``world_T_anchor`` placement.

        Returns:
            Flattened action tensor on the configured device, or ``None``
            while no Captury pose has been received yet (or tracking is
            stale).

        Raises:
            RuntimeError: If called outside of a context manager.
        """
        if self._pipeline is None:
            raise RuntimeError("CapturyTeleopDevice must be entered (with-statement) before advance()")

        if self._client.get_latest_transforms() is None:
            if not self._waiting_logged:
                logger.info("Waiting for Captury pose data (is the actor being tracked?)")
                self._waiting_logged = True
            return None
        self._waiting_logged = False

        if target_T_world is None and self._cfg.target_frame_prim_path is not None:
            target_T_world = self._get_prim_world_inverse(self._cfg.target_frame_prim_path)

        if anchor_torso_world is not None and isinstance(anchor_torso_world, torch.Tensor):
            anchor_torso_world = anchor_torso_world.detach().cpu().numpy()
        self._anchor_torso_world_override = (
            np.asarray(anchor_torso_world, dtype=np.float64) if anchor_torso_world is not None else None
        )
        try:
            external_inputs = self._build_external_inputs(target_T_world)
        # Let the pipeline auto-build its per-step ComputeContext (its shape
        # varies across isaacteleop versions). Run-state/reset signalling is not
        # needed here: SE3 wrist + dex-hand retargeting do not depend on it, and
        # environment resets are driven separately by the teleop loop.
            result = self._pipeline.execute_pipeline(external_inputs)

            # Joint poses in the simulation world frame, reused for elbow tracking
            # and the optional skeleton overlay.
            world_matrices = self._world_joint_matrices()
            self._update_elbow_directions_world(world_matrices)
            if self._skeleton_markers is not None:
                self._update_skeleton_visualization(world_matrices)

            action_array = result["action"][0]
            return torch.from_dlpack(action_array).to(dtype=torch.float32, device=self._device)
        finally:
            self._anchor_torso_world_override = None

    def get_elbow_directions_world(self) -> dict[str, np.ndarray | None]:
        """Per-arm shoulder->elbow unit directions in the simulation world frame.

        Computed from the most recent :meth:`advance` call. Used for
        teleop-only elbow tracking: the consumer places the robot elbow at the
        robot's own upper-arm length along this direction. Values are ``None``
        for an arm whose shoulder/elbow joints are not tracked.

        Prefer :meth:`get_arm_tracking_hints_world` for full flexion + swivel hints.

        Returns:
            ``{"left": dir | None, "right": dir | None}`` with (3,) float64
            unit vectors in the world frame.
        """
        return self._elbow_directions_world

    def get_arm_tracking_hints_world(self) -> dict[str, object | None]:
        """Per-arm elbow IK hints (position, orientation, swivel) in world frame.

        Returns:
            ``{"left": CapturyArmTrackingHints | None, "right": ...}`` from the
            most recent :meth:`advance` call.
        """
        return self._arm_tracking_hints

    def _world_joint_matrices(self) -> np.ndarray | None:
        """Latest Captury joint poses as (N, 4, 4) transforms in the world frame.

        Uses the same anchor / world transform chain as the wrist poses
        (``world_T_anchor @ anchor_T_captury``) so the skeleton lines up with
        where the hands drive the robot. No ``target_T_world`` rebase is applied.

        Returns:
            (N, 4, 4) float64 world-frame joint transforms, or ``None`` when no
            pose is available.
        """
        from isaaclab_arena.teleop.captury.captury_skeleton import prepare_captury_joint_matrices

        transforms = self._client.get_latest_transforms()
        if transforms is None:
            return None
        matrices = prepare_captury_joint_matrices(
            transforms,
            self._skeleton_map,
            anchor_T_captury=self._anchor_T_captury,
            euler_degrees=self._cfg.captury_euler_degrees,
            torso_relative=self._cfg.captury_torso_tracking,
        )
        world_T_anchor = self._get_world_T_anchor().astype(np.float64)
        return world_T_anchor @ matrices

    def _update_elbow_directions_world(self, world_matrices: np.ndarray | None) -> None:
        """Recompute world-frame arm tracking hints from joint poses."""
        from isaaclab_arena.teleop.captury.captury_skeleton import (
            captury_arm_tracking_hints,
            captury_upper_arm_directions,
        )

        if world_matrices is None or self._skeleton_map is None:
            self._elbow_directions_world = {"left": None, "right": None}
            self._arm_tracking_hints = {"left": None, "right": None}
            return
        self._arm_tracking_hints = captury_arm_tracking_hints(world_matrices, self._skeleton_map)
        self._elbow_directions_world = captury_upper_arm_directions(world_matrices, self._skeleton_map)

    def _setup_skeleton_visualization(self, skeleton_map) -> None:
        """Create the debug-overlay markers for the tracked Captury skeleton.

        Two sphere prototypes are used: a small marker for body joints and a
        larger highlighted marker for the retargeted joints (wrists, elbows,
        shoulders, fingers). Best-effort — a failure here does not stop teleop.
        """
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            radius = self._cfg.captury_skeleton_marker_radius
            cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/CapturySkeleton",
                markers={
                    "joint": sim_utils.SphereCfg(
                        radius=radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
                    ),
                    "key": sim_utils.SphereCfg(
                        radius=radius * 1.6,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.1)),
                    ),
                },
            )
            self._skeleton_markers = VisualizationMarkers(cfg)

            # Joints that participate in retargeting get the highlighted prototype.
            key = {skeleton_map.left.wrist, skeleton_map.right.wrist}
            if skeleton_map.torso is not None:
                key.add(skeleton_map.torso)
            for arm in (skeleton_map.left, skeleton_map.right):
                key.update(v for v in (arm.shoulder, arm.elbow) if v is not None)
                key.update(arm.fingers.values())
            self._skeleton_key_joints = key
            self._skeleton_marker_indices = None
        except Exception as e:
            logger.warning(f"Could not create Captury skeleton overlay: {e}")
            self._skeleton_markers = None

    def _update_skeleton_visualization(self, world_matrices: np.ndarray | None) -> None:
        """Update the debug skeleton overlay with the latest joint positions."""
        if self._skeleton_markers is None or world_matrices is None:
            return
        num_joints = world_matrices.shape[0]
        if self._skeleton_marker_indices is None or len(self._skeleton_marker_indices) != num_joints:
            self._skeleton_marker_indices = np.array(
                [1 if i in self._skeleton_key_joints else 0 for i in range(num_joints)], dtype=np.int64
            )
        positions = torch.from_numpy(np.ascontiguousarray(world_matrices[:, :3, 3], dtype=np.float32))
        self._skeleton_markers.visualize(translations=positions, marker_indices=self._skeleton_marker_indices)

    # ------------------------------------------------------------------
    # Pipeline inputs
    # ------------------------------------------------------------------

    def _build_external_inputs(self, target_T_world: np.ndarray | torch.Tensor | None) -> dict:
        """Build inputs for the pipeline's external leaf nodes.

        Currently only the well-known ``world_T_anchor`` ValueInput is
        recognized, mirroring ``TeleopSessionLifecycle``.
        """
        from isaacteleop.retargeting_engine.interface import TensorGroup, ValueInput
        from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

        external_inputs: dict = {}
        for leaf_name in self._external_leaf_names:
            if leaf_name == self.WORLD_T_ANCHOR_INPUT_NAME:
                anchor_matrix = self._get_world_T_anchor()
                if target_T_world is not None:
                    if isinstance(target_T_world, torch.Tensor):
                        target_T_world = target_T_world.detach().cpu().numpy()
                    anchor_matrix = np.asarray(target_T_world, dtype=np.float32) @ anchor_matrix
                xform_tg = TensorGroup(TransformMatrix())
                xform_tg[0] = anchor_matrix.astype(np.float32)
                external_inputs[leaf_name] = {ValueInput.VALUE: xform_tg}
            else:
                logger.warning(
                    f"Unrecognized external leaf node '{leaf_name}' in pipeline; "
                    "CapturyTeleopDevice does not know how to provide its inputs"
                )
        return external_inputs

    def _get_world_T_anchor(self) -> np.ndarray:
        """Anchor-to-world matrix for placing operator poses in the simulation.

        When torso tracking is enabled and :attr:`~isaaclab_teleop.XrCfg.anchor_prim_path`
        is set, the anchor follows the robot torso prim each step.  The OpenXR
        ``anchor_pos`` offset (typically ``(0, 0, -1)`` for headset placement) is
        **not** applied in that mode: Captury poses are already torso-relative, so
        the operator torso origin maps directly onto the robot torso link.  Use
        :attr:`captury_anchor_translation` for fine calibration in Captury space.

        Otherwise falls back to the static :attr:`~isaaclab_teleop.XrCfg.anchor_pos` /
        :attr:`~isaaclab_teleop.XrCfg.anchor_rot` configuration.
        """
        xr_cfg = self._cfg.xr_cfg
        anchor_pos = np.array([float(p) for p in xr_cfg.anchor_pos], dtype=np.float64)
        # Yaw the static anchor by the robot's spawn heading so the operator frame
        # follows the robot's facing. Captured once from the anchor prim (the robot
        # is fixed-base during teleop, so its base yaw is static) and cached — this
        # sidesteps the zero-norm quaternions that per-step FOLLOW_PRIM reads of a
        # physics-driven link can return. Zero yaw (e.g. the microwave scene) leaves
        # the anchor unchanged; a turned robot (e.g. +90 deg in the fridge kitchen)
        # rotates the operator/skeleton frame to match.
        anchor_yaw = 0.0

        if self._cfg.captury_torso_tracking:
            anchor_pos = np.zeros(3, dtype=np.float64)
            robot_torso = self._anchor_torso_world_override
            if robot_torso is None and xr_cfg.anchor_prim_path is not None:
                robot_torso = self._get_prim_world_matrix(xr_cfg.anchor_prim_path)
            if robot_torso is not None:
                pos = robot_torso[:3, 3].copy()
                if xr_cfg.fixed_anchor_height:
                    if self._anchor_prim_initial_height is None:
                        self._anchor_prim_initial_height = float(pos[2])
                    pos[2] = self._anchor_prim_initial_height
                anchor_pos = pos
                if self._anchor_prim_initial_yaw is None:
                    rot = robot_torso[:3, :3]
                    self._anchor_prim_initial_yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
                anchor_yaw = self._anchor_prim_initial_yaw
            elif xr_cfg.anchor_prim_path is not None:
                logger.warning(
                    "Captury torso tracking: no torso world transform (prim '%s' unreadable). "
                    "Pass anchor_torso_world from the embodiment articulation each advance(), "
                    "e.g. embodiment.get_captury_anchor_torso_world(env). "
                    "Hand targets fall back to the world origin with zero yaw.",
                    xr_cfg.anchor_prim_path,
                )

        r_anchor = Rotation.from_quat([float(q) for q in xr_cfg.anchor_rot]).as_matrix()
        yaw_rot = Rotation.from_euler("z", anchor_yaw).as_matrix()
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = yaw_rot @ r_anchor @ _ANCHOR_TO_USD_ROTATION
        mat[:3, 3] = anchor_pos.astype(np.float32)
        return mat

    @staticmethod
    def _get_prim_world_matrix(prim_path: str) -> np.ndarray | None:
        """Read a prim's world transform from Fabric as a (4, 4) matrix."""
        try:
            import omni.usd
            import usdrt
            from pxr import UsdUtils
            from usdrt import Rt

            stage = omni.usd.get_context().get_stage()
            stage_cache = UsdUtils.StageCache.Get()
            stage_id = stage_cache.GetId(stage).ToLongInt()
            if stage_id < 0:
                stage_id = stage_cache.Insert(stage).ToLongInt()
            rt_stage = usdrt.Usd.Stage.Attach(stage_id)
            if rt_stage is None:
                return None

            rt_prim = rt_stage.GetPrimAtPath(prim_path)
            if not rt_prim.IsValid():
                return None
            rt_xformable = Rt.Xformable(rt_prim)
            if not rt_xformable.GetPrim().IsValid():
                return None
            world_matrix_attr = rt_xformable.GetFabricHierarchyWorldMatrixAttr()
            if world_matrix_attr is None:
                return None
            rt_matrix = world_matrix_attr.Get()
            if rt_matrix is None:
                return None

            pos = rt_matrix.ExtractTranslation()
            rt_quat = rt_matrix.ExtractRotationQuat()
            quat_xyzw = [
                float(rt_quat.GetImaginary()[0]),
                float(rt_quat.GetImaginary()[1]),
                float(rt_quat.GetImaginary()[2]),
                float(rt_quat.GetReal()),
            ]
            R = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float64)
            t = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float64)

            mat = np.eye(4, dtype=np.float64)
            mat[:3, :3] = R
            mat[:3, 3] = t
            return mat
        except Exception as e:
            logger.warning(f"Failed to read anchor prim '{prim_path}': {e}")
            return None

    @staticmethod
    def _get_prim_world_inverse(prim_path: str) -> np.ndarray | None:
        """Read a prim's world transform from Fabric and return its inverse.

        Same approach as ``IsaacTeleopDevice._get_target_frame_T_world``.
        """
        world_mat = CapturyTeleopDevice._get_prim_world_matrix(prim_path)
        if world_mat is None:
            return None
        R = world_mat[:3, :3].astype(np.float32)
        t = world_mat[:3, 3].astype(np.float32)
        inv_mat = np.eye(4, dtype=np.float32)
        inv_mat[:3, :3] = R.T
        inv_mat[:3, 3] = -(R.T @ t)
        return inv_mat

    # ------------------------------------------------------------------
    # Keyboard commands
    # ------------------------------------------------------------------

    def _setup_keyboard(self) -> None:
        """Subscribe to Kit keyboard input for reset/start/stop commands."""
        try:
            import carb.input
            import omni.appwindow

            app_window = omni.appwindow.get_default_app_window()
            keyboard = app_window.get_keyboard()
            input_iface = carb.input.acquire_input_interface()
            self._keyboard_sub = input_iface.subscribe_to_keyboard_events(keyboard, self._on_keyboard_event)
            self._keyboard = keyboard
            self._input_iface = input_iface
        except (ImportError, ModuleNotFoundError, AttributeError):
            logger.info("Kit keyboard input not available; Captury teleop keyboard commands disabled")
            self._keyboard_sub = None

    def _teardown_keyboard(self) -> None:
        try:
            self._input_iface.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        except Exception as e:
            logger.debug(f"Suppressed error unsubscribing keyboard events: {e}")
        self._keyboard_sub = None

    def _on_keyboard_event(self, event) -> bool:
        import carb.input

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.R:
                self._fire_callback("R")
                self._fire_callback("RESET")
            elif event.input == carb.input.KeyboardInput.ENTER:
                self._fire_callback("START")
            elif event.input == carb.input.KeyboardInput.BACKSPACE:
                self._fire_callback("STOP")
        return True

    def _fire_callback(self, key: str) -> None:
        func = self._callbacks.get(key)
        if func is not None:
            func()


def advance_captury_with_env_anchor(
    teleop_interface: CapturyTeleopDevice,
    env: object,
    embodiment: object | None = None,
) -> torch.Tensor | None:
    """Run one Captury step with Alex torso anchoring from articulation state."""
    anchor_torso_world = None
    if embodiment is not None and hasattr(embodiment, "get_captury_anchor_torso_world"):
        anchor_torso_world = embodiment.get_captury_anchor_torso_world(env)
    return teleop_interface.advance(anchor_torso_world=anchor_torso_world)


def create_captury_teleop_device(
    cfg: CapturyDeviceCfg,
    sim_device: str | None = None,
    callbacks: dict[str, Callable] | None = None,
) -> CapturyTeleopDevice:
    """Create a :class:`CapturyTeleopDevice`, mirroring ``create_isaac_teleop_device``.

    Args:
        cfg: Captury teleoperation configuration.
        sim_device: If provided, overrides ``cfg.sim_device`` so action
            tensors land on the simulation device.
        callbacks: Optional mapping of command keys (``"START"``, ``"STOP"``,
            ``"RESET"``, ``"R"``) to callables registered on the device.

    Returns:
        A configured device ready for use in a ``with`` block.
    """
    if sim_device is not None:
        cfg.sim_device = sim_device

    logger.info("Using Captury mocap stack for teleoperation")
    device = CapturyTeleopDevice(cfg)

    if callbacks is not None:
        for key, func in callbacks.items():
            device.add_callback(key, func)

    return device
