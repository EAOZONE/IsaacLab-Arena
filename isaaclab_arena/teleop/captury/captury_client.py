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

"""Network client for streaming skeleton poses from Captury Live."""

import logging
import numpy as np
import signal
import threading
import time

logger = logging.getLogger(__name__)


def _restore_default_fault_signal_handlers() -> None:
    """Undo IsaacLab's SIGSEGV/SIGABRT/SIGTERM handlers before going async.

    ``AppLauncher`` installs ``_abort_signal_handle_callback`` for SIGSEGV,
    SIGABRT and SIGTERM (app_launcher.py); that handler calls
    ``simulation_app.close()``, which itself imports modules. When the
    RemoteCaptury native receive thread raises one of these signals, the import
    inside ``close()`` re-enters the handler, recursing ``close()`` until the
    stack overflows -> segfault (or a silent app exit). Restoring the OS default
    handlers around the Captury connection breaks that recursion; a genuine
    fault then cores normally instead of hanging the whole teleop session.
    SIGINT is left untouched so Ctrl+C still stops teleop cleanly.
    """
    for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGTERM):
        try:
            if signal.getsignal(sig) not in (signal.SIG_DFL, signal.SIG_IGN, None):
                signal.signal(sig, signal.SIG_DFL)
        except (ValueError, OSError, RuntimeError) as e:
            # Not on the main thread, or platform restriction; best-effort only.
            logger.debug(f"Could not restore default handler for {sig}: {e}")


# Actor tracking modes reported by the RemoteCaptury actor-changed callback.
ACTOR_SCALING = 0
ACTOR_TRACKING = 1
ACTOR_STOPPED = 2
ACTOR_DELETED = 3
ACTOR_UNKNOWN = 4

CAPTURY_DEFAULT_PORT = 2101


class CapturyClient:
    """Streams skeleton poses from a Captury Live server.

    Wraps the ``remotecaptury`` Python bindings (built from
    https://github.com/thecaptury/RemoteCaptury, ``python/`` subdirectory).
    Pose callbacks arrive on the RemoteCaptury receive thread; the most
    recent pose per actor is kept in a small lock-protected buffer that the
    simulation loop polls via :meth:`get_latest_transforms`.

    Example:
        .. code-block:: python

            with CapturyClient(host="192.168.1.10") as client:
                while running:
                    transforms = client.get_latest_transforms()
                    if transforms is not None:
                        ...  # (N, 6) [tx, ty, tz, rx, ry, rz] per joint
    """

    def __init__(
        self,
        host: str,
        port: int = CAPTURY_DEFAULT_PORT,
        actor_id: int | None = None,
        stale_timeout_s: float = 0.5,
    ):
        """Initialize the client.

        Args:
            host: IP address or hostname of the Captury Live server.
            port: RemoteCaptury port on the server.
            actor_id: Captury actor to follow. When ``None``, the first actor
                that reports TRACKING status (or the first actor for which a
                pose arrives) is followed.
            stale_timeout_s: Poses older than this (wall clock, seconds) are
                treated as missing by :meth:`get_latest_transforms`.
        """
        self.host = host
        self.port = port
        self.actor_id = actor_id
        self.stale_timeout_s = stale_timeout_s

        self._remote = None
        self._lock = threading.Lock()
        self._selected_actor_id: int | None = actor_id
        self._latest_transforms: np.ndarray | None = None
        self._latest_receive_time: float = 0.0
        self._latest_timestamp_us: int = 0
        self._pose_count: int = 0
        self.joint_count: int | None = None
        """Number of joints in the streamed skeleton, detected on :meth:`start`."""
        self.joint_names: list[str] | None = None
        """Streamed skeleton's joint names, selected by :attr:`joint_count` on
        :meth:`start` (``None`` if the count has no built-in map)."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to the Captury Live server and start streaming poses.

        Raises:
            ImportError: If the ``remotecaptury`` package is not installed.
            ConnectionError: If the server cannot be reached.
        """
        try:
            from remotecaptury import RemoteCaptury
        except ImportError as e:
            raise ImportError(
                "The 'remotecaptury' package is required for Captury teleoperation. Build it inside the Arena"
                " container (the C++ sources must be staged first):\n"
                "  git clone https://github.com/thecaptury/RemoteCaptury.git\n"
                "  cd RemoteCaptury/python && bash prepare_build.sh\n"
                "  /isaac-sim/python.sh -m pip install ."
            ) from e

        # The native receive thread can trip IsaacLab's recursive abort-signal
        # handler and segfault the whole session; restore default handlers first.
        _restore_default_fault_signal_handlers()

        self._remote = RemoteCaptury()
        if not self._remote.connect(self.host, self.port):
            self._remote = None
            raise ConnectionError(f"Failed to connect to Captury Live at {self.host}:{self.port}")

        self._remote.register_actor_callback(self._on_actor)
        self._remote.register_pose_callback(self._on_pose)
        if not self._remote.start_streaming():
            self._remote.disconnect()
            self._remote = None
            raise ConnectionError(f"Connected to {self.host}:{self.port} but failed to start pose streaming")

        logger.info(f"CapturyClient streaming from {self.host}:{self.port}")

        # Identify the skeleton from the joint count of the first streamed pose,
        # so the joint map matches the transform order without a second
        # connection (a separate RemoteCaptury handle is unsafe inside Isaac Sim).
        self.joint_count = self.wait_for_skeleton()
        if self.joint_count is not None:
            from isaaclab_arena.teleop.captury.captury_skeleton import captury_joint_names_for_count

            self.joint_names = captury_joint_names_for_count(self.joint_count)
            logger.info(
                f"CapturyClient detected {self.joint_count}-joint skeleton"
                f"{'' if self.joint_names else ' (no built-in map; using default — set captury_joint_names if wrong)'}"
            )
        else:
            logger.info("CapturyClient: no pose yet at startup; skeleton will be inferred when tracking begins")

    def wait_for_skeleton(self, timeout_s: float = 3.0) -> int | None:
        """Block up to ``timeout_s`` for the first pose and return its joint count.

        Returns:
            The number of joints in the streamed skeleton, or ``None`` if no
            pose arrived (e.g. no actor is being tracked yet).
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._latest_transforms is not None:
                    return int(self._latest_transforms.shape[0])
            time.sleep(0.05)
        return None

    def stop(self) -> None:
        """Stop streaming from Captury Live.

        The native library's ``disconnect`` crashes when a Python pose callback
        is registered (a teardown bug in the ``remotecaptury`` wrapper), so it
        is deliberately not called — streaming is stopped and the handle is
        dropped, leaving the socket for OS cleanup at process exit.
        """
        if self._remote is not None:
            try:
                self._remote.stop_streaming()
            except Exception as e:
                logger.warning(f"Error while stopping Captury stream: {e}")
            self._remote = None
            logger.info("CapturyClient stopped")

    def __enter__(self) -> "CapturyClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Whether the client is connected to a Captury Live server."""
        return self._remote is not None

    @property
    def pose_count(self) -> int:
        """Number of poses received for the followed actor since start."""
        with self._lock:
            return self._pose_count

    def get_latest_transforms(self) -> np.ndarray | None:
        """Return the most recent skeleton pose for the followed actor.

        Returns:
            (N, 6) float64 array of [tx, ty, tz, rx, ry, rz] per joint
            (translation [mm], global XYZ Euler rotation [deg]), or ``None``
            when no pose has been received or the latest pose is older than
            ``stale_timeout_s``.
        """
        with self._lock:
            if self._latest_transforms is None:
                return None
            if time.monotonic() - self._latest_receive_time > self.stale_timeout_s:
                return None
            return self._latest_transforms

    # ------------------------------------------------------------------
    # RemoteCaptury callbacks (called on the receive thread)
    # ------------------------------------------------------------------

    def _on_actor(self, actor_id: int, mode: int) -> None:
        with self._lock:
            if self.actor_id is None and self._selected_actor_id is None and mode == ACTOR_TRACKING:
                self._selected_actor_id = actor_id
                logger.info(f"CapturyClient following actor {actor_id}")
            elif actor_id == self._selected_actor_id and mode in (ACTOR_STOPPED, ACTOR_DELETED):
                logger.warning(f"Captury actor {actor_id} stopped tracking")
                if self.actor_id is None:
                    # Auto-selected actor went away; allow re-selection.
                    self._selected_actor_id = None
                    self._latest_transforms = None

    def _on_pose(self, actor_id: int, pose: dict) -> None:
        with self._lock:
            if self._selected_actor_id is None:
                self._selected_actor_id = actor_id
                logger.info(f"CapturyClient following actor {actor_id}")
            elif actor_id != self._selected_actor_id:
                return
            transforms = pose.get("transforms", [])
            if not transforms:
                return
            array = np.empty((len(transforms), 6), dtype=np.float64)
            for i, transform in enumerate(transforms):
                array[i, 0:3] = transform["translation"]
                array[i, 3:6] = transform["rotation"]
            self._latest_transforms = array
            self._latest_timestamp_us = pose.get("timestamp", 0)
            self._latest_receive_time = time.monotonic()
            self._pose_count += 1
