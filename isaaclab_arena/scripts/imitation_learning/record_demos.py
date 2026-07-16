# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Device for interacting with environment. (default: keyboard)
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful. (default: 10)
    --timed_episode_s         If set, export each recording as success after this many seconds and auto-reset,
                              ignoring task success (useful for mocap batch collection with manual curation).
    --disable_full_sim_buffer_reset
                              Disable env.sim.reset() calls that fully clear sim context buffers before each episode. (default: False)
"""

"""Launch Isaac Sim Simulator first."""

# Standard library imports
import contextlib
import math

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)

# add argparse arguments
parser = get_isaaclab_arena_cli_parser()
parser.add_argument(
    "--dataset_file",
    type=str,
    required=True,
    help="File path to export recorded demos.",
)
parser.add_argument(
    "--step_hz", type=int, default=30, help="Environment stepping rate in Hz."
)
parser.add_argument(
    "--num_demos",
    type=int,
    default=1,
    help="Number of demonstrations to record. Set to 0 for infinite.",
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
parser.add_argument(
    "--min_episode_s",
    type=float,
    default=0.0,
    help=(
        "Minimum recording duration in seconds before a task-success episode can export. "
        "Use with --num_success_steps to avoid very short demos when success fires early."
    ),
)
parser.add_argument(
    "--timed_episode_s",
    type=float,
    default=None,
    help=(
        "Export each active recording as success after this many seconds, then auto-reset. "
        "Bypasses task success checks; use for batch mocap collection with manual curation."
    ),
)
parser.add_argument(
    "--disable_full_sim_buffer_reset",
    dest="disable_full_sim_buffer_reset",
    action="store_true",
    default=False,
    help="Disable calling env.sim.reset() to fully clear sim context buffers before each episode recording.",
)
parser.add_argument(
    "--head_view",
    action="store_true",
    default=False,
    help=(
        "Pin the Kit viewport camera at the robot's HEAD_LINK position (static, does not track the head) "
        "looking at the task object, with a widened field of view. Useful as a first-person operator view."
    ),
)
parser.add_argument(
    "--head_view_focal",
    type=float,
    default=12.0,
    help=(
        "Focal length [mm] of the viewport camera when --head_view is set. Lower = wider FOV "
        "(default 12.0; Kit default is ~18.15)."
    ),
)
parser.add_argument(
    "--print_lever_angle_hz",
    type=float,
    default=5.0,
    help="Print live lever Handle_1 angle at this rate for lever USD scenes. Set <= 0 to disable.",
)
parser.add_argument(
    "--start_recording_immediately",
    action="store_true",
    help=(
        "Start applying teleop actions immediately instead of waiting for the START callback. "
        "Useful for OpenXR collection when tracking is active but the robot appears frozen."
    ),
)
# Add the example environments CLI args
# NOTE(alexmillane, 2025.09.04): This has to be added last, because
# of the app specific flags being parsed after the global flags.
add_example_environments_cli_args(parser)

# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# TODO(cvolk): XR mode is inferred from teleop device name via string matching.
# Ideally, AppLauncher or the device config would auto-detect XR requirements.
if "openxr" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Patch isaaclab_physx's RTX render-update helper so the first frame's annotator
# buffers are populated before the teleop loop renders. Without this the initial
# pre-step render emerges black under the visualizer that record_demos.py uses.
from isaaclab_arena.utils.isaaclab_utils.isaac_rtx_renderer_patch import (
    patch_isaac_rtx_renderer,
)

patch_isaac_rtx_renderer()

"""Rest everything follows."""


# Third-party imports
import gymnasium as gym
import os
import time
import torch
from collections.abc import Callable

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

# Omniverse logger
import omni.log
import omni.ui as ui
from isaaclab.devices import (
    Se3Keyboard,
    Se3KeyboardCfg,
    Se3SpaceMouse,
    Se3SpaceMouseCfg,
)
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.envs.ui import EmptyWindow
from isaaclab.managers import DatasetExportMode
from isaaclab_mimic.ui.instruction_display import (
    InstructionDisplay,
    show_subtask_instructions,
)
from isaaclab_teleop import (
    IsaacTeleopCfg,
    create_isaac_teleop_device,
    remove_camera_configs,
)

from isaaclab_arena.teleop.captury.captury_teleop_device import (
    CapturyDeviceCfg,
    CapturyTeleopDevice,
    advance_captury_with_env_anchor,
    create_captury_teleop_device,
)
from isaaclab_arena.utils.cameras import clear_rtx_camera_output_buffers
from isaaclab_arena.utils.isaaclab_utils.manager_terms import (
    bind_extracted_manager_term,
)
from isaaclab_arena.utils.isaaclab_utils.recorders import (
    ArenaEnvRecorderManagerCfg,
    PreStepTestObsNewActionRecorderCfg,
    PreStepTestObsNewStateRecorderCfg,
)

# Imports have to follow simulation startup.


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def setup_output_directories() -> tuple[str, str]:
    """Set up output directories for saving demonstrations.

    Creates the output directory if it doesn't exist and extracts the file name
    from the dataset file path.

    Returns:
        tuple[str, str]: A tuple containing:
            - output_dir: The directory path where the dataset will be saved
            - output_file_name: The filename (without extension) for the dataset
    """
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir, output_file_name


def create_environment_config(
    output_dir: str, output_file_name: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, str, object | None, object | None]:
    """Create and configure the environment configuration.

    Parses the environment configuration and makes necessary adjustments for demo recording.
    Extracts the success termination function and configures the recorder manager.

    Args:
        output_dir: Directory where recorded demonstrations will be saved
        output_file_name: Name of the file to store the demonstrations

    Returns:
        tuple[isaaclab_tasks.utils.parse_cfg.EnvCfg, Optional[object]]: A tuple containing:
            - env_cfg: The configured environment configuration
            - success_term: The success termination object or None if not available
            - embodiment: The arena embodiment instance, if any

    Raises:
        Exception: If parsing the environment configuration fails
    """
    # parse configuration
    try:
        arena_builder = get_arena_builder_from_cli(args_cli)
        embodiment = arena_builder.arena_env.embodiment
        env_name, env_cfg = arena_builder.build_registered()

    except Exception as e:
        omni.log.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        omni.log.warn(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    if args_cli.xr:
        # If cameras are not enabled and XR is enabled, remove camera configs
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    if args_cli.enable_cameras:
        env_cfg.recorders = ArenaEnvRecorderManagerCfg()
        # Match visuomotor IL envs: render RTX sensors before reading observations on reset.
        env_cfg.num_rerenders_on_reset = 3
    else:
        env_cfg.recorders = ActionStateRecorderManagerCfg()
    if getattr(args_cli, "embodiment", None) == "alex_v2_ability_hands":
        env_cfg.recorders.record_pre_step_test_obs_new_state = (
            PreStepTestObsNewStateRecorderCfg()
        )
        env_cfg.recorders.record_pre_step_test_obs_new_action = (
            PreStepTestObsNewActionRecorderCfg()
        )
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, env_name, success_term, embodiment


def create_environment(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, env_name: str
) -> gym.Env:
    """Create the environment from the configuration.

    Args:
        env_cfg: The environment configuration object that defines the environment properties.
            This should be an instance of EnvCfg created by parse_env_cfg().

    Returns:
        gym.Env: A Gymnasium environment instance for the specified task.

    Raises:
        Exception: If environment creation fails for any reason.
    """
    try:
        env = gym.make(env_name, cfg=env_cfg)
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import (
            reapply_viewer_cfg,
        )

        reapply_viewer_cfg(env)
        return env.unwrapped
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        exit(1)


def _apply_teleop_elbow_targets(embodiment, teleop_interface, env) -> None:
    """Feed operator elbow directions into the IK solve, if both sides support it.

    No-op unless the device exposes ``get_elbow_directions_world`` (Captury) and
    the embodiment supports ``apply_teleop_elbow_targets`` (Alex ability hands
    with elbow tracking enabled).
    """
    if embodiment is None or not hasattr(embodiment, "apply_teleop_elbow_targets"):
        return
    if not hasattr(teleop_interface, "get_elbow_directions_world"):
        return
    embodiment.apply_teleop_elbow_targets(
        env, teleop_interface.get_elbow_directions_world()
    )


def setup_teleop_device(callbacks: dict[str, Callable]) -> object:
    """Set up the teleoperation device based on configuration.

    Attempts to create a teleoperation device based on the environment configuration.
    Falls back to default devices if the specified device is not found in the configuration.

    Args:
        callbacks: Dictionary mapping callback keys to functions that will be
                   attached to the teleop device

    Returns:
        object: The configured teleoperation device interface

    Raises:
        Exception: If teleop device creation fails
    """
    teleop_interface = None
    try:
        if hasattr(env_cfg, "isaac_teleop") and isinstance(
            env_cfg.isaac_teleop, CapturyDeviceCfg
        ):
            teleop_interface = create_captury_teleop_device(
                env_cfg.isaac_teleop,
                sim_device=env_cfg.sim.device,
                callbacks=callbacks,
            )
        elif hasattr(env_cfg, "isaac_teleop") and isinstance(
            env_cfg.isaac_teleop, IsaacTeleopCfg
        ):
            teleop_interface = create_isaac_teleop_device(
                env_cfg.isaac_teleop,
                sim_device=env_cfg.sim.device,
                callbacks=callbacks,
            )
        elif (
            hasattr(env_cfg, "teleop_devices")
            and args_cli.teleop_device in env_cfg.teleop_devices.devices
        ):
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks
            )
        else:
            omni.log.warn(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5)
                )
            else:
                omni.log.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                omni.log.error(
                    "Supported devices: keyboard, spacemouse, avp_handtracking"
                )
                exit(1)

            # Add callbacks to fallback device
            for key, callback in callbacks.items():
                teleop_interface.add_callback(key, callback)
    except Exception as e:
        omni.log.error(f"Failed to create teleop device: {e}")
        exit(1)

    if teleop_interface is None:
        omni.log.error("Failed to create teleop interface")
        exit(1)

    return teleop_interface


def setup_ui(label_text: str, env: gym.Env) -> InstructionDisplay:
    """Set up the user interface elements.

    Creates instruction display and UI window with labels for showing information
    to the user during demonstration recording.

    Args:
        label_text: Text to display showing current recording status
        env: The environment instance for which UI is being created

    Returns:
        InstructionDisplay: The configured instruction display object
    """
    instruction_display = InstructionDisplay(args_cli.xr)
    if not args_cli.xr:
        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)

    return instruction_display


def export_episode_as_success(env: gym.Env) -> None:
    """Mark the current episode successful and write it to the dataset file."""
    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    env.recorder_manager.set_success_to_episodes(
        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
    )
    env.recorder_manager.export_episodes([0])


def _find_lever_object_name(env: gym.Env) -> str | None:
    from isaaclab_arena_environments.lever_scene_builder import LEVER_USD_STEMS

    scene_keys = set(env.scene.keys())
    for stem in LEVER_USD_STEMS:
        key = stem.replace("(", "_").replace(")", "_")
        if key in scene_keys:
            return key
    return None


def _lever_handle_quat_xyzw(env: gym.Env, object_name: str) -> torch.Tensor:
    from isaacsim.core.prims import RigidPrim

    from isaaclab_arena_environments.lever_scene_builder import (
        LEVER_HANDLE_RIGID_BODY_SUFFIX,
    )

    prim_path = f"/World/envs/env_0/{object_name}{LEVER_HANDLE_RIGID_BODY_SUFFIX}"
    _, quat_wxyz = RigidPrim(prim_path).get_world_poses()
    return quat_wxyz[0][[1, 2, 3, 0]].to(device=env.device)


def _set_lever_success_rest_quat(
    env: gym.Env, object_name: str, rest_quat_xyzw: torch.Tensor
) -> None:
    """Update the reset-relative lever success cache used by the termination term."""
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    if not hasattr(base_env, "_lever_rest_quat_by_object"):
        base_env._lever_rest_quat_by_object = {}
    rest_quats = rest_quat_xyzw.detach().clone().reshape(1, 4).repeat(
        base_env.num_envs, 1
    )
    base_env._lever_rest_quat_by_object[object_name] = rest_quats.to(
        device=base_env.device
    )


def _quat_angle_deg(quat_xyzw: torch.Tensor, rest_quat_xyzw: torch.Tensor) -> float:
    dot = torch.abs(torch.dot(quat_xyzw, rest_quat_xyzw))
    dot = torch.clamp(dot, -1.0, 1.0)
    return math.degrees(float(2.0 * torch.acos(dot)))


def process_success_condition(
    env: gym.Env,
    success_term: object | None,
    success_step_count: int,
    episode_step_count: int,
) -> tuple[int, bool]:
    """Process the success condition for the current step.

    Checks if the environment has met the success condition for the required
    number of consecutive steps. Marks the episode as successful if criteria are met.

    Args:
        env: The environment instance to check
        success_term: The success termination object or None if not available
        success_step_count: Current count of consecutive successful steps

    Returns:
        tuple[int, bool]: A tuple containing:
            - updated success_step_count: The updated count of consecutive successful steps
            - success_reset_needed: Boolean indicating if reset is needed due to success
    """
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        min_episode_steps = math.ceil(args_cli.min_episode_s * args_cli.step_hz)
        if (
            success_step_count >= args_cli.num_success_steps
            and episode_step_count >= min_episode_steps
        ):
            export_episode_as_success(env)
            print("Success condition met! Recording completed.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


def reset_sim_context_buffers(env: gym.Env) -> None:
    """Reset the simulation context before an episode when enabled."""
    if args_cli.disable_full_sim_buffer_reset:
        return
    env.sim.reset()
    if args_cli.enable_cameras:
        clear_rtx_camera_output_buffers(env)


def handle_reset(
    env: gym.Env,
    success_step_count: int,
    instruction_display: InstructionDisplay,
    label_text: str,
) -> int:
    """Handle resetting the environment.

    Resets the environment, recorder manager, and related state variables.
    Updates the instruction display with current status.

    Args:
        env: The environment instance to reset
        success_step_count: Current count of consecutive successful steps
        instruction_display: The display object to update
        label_text: Text to display showing current recording status

    Returns:
        int: Reset success step count (0)
    """
    print("Resetting environment...")
    reset_sim_context_buffers(env)
    env.recorder_manager.reset()
    env.reset()
    success_step_count = 0
    instruction_display.show_demo(label_text)
    return success_step_count


def apply_head_view(env: gym.Env, focal_length_mm: float) -> None:
    """Pin the viewport camera at the robot head, static, with a widened FOV.

    Reads the HEAD_LINK world position from the ``robot`` articulation and points
    the Kit viewport camera at the task object. The camera is set once via
    ``sim.set_camera_view`` (origin stays in the world frame), so it does not
    track the head as the robot moves. The field of view is widened by lowering
    the persp camera prim's focal length.

    No-op if the robot or its HEAD_LINK body is unavailable.

    Args:
        env: The (unwrapped) environment instance.
        focal_length_mm: Focal length to set on the viewport camera. Lower values
            give a wider field of view.
    """
    import warp as wp

    if "robot" not in env.scene.articulations:
        omni.log.warn(
            "--head_view: no 'robot' articulation in scene; skipping head view."
        )
        return
    robot = env.scene["robot"]
    head_body_ids, _ = robot.find_bodies(["HEAD_LINK"])
    if not head_body_ids:
        omni.log.warn("--head_view: robot has no HEAD_LINK body; skipping head view.")
        return
    head_idx = int(head_body_ids[0])
    eye = wp.to_torch(robot.data.body_pos_w)[0, head_idx].cpu().tolist()

    # Aim at the microwave/task object if present, else look forward along world +x.
    target = None
    for key in ("microwave",):
        if key in env.scene.articulations or key in env.scene.rigid_objects:
            target = wp.to_torch(env.scene[key].data.root_pos_w)[0].cpu().tolist()
            break
    if target is None:
        target = [eye[0] + 1.0, eye[1], eye[2] - 0.2]

    env.sim.set_camera_view(eye=tuple(eye), target=tuple(target))

    # Widen FOV: focal length lives on the camera prim, not in ViewerCfg.
    import isaacsim.core.utils.prims as prim_utils

    cam_path = getattr(env.cfg.viewer, "cam_prim_path", "/OmniverseKit_Persp")
    cam_prim = prim_utils.get_prim_at_path(cam_path)
    focal_attr = cam_prim.GetAttribute("focalLength")
    if focal_attr.IsValid():
        focal_attr.Set(float(focal_length_mm))
        print(
            f"--head_view: viewport at HEAD_LINK {tuple(round(v, 3) for v in eye)}, focal {focal_length_mm} mm"
        )
    else:
        omni.log.warn(
            f"--head_view: no focalLength attribute on {cam_path}; FOV unchanged."
        )


def run_simulation_loop(
    env: gym.Env,
    teleop_interface: object | None,
    success_term: object | None,
    rate_limiter: RateLimiter | None,
    embodiment: object | None = None,
) -> int:
    """Run the main simulation loop for collecting demonstrations.

    Sets up callback functions for teleop device, initializes the UI,
    and runs the main loop that processes user inputs and environment steps.
    Records demonstrations when success conditions are met.

    Args:
        env: The environment instance
        teleop_interface: Optional teleop interface (will be created if None)
        success_term: The success termination object or None if not available
        rate_limiter: Optional rate limiter to control simulation speed

    Returns:
        int: Number of successful demonstrations recorded
    """
    current_recorded_demo_count = 0
    success_step_count = 0
    episode_step_count = 0
    should_reset_recording_instance = False
    running_recording_instance = (not args_cli.xr) or args_cli.start_recording_immediately
    episode_recording_start_time: float | None = None
    timed_episode_s = args_cli.timed_episode_s
    lever_object_name: str | None = None
    lever_rest_quat: torch.Tensor | None = None
    last_lever_angle_print_time = 0.0
    last_waiting_for_teleop_action_print_time = 0.0

    # Callback closures for the teleop device
    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        if success_step_count > 0:
            print(
                "Manual reset ignored. Success has fired and post-success steps are still recording. Please wait for"
                " the auto-reset."
            )
            return
        should_reset_recording_instance = True
        print("Recording instance reset requested")

    def start_recording_instance():
        nonlocal running_recording_instance, episode_recording_start_time, episode_step_count
        running_recording_instance = True
        episode_recording_start_time = time.time()
        episode_step_count = 0
        if embodiment is not None and hasattr(embodiment, "begin_teleop_action_warmup"):
            embodiment.begin_teleop_action_warmup()
        print("Recording started")

    def stop_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = False
        print("Recording paused")

    # Set up teleoperation callbacks
    teleoperation_callbacks = {
        "R": reset_recording_instance,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_recording_instance,
    }

    teleop_interface = setup_teleop_device(teleoperation_callbacks)
    teleop_interface.add_callback("R", reset_recording_instance)
    # Devices built on the IsaacTeleop pipeline stack (OpenXR, Captury) are
    # context managers and must be entered before advance().
    use_isaac_teleop = hasattr(teleop_interface, "__enter__") and hasattr(
        teleop_interface, "__exit__"
    )

    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
    instruction_display = setup_ui(label_text, env)

    def inner_loop():
        """Inner loop function with access to nonlocal variables."""
        nonlocal current_recorded_demo_count, success_step_count, should_reset_recording_instance
        nonlocal running_recording_instance, label_text, episode_recording_start_time
        nonlocal episode_step_count
        nonlocal lever_object_name, lever_rest_quat, last_lever_angle_print_time
        nonlocal last_waiting_for_teleop_action_print_time

        def maybe_finish_timed_episode() -> None:
            nonlocal should_reset_recording_instance
            if (
                timed_episode_s is not None
                and running_recording_instance
                and episode_recording_start_time is not None
                and time.time() - episode_recording_start_time >= timed_episode_s
            ):
                export_episode_as_success(env)
                print(
                    f"Timed episode complete ({timed_episode_s:g}s). Exporting and resetting."
                )
                should_reset_recording_instance = True

        def perform_episode_reset() -> None:
            nonlocal success_step_count, should_reset_recording_instance, episode_recording_start_time
            nonlocal episode_step_count
            nonlocal lever_rest_quat, last_lever_angle_print_time
            success_step_count = handle_reset(
                env, success_step_count, instruction_display, label_text
            )
            episode_step_count = 0
            teleop_interface.reset()
            if embodiment is not None and hasattr(
                embodiment, "reset_teleop_action_warmup"
            ):
                embodiment.reset_teleop_action_warmup()
            if lever_object_name is not None:
                lever_rest_quat = _lever_handle_quat_xyzw(env, lever_object_name)
                last_lever_angle_print_time = 0.0
            if running_recording_instance:
                episode_recording_start_time = time.time()
            else:
                episode_recording_start_time = None
            should_reset_recording_instance = False
            if lever_object_name is not None and lever_rest_quat is not None:
                _set_lever_success_rest_quat(env, lever_object_name, lever_rest_quat)

        # Reset before starting
        reset_sim_context_buffers(env)
        env.reset()
        teleop_interface.reset()
        lever_object_name = _find_lever_object_name(env)
        if lever_object_name is not None:
            lever_rest_quat = _lever_handle_quat_xyzw(env, lever_object_name)
            _set_lever_success_rest_quat(env, lever_object_name, lever_rest_quat)
        if lever_object_name is not None and args_cli.print_lever_angle_hz > 0:
            print(f"Lever angle debug enabled for scene object '{lever_object_name}'.")
        if running_recording_instance:
            episode_recording_start_time = time.time()
            episode_step_count = 0

        # Pin a static first-person viewport at the head once the robot pose is valid.
        if args_cli.head_view:
            apply_head_view(env, args_cli.head_view_focal)

        subtasks = {}
        stack_name = "IsaacTeleop" if use_isaac_teleop else "native"
        if timed_episode_s is not None:
            print(
                f"{stack_name} recording started (timed episodes: {timed_episode_s:g}s, forced success)."
            )
        else:
            print(f"{stack_name} recording started.")

        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running():
                # Get teleop command (may be None while waiting for session start)
                if isinstance(teleop_interface, CapturyTeleopDevice):
                    action = advance_captury_with_env_anchor(
                        teleop_interface, env, embodiment
                    )
                else:
                    action = teleop_interface.advance()
                if action is None:
                    now = time.time()
                    if now - last_waiting_for_teleop_action_print_time >= 2.0:
                        print(
                            "Waiting for IsaacTeleop action. In Kit/XR, click Start AR and send the START teleop command."
                        )
                        last_waiting_for_teleop_action_print_time = now
                    maybe_finish_timed_episode()
                    env.sim.render()
                    if should_reset_recording_instance:
                        perform_episode_reset()
                    continue
                if not torch.isfinite(action).all():
                    omni.log.warn(
                        "Skipping teleop step: non-finite action from IsaacTeleop."
                    )
                    env.sim.render()
                    continue
                if (
                    running_recording_instance
                    and embodiment is not None
                    and hasattr(embodiment, "stabilize_teleop_action")
                ):
                    action = embodiment.stabilize_teleop_action(env, action)
                if running_recording_instance:
                    _apply_teleop_elbow_targets(embodiment, teleop_interface, env)
                # Expand to batch dimension
                actions = action.repeat(env.num_envs, 1)

                # Perform action on environment
                if running_recording_instance:
                    # Compute actions based on environment
                    obv = env.step(actions)
                    episode_step_count += 1
                    if (
                        lever_object_name is not None
                        and lever_rest_quat is not None
                        and args_cli.print_lever_angle_hz > 0
                    ):
                        now = time.time()
                        if (
                            now - last_lever_angle_print_time
                            >= 1.0 / args_cli.print_lever_angle_hz
                        ):
                            angle_deg = _quat_angle_deg(
                                _lever_handle_quat_xyzw(env, lever_object_name),
                                lever_rest_quat,
                            )
                            print(f"lever_angle_deg={angle_deg:.2f}")
                            last_lever_angle_print_time = now
                    if subtasks is not None:
                        if subtasks == {}:
                            subtasks = obv[0].get("subtask_terms")
                        elif subtasks:
                            show_subtask_instructions(
                                instruction_display, subtasks, obv, env.cfg
                            )
                else:
                    env.sim.render()

                # Timed episodes: export as success and reset after the configured duration.
                maybe_finish_timed_episode()
                if timed_episode_s is None:
                    # Check for task success condition
                    success_step_count_new, success_reset_needed = (
                        process_success_condition(
                            env, success_term, success_step_count, episode_step_count
                        )
                    )
                    success_step_count = success_step_count_new
                    if success_reset_needed:
                        should_reset_recording_instance = True

                # Update demo count if it has changed
                if (
                    env.recorder_manager.exported_successful_episode_count
                    > current_recorded_demo_count
                ):
                    current_recorded_demo_count = (
                        env.recorder_manager.exported_successful_episode_count
                    )
                    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                    print(label_text)

                # Check if we've reached the desired number of demos
                if (
                    args_cli.num_demos > 0
                    and env.recorder_manager.exported_successful_episode_count
                    >= args_cli.num_demos
                ):
                    label_text = f"All {current_recorded_demo_count} demonstrations recorded.\nExiting the app."
                    instruction_display.show_demo(label_text)
                    print(label_text)
                    target_time = time.time() + 0.8
                    while time.time() < target_time:
                        if rate_limiter:
                            rate_limiter.sleep(env)
                        else:
                            env.sim.render()
                    break

                # Handle reset if requested
                if should_reset_recording_instance:
                    perform_episode_reset()

                # Check if simulation is stopped
                if env.sim.is_stopped():
                    break

                # Rate limiting
                if rate_limiter:
                    rate_limiter.sleep(env)

    # Run the loop with or without context manager based on stack
    if use_isaac_teleop:
        with teleop_interface:
            inner_loop()
    else:
        inner_loop()

    return current_recorded_demo_count


def main() -> None:
    """Collect demonstrations from the environment using teleop interfaces.

    Main function that orchestrates the entire process:
    1. Sets up rate limiting based on configuration
    2. Creates output directories for saving demonstrations
    3. Configures the environment
    4. Runs the simulation loop to collect demonstrations
    5. Cleans up resources when done

    Raises:
        Exception: Propagates exceptions from any of the called functions
    """
    # if handtracking is selected, rate limiting is achieved via OpenXR
    if args_cli.xr:
        rate_limiter = None
        from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization

        # Assign the teleop visualization manager to the visualization system
        XRVisualization.assign_manager(TeleopVisualizationManager)
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    # Set up output directories
    output_dir, output_file_name = setup_output_directories()

    # Create and configure environment
    global env_cfg  # Make env_cfg available to setup_teleop_device
    env_cfg, env_name, success_term, embodiment = create_environment_config(
        output_dir, output_file_name
    )

    # Create environment
    env = create_environment(env_cfg, env_name)
    success_term = bind_extracted_manager_term(success_term, env)

    # Run simulation loop
    current_recorded_demo_count = run_simulation_loop(
        env, None, success_term, rate_limiter, embodiment
    )

    # Clean up
    env.close()
    print(
        f"Recording session completed with {current_recorded_demo_count} successful demonstrations"
    )
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
