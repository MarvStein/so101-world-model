"""Robot controller for SO101 closed-loop deployment.

Connects to the SO101 robot arm and runs the closed-loop control loop.
Model inference is delegated to model_server.py via a ZeroMQ IPC socket.

This process runs in the **lerobot environment** (numpy >= 2.0).
The model server process (numpy == 1.26.4) must be started first; eval.sh
handles this automatically.

Control loop
------------
1. Capture a camera frame and joint positions from the SO101.
2. Assemble rolling image history and current state into numpy arrays.
3. Send them to the model server over ZMQ and wait for the action chunk.
4. Execute the returned actions on the robot at --fps Hz.
5. During execution, keep accumulating frames into the image history.
6. After the chunk, wait until the scene is still.
7. Repeat from step 1 until --max_steps inference calls are reached.

Usage
-----
    python eval/so101/robot_controller.py \\
        --socket_path /tmp/vam_12345 \\
        [--robot_port /dev/ttyACM1] \\
        [--task_description "Push ..."] \\
        [see --help for all options]

Normally started by eval.sh, which supplies --socket_path automatically and
passes all other arguments through.

Notes
-----
* einops is intentionally NOT imported.  The HWC→CHW transpose is done with
  plain numpy so this file has no dependency on the model environment.
* Both Python scripts receive all CLI arguments from eval.sh and silently
  ignore the ones they don't recognise (via parse_known_args).
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import pathlib
import sys
import time

import numpy as np
import zmq
from PIL import Image

# ---------------------------------------------------------------------------
# LeRobot imports  (lerobot/src must be on PYTHONPATH — set by eval.sh)
# ---------------------------------------------------------------------------
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

# ---------------------------------------------------------------------------
# Shared video transform config
# (data_preprocessing/ must be on PYTHONPATH — set by eval.sh, or via
# the sys.path.insert below as a fallback)
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(_REPO_ROOT / "data_preprocessing"))
from video_config import SO101_VIDEO_CONFIG  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — must stay in sync with model/cosmos_predict2 configuration
# and data_preprocessing/action/process_lerobot.py
# ---------------------------------------------------------------------------

# SO101 joint names in the order stored by the robot (gripper excluded from model I/O)
JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

VIDEO_CROP_X = SO101_VIDEO_CONFIG.crop_x
VIDEO_CROP_Y = SO101_VIDEO_CONFIG.crop_y
VIDEO_CROP_W = SO101_VIDEO_CONFIG.crop_w
VIDEO_CROP_H = SO101_VIDEO_CONFIG.crop_h
TARGET_H = SO101_VIDEO_CONFIG.target_h
TARGET_W = SO101_VIDEO_CONFIG.target_w

# Horizons — must match model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml
IMG_HORIZON = 5       # frames at IMG_FPS fed to the video backbone
LOWDIM_HORIZON = 1    # single current state observation
ACTION_HORIZON = 15   # actions predicted per inference call

# Camera runs at 10 fps; model was trained at 5 fps
CAMERA_FPS = 10
IMG_FPS = 5
IMG_SUBSAMPLE = CAMERA_FPS // IMG_FPS  # = 2: keep every 2nd frame

# Fixed video-model denoising budget (matches run.py; exposed as server protocol field
# so the server can pass it to model() without hard-coding it there)
_NUM_SAMPLING_STEP = 35

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image / state preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Crop, resize, and normalise a raw camera frame to model input format.

    Args:
        image: uint8 RGB array of shape (H, W, 3), e.g. (1080, 1920, 3).

    Returns:
        float32 array of shape (3, 1, TARGET_H, TARGET_W) in [-1, 1].

    Note:
        einops is intentionally avoided to remove the dependency on the
        model environment from this (lerobot) process.
    """
    cropped = image[
        VIDEO_CROP_Y : VIDEO_CROP_Y + VIDEO_CROP_H,
        VIDEO_CROP_X : VIDEO_CROP_X + VIDEO_CROP_W,
    ]
    resized = np.array(Image.fromarray(cropped).resize((TARGET_W, TARGET_H)))
    # HWC (H, W, C) → CHW (C, H, W), then add temporal dim T=1 → (C, 1, H, W)
    chw = resized.transpose(2, 0, 1)[:, None, :, :]
    return 2.0 * (chw.astype(np.float32) / 255.0 - 0.5)


def _to_grayscale_small(image: np.ndarray) -> np.ndarray:
    """Return a small float32 grayscale crop of the scene for motion detection."""
    cropped = image[
        VIDEO_CROP_Y : VIDEO_CROP_Y + VIDEO_CROP_H,
        VIDEO_CROP_X : VIDEO_CROP_X + VIDEO_CROP_W,
    ]
    gray = np.array(Image.fromarray(cropped).resize((TARGET_W, TARGET_H)).convert("L"))
    return gray.astype(np.float32)


def build_state(obs: dict) -> np.ndarray:
    """Extract the 5-DOF joint state in radians from a robot observation dict.

    The robot API returns positions in degrees.
    The model was trained on radians (process_lerobot.py applies deg2rad).

    Args:
        obs: dict from robot.get_observation() with keys like "shoulder_pan.pos".

    Returns:
        float32 array of shape (5,) in radians.
    """
    joints_deg = np.array([obs[f"{j}.pos"] for j in JOINT_NAMES], dtype=np.float32)
    return np.deg2rad(joints_deg)


# ---------------------------------------------------------------------------
# History assembly  (numpy-only; tensor conversion happens in model_server.py)
# ---------------------------------------------------------------------------

def _assemble_image_input(img_deque: collections.deque) -> np.ndarray:
    """Subsample the rolling image deque and concatenate into a model input array.

    Converts 10 fps → 5 fps by taking every IMG_SUBSAMPLE-th frame, then
    stacks IMG_HORIZON frames along the time axis.

    Returns:
        float32 C-contiguous array of shape (3, IMG_HORIZON, TARGET_H, TARGET_W).
    """
    frames = list(img_deque)
    subsampled = frames[::IMG_SUBSAMPLE]
    # Guard: pad with oldest frame if somehow fewer than IMG_HORIZON remain
    while len(subsampled) < IMG_HORIZON:
        subsampled.insert(0, subsampled[0])
    subsampled = subsampled[:IMG_HORIZON]
    # Each frame: (3, 1, H, W) → concatenate along axis 1 → (3, IMG_HORIZON, H, W)
    vid = np.concatenate(subsampled, axis=1)
    return np.ascontiguousarray(vid)


def _assemble_state_input(lowdim_deque: collections.deque) -> np.ndarray:
    """Stack the joint-state history into a model input array.

    Returns:
        float32 C-contiguous array of shape (LOWDIM_HORIZON, 5).
    """
    state = np.stack(list(lowdim_deque), axis=0)
    return np.ascontiguousarray(state)


# ---------------------------------------------------------------------------
# ZMQ inference call
# ---------------------------------------------------------------------------

def call_model_server(
    sock: zmq.Socket,
    img_np: np.ndarray,
    state_np: np.ndarray,
    task: str,
    seed: int,
    stop_denoising_step: int | None,
) -> np.ndarray:
    """Send one inference request to the model server and return the action chunk.

    Args:
        sock:               ZMQ REQ socket connected to model_server.py.
        img_np:             float32 array (3, IMG_HORIZON, TARGET_H, TARGET_W),
                            C-contiguous (guaranteed by _assemble_image_input).
        state_np:           float32 array (LOWDIM_HORIZON, 5),
                            C-contiguous (guaranteed by _assemble_state_input).
        task:               Natural-language task description.
        seed:               Inference-step index used as the RNG seed.
        stop_denoising_step: Early-exit denoising step, or None for full denoising.

    Returns:
        float32 array of shape (ACTION_HORIZON, 5) in radians.

    Raises:
        RuntimeError:      If the server returns an error response.
        zmq.error.Again:   If the server does not reply within the socket's
                           RCVTIMEO (set in main()).  The REQ socket is then
                           in an error state and must not be reused — the
                           caller's finally block should close it.
    """
    header = {
        "type": "infer",
        "img_shape": list(img_np.shape),
        "img_dtype": str(img_np.dtype),
        "state_shape": list(state_np.shape),
        "state_dtype": str(state_np.dtype),
        "task": task,
        "num_sampling_step": _NUM_SAMPLING_STEP,
        "stop_denoising_step": stop_denoising_step,
        "seed": seed,
    }
    sock.send_multipart([json.dumps(header).encode(), img_np.tobytes(), state_np.tobytes()])

    parts = sock.recv_multipart()
    resp_header = json.loads(parts[0])

    if resp_header["type"] == "error":
        raise RuntimeError(f"Model server returned an error:\n{resp_header['msg']}")

    actions_shape = tuple(resp_header["actions_shape"])
    actions_dtype = np.dtype(resp_header["actions_dtype"])
    return np.frombuffer(parts[1], dtype=actions_dtype).reshape(actions_shape).copy()


# ---------------------------------------------------------------------------
# Stopping detection
# ---------------------------------------------------------------------------

def wait_until_still(
    robot: SO101Follower,
    camera_key: str,
    threshold: float = 3.0,
    consecutive_checks: int = 3,
    check_interval_s: float = 0.1,
    max_wait_s: float = 10.0,
) -> None:
    """Block until the scene is still (arm / object has stopped moving).

    Compares consecutive grayscale frames; returns once the mean absolute
    per-pixel difference stays below `threshold` for `consecutive_checks`
    checks in a row.

    Args:
        robot:             Connected SO101Follower instance.
        camera_key:        Key in the observation dict for the camera image.
        threshold:         Mean per-pixel diff (0–255) below which motion is
                           considered negligible (default 3.0).
        consecutive_checks: Checks in a row that must pass before returning.
        check_interval_s:  Time between frames in seconds.
        max_wait_s:        Hard upper limit; returns regardless after this.
    """
    prev_frame: np.ndarray | None = None
    still_count = 0
    deadline = time.monotonic() + max_wait_s

    while time.monotonic() < deadline:
        obs = robot.get_observation()
        frame = _to_grayscale_small(obs[camera_key])

        if prev_frame is not None:
            diff = float(np.mean(np.abs(frame - prev_frame)))
            if diff < threshold:
                still_count += 1
                logger.debug(
                    "still check %d/%d (diff=%.2f)", still_count, consecutive_checks, diff
                )
                if still_count >= consecutive_checks:
                    logger.debug("Scene declared still after %d checks.", still_count)
                    return
            else:
                still_count = 0
                logger.debug("Scene moving (diff=%.2f)", diff)

        prev_frame = frame
        time.sleep(check_interval_s)

    logger.warning("wait_until_still: timed out after %.1f s.", max_wait_s)


# ---------------------------------------------------------------------------
# Closed-loop control
# ---------------------------------------------------------------------------

def run_episode(
    robot: SO101Follower,
    zmq_socket: zmq.Socket,
    task_description: str,
    camera_key: str,
    fps: int,
    num_execute_actions: int,
    stop_denoising_step: int | None,
    still_threshold: float,
    max_wait_s: float,
    max_steps: int,
) -> None:
    """Run one closed-loop episode.

    On each inference step the model predicts ACTION_HORIZON actions;
    `num_execute_actions` of them are executed before re-planning.
    The image history accumulates during execution so the model always
    receives the most recent visual context.

    Args:
        robot:               Connected SO101Follower.
        zmq_socket:          ZMQ REQ socket connected to model_server.py.
        task_description:    Natural-language task prompt.
        camera_key:          Observation dict key for the front camera.
        fps:                 Robot control frequency (Hz).
        num_execute_actions: Actions from the chunk to execute before re-planning.
        stop_denoising_step: Early-exit step for video denoising (None = full).
        still_threshold:     Per-pixel grayscale diff for motion detection.
        max_wait_s:          Max seconds to wait for scene to settle per chunk.
        max_steps:           Maximum number of model inference calls.
    """
    step_period = 1.0 / fps

    # Rolling histories.
    # maxlen: (IMG_HORIZON - 1) * IMG_SUBSAMPLE + 1 frames at CAMERA_FPS yield
    # exactly IMG_HORIZON frames at IMG_FPS when taking every IMG_SUBSAMPLE-th.
    img_deque: collections.deque = collections.deque(
        maxlen=(IMG_HORIZON - 1) * IMG_SUBSAMPLE + 1
    )
    lowdim_deque: collections.deque = collections.deque(maxlen=LOWDIM_HORIZON)

    def _observe_and_enqueue() -> dict:
        """Take one robot observation and push it into the history deques."""
        obs = robot.get_observation()
        img_deque.append(preprocess_image(obs[camera_key]))
        lowdim_deque.append(build_state(obs))
        return obs

    # Seed both deques with the initial observation, then pad to full length.
    _observe_and_enqueue()
    while len(img_deque) < img_deque.maxlen:
        img_deque.appendleft(img_deque[0].copy())
    while len(lowdim_deque) < LOWDIM_HORIZON:
        lowdim_deque.appendleft(lowdim_deque[0].copy())

    logger.info(
        "Image deque: %d/%d  |  lowdim deque: %d/%d",
        len(img_deque), img_deque.maxlen,
        len(lowdim_deque), LOWDIM_HORIZON,
    )

    for inference_step in range(max_steps):
        logger.info("=== Inference step %d / %d ===", inference_step + 1, max_steps)

        # ── Assemble numpy inputs for the model server ─────────────────────
        img_np = _assemble_image_input(img_deque)       # (3, IMG_HORIZON, H, W)
        state_np = _assemble_state_input(lowdim_deque)  # (LOWDIM_HORIZON, 5)

        logger.debug("img_np: %s  |  state_np: %s", img_np.shape, state_np.shape)

        # ── Request inference from the model server ─────────────────────────
        logger.info("Requesting model inference (seed=%d) ...", inference_step)
        actions_rad = call_model_server(
            sock=zmq_socket,
            img_np=img_np,
            state_np=state_np,
            task=task_description,
            seed=inference_step,
            stop_denoising_step=stop_denoising_step,
        )
        # actions_rad: (ACTION_HORIZON, 5) in radians
        logger.info("Received %d actions from model server.", len(actions_rad))

        # ── Execute action chunk ────────────────────────────────────────────
        n_exec = min(num_execute_actions, len(actions_rad))
        logger.info("Executing %d / %d actions at %d Hz ...", n_exec, len(actions_rad), fps)

        for i in range(n_exec):
            t_start = time.monotonic()

            # Observe before sending — accumulates visual context during execution.
            _observe_and_enqueue()

            # Convert radians → degrees and send to robot.
            action_deg = np.rad2deg(actions_rad[i])
            robot_action = {
                f"{joint}.pos": float(action_deg[j])
                for j, joint in enumerate(JOINT_NAMES)
            }
            robot.send_action(robot_action)

            elapsed = time.monotonic() - t_start
            sleep_remaining = step_period - elapsed
            if sleep_remaining > 0:
                time.sleep(sleep_remaining)

        # ── Wait for scene to settle ────────────────────────────────────────
        logger.info("Waiting for scene to settle ...")
        wait_until_still(
            robot=robot,
            camera_key=camera_key,
            threshold=still_threshold,
            max_wait_s=max_wait_s,
        )

        # Capture one final post-settle observation to anchor the next inference.
        _observe_and_enqueue()

    logger.info("Episode complete — %d inference steps executed.", max_steps)


# ---------------------------------------------------------------------------
# Robot initialisation
# ---------------------------------------------------------------------------

def make_robot(
    robot_port: str,
    camera_index: int,
    camera_key: str,
    fps: int,
) -> SO101Follower:
    """Construct an SO101Follower with a single front camera."""
    cam_cfg = OpenCVCameraConfig(
        index_or_path=camera_index,
        fps=fps,
        width=1920,
        height=1080,
        warmup_s=5,
    )
    robot_cfg = SOFollowerRobotConfig(
        port=robot_port,
        cameras={camera_key: cam_cfg},
        disable_torque_on_disconnect=True,
        # Safety: limit how far each joint can move in one command (degrees).
        # Remove or increase if the trained model produces larger deltas.
        max_relative_target=10.0,
    )
    return SO101Follower(robot_cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    repo_root = pathlib.Path(__file__).parents[2]
    default_task_path = repo_root / "description_task1.txt"
    default_task = (
        default_task_path.read_text(encoding="utf-8").strip()
        if default_task_path.exists()
        else "Push the white polyhedron to the goal position."
    )

    p = argparse.ArgumentParser(
        description="Robot controller for the SO101 Video-World-Action pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── IPC ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("IPC")
    g.add_argument(
        "--socket_path", required=True,
        help="ZMQ IPC socket path (must match --socket_path of model_server.py).",
    )
    g.add_argument(
        "--recv_timeout_ms", type=int, default=300_000,
        help=(
            "Milliseconds to wait for a model-server reply before raising an error. "
            "Full 35-step denoising takes ~60 s on an H100; default 300 s is generous."
        ),
    )

    # ── Robot ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Robot")
    g.add_argument("--robot_port", default="/dev/ttyACM1",
                   help="Serial port for the SO101 motor bus (find with: lerobot-find-port).")
    g.add_argument("--camera_index", type=int, default=0,
                   help="OpenCV camera device index.")
    g.add_argument("--camera_key", default="front",
                   help="Observation dict key for the front camera.")

    # ── Task ─────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Task")
    g.add_argument("--task_description", default=default_task,
                   help="Natural-language task prompt fed to the video model.")

    # ── Control ──────────────────────────────────────────────────────────────
    g = p.add_argument_group("Control")
    g.add_argument("--fps", type=int, default=10,
                   help="Robot control frequency (Hz); also sets camera capture rate.")
    g.add_argument("--num_execute_actions", type=int, default=8,
                   help="Number of actions from the 15-step chunk to execute before re-planning.")
    g.add_argument("--max_steps", type=int, default=20,
                   help="Maximum number of model inference calls (episode length).")
    g.add_argument(
        "--stop_denoising_step", type=int, default=20,
        help=(
            "Early-stop step for video denoising "
            "(default 20 trades quality for speed; pass 35 for full denoising)."
        ),
    )

    # ── Stopping detection ───────────────────────────────────────────────────
    g = p.add_argument_group("Stopping detection")
    g.add_argument("--still_threshold", type=float, default=3.0,
                   help="Mean per-pixel grayscale diff (0–255) below which scene is 'still'.")
    g.add_argument("--max_wait_s", type=float, default=10.0,
                   help="Maximum seconds to wait for the scene to settle after each chunk.")

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # parse_known_args so eval.sh can forward all CLI args to both processes
    # without this script failing on model-server-only args (--video_model etc.).
    return p.parse_known_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args, unknown = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  [controller]  %(message)s",
        datefmt="%H:%M:%S",
    )
    if unknown:
        logger.debug(
            "Ignoring unrecognised arguments (belong to model_server): %s", unknown
        )

    # ── ZMQ setup ─────────────────────────────────────────────────────────────
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    # LINGER=0: discard unsent messages immediately on close.
    sock.setsockopt(zmq.LINGER, 0)
    # RCVTIMEO: raise zmq.error.Again if server doesn't reply within this window.
    # After a timeout the REQ socket is in an error state and must not be reused;
    # the finally block below closes it.
    sock.setsockopt(zmq.RCVTIMEO, args.recv_timeout_ms)
    sock.connect(f"ipc://{args.socket_path}")
    logger.info("Connected to model server at ipc://%s", args.socket_path)

    # ── Robot setup ───────────────────────────────────────────────────────────
    robot = make_robot(
        robot_port=args.robot_port,
        camera_index=args.camera_index,
        camera_key=args.camera_key,
        fps=args.fps,
    )

    try:
        logger.info("Connecting to SO101 on %s ...", args.robot_port)
        robot.connect(calibrate=True)
        logger.info("Robot connected.")

        # Close gripper before starting — task is a push, gripper stays closed.
        logger.info("Closing gripper ...")
        robot.send_action({"gripper.pos": 0.0})
        time.sleep(1.0)

        # The OpenCVCamera already warmed up during connect(warmup_s=5).
        # Flush a few extra frames to ensure stable exposure / white balance.
        logger.info("Flushing camera buffer ...")
        for _ in range(10):
            robot.get_observation()
            time.sleep(1.0 / args.fps)

        logger.info("Task: %s", args.task_description)
        logger.info("Starting closed-loop episode (max_steps=%d) ...", args.max_steps)

        run_episode(
            robot=robot,
            zmq_socket=sock,
            task_description=args.task_description,
            camera_key=args.camera_key,
            fps=args.fps,
            num_execute_actions=args.num_execute_actions,
            stop_denoising_step=args.stop_denoising_step,
            still_threshold=args.still_threshold,
            max_wait_s=args.max_wait_s,
            max_steps=args.max_steps,
        )

        logger.info("Episode complete.")

    finally:
        logger.info("Disconnecting robot ...")
        robot.disconnect()
        logger.info("Robot disconnected.")
        sock.close()
        ctx.term()
        logger.info("ZMQ context terminated.")


if __name__ == "__main__":
    main()
