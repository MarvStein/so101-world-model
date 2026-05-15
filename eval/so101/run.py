"""Closed-loop deployment of the Video-World-Action (VAM) pipeline on the SO101 robot arm.

The control loop
----------------
1. Capture a camera frame and joint positions from the SO101.
2. Pass the rolling image history and current joint state to Video2World2ActionPipeline.
3. Execute the returned action chunk on the robot (one action per 1/fps second).
4. During execution, keep accumulating frames into the image history.
5. After the chunk, wait until the scene is still (polyhedron stops moving) using
   frame-difference detection on the camera feed.
6. Repeat from step 1 until max_steps inference calls are reached.

Usage
-----
    python eval/so101/run.py \\
        --video_model  /path/to/video_backbone.pt \\
        --action_model /path/to/action_decoder.pt \\
        --stats_path   /path/to/dataset_statistics.json

See --help for all options.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import pathlib
import sys
import time

import einops
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Model imports (model/ must be on PYTHONPATH — see eval.sh)
# ---------------------------------------------------------------------------
from cosmos_predict2.configs.config import make_config
from cosmos_predict2.data.action.utils import extract_normalization_types
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override

# ---------------------------------------------------------------------------
# LeRobot imports (lerobot/src must be on PYTHONPATH — see eval.sh)
# ---------------------------------------------------------------------------
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

# ---------------------------------------------------------------------------
# Constants matching data_preprocessing/action/process_lerobot.py and
# data_preprocessing/video/process_lerobot_video.py  (-vf "crop=1268:951:326:0,fps=10")
# ---------------------------------------------------------------------------

# SO101 joint names in the order stored by the robot (gripper excluded from model I/O)
JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# Crop parameters: w=1268, h=951, x=326, y=0
VIDEO_CROP_X = 326
VIDEO_CROP_Y = 0
VIDEO_CROP_W = 1268
VIDEO_CROP_H = 951

# Model input resolution
TARGET_H = 480
TARGET_W = 640

# Horizons from model/cosmos_predict2/configs/dataloading/policy_io/lerobot.yaml
IMG_HORIZON = 5    # 5 frames at IMG_FPS
LOWDIM_HORIZON = 1  # single current state observation
ACTION_HORIZON = 15  # model predicts 15-step sequence

# Camera runs at 10 fps; model was trained with workspace_rgb.target_frequency=5 fps
CAMERA_FPS = 10
IMG_FPS = 5
IMG_SUBSAMPLE = CAMERA_FPS // IMG_FPS  # = 2: take every 2nd frame

# Experiment name from model/cosmos_predict2/configs/experiment/world2action.py
DEFAULT_EXPERIMENT = "w2a_lerobot_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline(
    video_model_path: str,
    action_model_path: str,
    stats_path: pathlib.Path,
    experiment_name: str = DEFAULT_EXPERIMENT,
) -> Video2World2ActionPipeline:
    """Load the Video2World2Action pipeline and initialise the normaliser from stats.

    Mirrors eval/bridge/SimplerEnv/simpler_env/policies/vam/video_action_model.py::
    load_video2world2action_pipeline().
    """
    config = make_config()
    config = override(config, ["--", f"experiment={experiment_name}"])

    # Task descriptions for SO101 are benign; disable text guardrail.
    config.model.config.video_pipe_config.guardrail_config.enabled = False

    video2world_pipe = Video2WorldPipeline.from_config(
        config=config.model.config.video_pipe_config,
        dit_path=video_model_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=False,
    )

    world2action_pipe = World2ActionPipeline.from_config(
        config.model.config.pipe_config,
        dit_path=action_model_path,
        device="cuda",
        dtype=torch.bfloat16,
    )

    data_config = instantiate(config.data_config)

    with stats_path.open("rb") as f:
        stats = json.load(f)

    world2action_pipe.normalizer.build_from_stats(
        stats,
        normalization_types=extract_normalization_types(data_config.policy_io.policy_io),
        concat_groups=data_config.policy_io.concat_groups,
        device="cuda",
        dtype=torch.bfloat16,
    )

    return Video2World2ActionPipeline(video2world_pipe, world2action_pipe).cuda()


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Crop, resize, and normalise a raw camera frame to model input format.

    Args:
        image: uint8 RGB array of shape (H, W, 3), e.g. (1080, 1920, 3).

    Returns:
        float32 array of shape (3, 1, 480, 640) with values in [-1, 1].
    """
    cropped = image[
        VIDEO_CROP_Y : VIDEO_CROP_Y + VIDEO_CROP_H,
        VIDEO_CROP_X : VIDEO_CROP_X + VIDEO_CROP_W,
    ]
    # PIL resize expects (width, height)
    resized = np.array(Image.fromarray(cropped).resize((TARGET_W, TARGET_H)))
    # HWC → CHW, add temporal dim T=1 → (C, 1, H, W)
    chw = einops.rearrange(resized, "h w c -> c h w")[:, None, :, :]
    return 2.0 * (chw.astype(np.float32) / 255.0 - 0.5)


def _to_grayscale_small(image: np.ndarray) -> np.ndarray:
    """Return a small float32 grayscale crop for motion detection."""
    cropped = image[
        VIDEO_CROP_Y : VIDEO_CROP_Y + VIDEO_CROP_H,
        VIDEO_CROP_X : VIDEO_CROP_X + VIDEO_CROP_W,
    ]
    gray = np.array(Image.fromarray(cropped).resize((TARGET_W, TARGET_H)).convert("L"))
    return gray.astype(np.float32)


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------

def build_state(obs: dict) -> np.ndarray:
    """Extract the 5-DOF joint state in radians from a robot observation dict.

    The robot API returns joint positions in degrees (use_degrees=True).
    The model was trained on radians (process_lerobot.py applies deg2rad).

    Args:
        obs: dict from robot.get_observation() containing keys like
             "shoulder_pan.pos" (float, degrees).

    Returns:
        float32 array of shape (5,) in radians.
    """
    joints_deg = np.array([obs[f"{j}.pos"] for j in JOINT_NAMES], dtype=np.float32)
    return np.deg2rad(joints_deg)


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------

def _build_image_input(
    img_deque: collections.deque,
) -> torch.Tensor:
    """Stack the rolling image deque into a model-ready tensor.

    Subsamples every IMG_SUBSAMPLE-th element to convert 10 fps → 5 fps,
    then stacks IMG_HORIZON frames along the time axis.

    Returns:
        bfloat16 CUDA tensor of shape (1, 3, IMG_HORIZON, 480, 640).
    """
    frames = list(img_deque)
    # Take every IMG_SUBSAMPLE-th frame (oldest first) to get IMG_HORIZON frames at IMG_FPS
    subsampled = frames[::IMG_SUBSAMPLE]
    # Guard: pad with first frame if we somehow have fewer than IMG_HORIZON
    while len(subsampled) < IMG_HORIZON:
        subsampled.insert(0, subsampled[0])
    subsampled = subsampled[:IMG_HORIZON]
    # Each frame: (3, 1, 480, 640) → concatenate along axis 1 → (3, IMG_HORIZON, 480, 640)
    vid = np.concatenate(subsampled, axis=1)
    return torch.from_numpy(vid).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)


def _build_state_input(lowdim_deque: collections.deque) -> torch.Tensor:
    """Stack the lowdim history into a model-ready tensor.

    Returns:
        bfloat16 CUDA tensor of shape (1, LOWDIM_HORIZON, 5).
    """
    lowdims = np.stack(list(lowdim_deque), axis=0)  # (LOWDIM_HORIZON, 5)
    return torch.from_numpy(lowdims).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)


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
    """Block until the scene is still (polyhedron has stopped moving).

    Compares consecutive grayscale frames; returns once the mean absolute
    per-pixel difference stays below `threshold` for `consecutive_checks`
    checks in a row.

    Args:
        robot: connected SO101Follower instance.
        camera_key: key in observation dict for the camera image.
        threshold: mean per-pixel grayscale difference below which motion is
                   considered negligible (default 3.0 on a 0-255 scale).
        consecutive_checks: number of consecutive below-threshold checks required.
        check_interval_s: time between frames in seconds.
        max_wait_s: hard upper limit on waiting time; returns regardless.
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
                logger.debug("still check %d/%d (diff=%.2f)", still_count, consecutive_checks, diff)
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
    model: Video2World2ActionPipeline,
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
    The image history is accumulated during execution so the model always
    receives the most recent visual context.

    Args:
        robot: connected SO101Follower.
        model: loaded Video2World2ActionPipeline on CUDA in eval mode.
        task_description: natural-language task prompt.
        camera_key: observation dict key for the front camera.
        fps: robot control frequency (Hz); determines sleep between actions.
        num_execute_actions: how many actions from the predicted chunk to execute.
        stop_denoising_step: optional early exit for video denoising
            (None = full 35 steps; ~20 trades quality for speed).
        still_threshold: per-pixel grayscale diff threshold for motion detection.
        max_wait_s: maximum seconds to wait for scene to settle per chunk.
        max_steps: maximum number of model inference calls.
    """
    step_period = 1.0 / fps

    # Rolling histories — maxlen ensures we keep enough frames for subsampling
    # (IMG_HORIZON - 1) * IMG_SUBSAMPLE + 1 frames at CAMERA_FPS give IMG_HORIZON
    # frames at IMG_FPS when taking every IMG_SUBSAMPLE-th.
    img_deque: collections.deque[np.ndarray] = collections.deque(
        maxlen=(IMG_HORIZON - 1) * IMG_SUBSAMPLE + 1
    )
    lowdim_deque: collections.deque[np.ndarray] = collections.deque(maxlen=LOWDIM_HORIZON)

    def _observe_and_enqueue() -> dict:
        """Take one robot observation and push it into the history deques."""
        obs = robot.get_observation()
        img_deque.append(preprocess_image(obs[camera_key]))
        lowdim_deque.append(build_state(obs))
        return obs

    # Seed both deques with the initial observation
    initial_obs = _observe_and_enqueue()
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

        # ── Build model inputs from the current rolling histories ──────────
        input_vid = _build_image_input(img_deque)    # (1, 3, 5, 480, 640)
        state_tensor = _build_state_input(lowdim_deque)  # (1, 1, 5)

        logger.debug(
            "input_vid shape: %s  |  state shape: %s",
            tuple(input_vid.shape), tuple(state_tensor.shape),
        )

        # ── Model inference ────────────────────────────────────────────────
        logger.info("Running model inference ...")
        pred_actions = model(
            input_vid=input_vid,
            state_B_HO_O=state_tensor,
            prompt=task_description,
            num_sampling_step=35,
            stop_after_step=stop_denoising_step,
            use_cuda_graphs=True,
            seed=inference_step,
        )
        # (1, ACTION_HORIZON, 5) → (ACTION_HORIZON, 5) in radians
        actions_rad: np.ndarray = pred_actions[0].float().cpu().numpy()
        logger.info("Predicted %d actions.", len(actions_rad))

        # ── Execute action chunk ───────────────────────────────────────────
        n_exec = min(num_execute_actions, len(actions_rad))
        logger.info("Executing %d actions at %d Hz ...", n_exec, fps)

        for i in range(n_exec):
            t_start = time.monotonic()

            # Observe before sending (accumulates visual context during execution)
            _observe_and_enqueue()

            # Convert radians → degrees and send to robot
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

        # ── Wait for scene to settle ───────────────────────────────────────
        logger.info("Waiting for scene to settle ...")
        wait_until_still(
            robot=robot,
            camera_key=camera_key,
            threshold=still_threshold,
            max_wait_s=max_wait_s,
        )

        # Capture the final post-settle observation to anchor next inference
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

def parse_args() -> argparse.Namespace:
    repo_root = pathlib.Path(__file__).parents[2]
    default_task_path = repo_root / "description_task1.txt"
    default_task = (
        default_task_path.read_text(encoding="utf-8").strip()
        if default_task_path.exists()
        else "Push the white polyhedron to the goal position."
    )

    p = argparse.ArgumentParser(
        description="Closed-loop SO101 deployment with the Video-World-Action pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Robot ────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Robot")
    g.add_argument("--robot_port", default="/dev/ttyACM1",
                   help="Serial port for the SO101 motor bus (find with: lerobot-find-port).")
    g.add_argument("--camera_index", type=int, default=4,
                   help="OpenCV camera device index.")
    g.add_argument("--camera_key", default="front",
                   help="Observation dict key to use for the front camera.")

    # ── Model checkpoints ────────────────────────────────────────────────────
    g = p.add_argument_group("Model checkpoints")
    g.add_argument("--video_model", required=True,
                   help="Path to the fine-tuned video backbone .pt checkpoint.")
    g.add_argument("--action_model", required=True,
                   help="Path to the trained action decoder .pt checkpoint.")
    g.add_argument("--stats_path", required=True,
                   help="Path to dataset_statistics.json used to initialise the normaliser.")
    g.add_argument("--experiment", default=DEFAULT_EXPERIMENT,
                   help="Experiment config name (override if fine-tuned under a different name).")

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
    g.add_argument("--stop_denoising_step", type=int, default=None,
                   help="Early-stop step for video denoising (None = full 35 steps; ~20 for faster eval).")

    # ── Stopping detection ───────────────────────────────────────────────────
    g = p.add_argument_group("Stopping detection")
    g.add_argument("--still_threshold", type=float, default=3.0,
                   help="Mean per-pixel grayscale diff (0-255) below which scene is considered still.")
    g.add_argument("--max_wait_s", type=float, default=10.0,
                   help="Maximum seconds to wait for the scene to settle after each action chunk.")

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading Video2World2Action pipeline ...")
    model = load_pipeline(
        video_model_path=args.video_model,
        action_model_path=args.action_model,
        stats_path=pathlib.Path(args.stats_path),
        experiment_name=args.experiment,
    )
    model.eval()
    logger.info("Pipeline loaded and in eval mode.")

    # ── Init robot ────────────────────────────────────────────────────────────
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

        # Close gripper before starting — task is a push, gripper stays closed
        logger.info("Closing gripper ...")
        robot.send_action({"gripper.pos": 0.0})
        time.sleep(1.0)

        # The OpenCVCamera already warmed up during connect(warmup_s=5).
        # Flush a few extra frames to ensure stable exposure.
        logger.info("Flushing camera buffer ...")
        for _ in range(10):
            robot.get_observation()
            time.sleep(1.0 / args.fps)

        logger.info("Task: %s", args.task_description)
        logger.info("Starting closed-loop episode (max_steps=%d) ...", args.max_steps)

        run_episode(
            robot=robot,
            model=model,
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


if __name__ == "__main__":
    main()
