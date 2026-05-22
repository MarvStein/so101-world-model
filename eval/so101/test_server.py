"""Client-side emulation script for testing model_server.py.

Sends a single inference request with random (or zero) dummy arrays to the
model server over ZMQ TCP and prints the response.  No robot or lerobot
environment is required — only numpy and pyzmq.

Usage
-----
    # Basic test against a locally forwarded brev port:
    python eval/so101/test_server.py --task_id 1

    # Custom server address, zero arrays, full denoising:
    python eval/so101/test_server.py \\
        --task_id 1 \\
        --server_host localhost --server_port 5555 \\
        --num_sampling_step 35 --stop_denoising_step 35 \\
        --zeros

    # Multiple back-to-back requests (stress test):
    python eval/so101/test_server.py --task_id 1 --num_requests 3
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import zmq

# ---------------------------------------------------------------------------
# Constants — must match robot_controller.py / model training config
# ---------------------------------------------------------------------------
IMG_HORIZON = 5
LOWDIM_HORIZON = 1
TARGET_H = 480
TARGET_W = 640
DOF = 5
_NUM_SAMPLING_STEP = 35

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

def build_request(
    task_id: str,
    seed: int,
    num_sampling_step: int,
    stop_denoising_step: int | None,
    zeros: bool,
) -> tuple[bytes, bytes, bytes]:
    """Build the 3-part ZMQ message for one inference request.

    Returns:
        (header_bytes, img_bytes, state_bytes) ready for send_multipart().
    """
    if zeros:
        img_np = np.zeros((3, IMG_HORIZON, TARGET_H, TARGET_W), dtype=np.float32)
        state_np = np.zeros((LOWDIM_HORIZON, DOF), dtype=np.float32)
    else:
        rng = np.random.default_rng(seed)
        # Image in [-1, 1] (same range as preprocess_image output)
        img_np = rng.uniform(-1.0, 1.0, size=(3, IMG_HORIZON, TARGET_H, TARGET_W)).astype(np.float32)
        # State in radians — small random joint angles
        state_np = rng.uniform(-0.5, 0.5, size=(LOWDIM_HORIZON, DOF)).astype(np.float32)

    img_np = np.ascontiguousarray(img_np)
    state_np = np.ascontiguousarray(state_np)

    header = {
        "type": "infer",
        "img_shape": list(img_np.shape),
        "img_dtype": str(img_np.dtype),
        "state_shape": list(state_np.shape),
        "state_dtype": str(state_np.dtype),
        "task_id": task_id,
        "num_sampling_step": num_sampling_step,
        "stop_denoising_step": stop_denoising_step,
        "seed": seed,
    }
    return json.dumps(header).encode(), img_np.tobytes(), state_np.tobytes()


def parse_response(parts: list[bytes]) -> np.ndarray:
    """Decode a server response and return the actions array.

    Raises:
        RuntimeError: if the server returned an error.
        ValueError:   if the response is malformed.
    """
    if not parts:
        raise ValueError("Empty response from server.")
    resp_header = json.loads(parts[0])
    if resp_header["type"] == "error":
        raise RuntimeError(f"Server error:\n{resp_header['msg']}")
    if resp_header["type"] != "result" or len(parts) < 2:
        raise ValueError(f"Unexpected response: {resp_header}")
    actions_shape = tuple(resp_header["actions_shape"])
    actions_dtype = np.dtype(resp_header["actions_dtype"])
    return np.frombuffer(parts[1], dtype=actions_dtype).reshape(actions_shape).copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, args.recv_timeout_ms)

    server_addr = f"tcp://{args.server_host}:{args.server_port}"
    sock.connect(server_addr)
    logger.info("Connected to %s", server_addr)

    stop_step = args.stop_denoising_step if args.stop_denoising_step > 0 else None

    try:
        for i in range(args.num_requests):
            logger.info("--- Request %d / %d ---", i + 1, args.num_requests)
            logger.info(
                "  task_id=%s  seed=%d  sampling_steps=%d  stop_at=%s  zeros=%s",
                args.task_id, i, args.num_sampling_step, stop_step, args.zeros,
            )

            header_b, img_b, state_b = build_request(
                task_id=args.task_id,
                seed=i,
                num_sampling_step=args.num_sampling_step,
                stop_denoising_step=stop_step,
                zeros=args.zeros,
            )

            logger.info(
                "  Sending: img %.1f MB  state %.1f KB",
                len(img_b) / 1e6,
                len(state_b) / 1e3,
            )

            t0 = time.monotonic()
            sock.send_multipart([header_b, img_b, state_b])
            parts = sock.recv_multipart()
            elapsed = time.monotonic() - t0

            actions = parse_response(parts)
            logger.info(
                "  Response: actions shape=%s  dtype=%s  elapsed=%.2f s",
                actions.shape, actions.dtype, elapsed,
            )
            logger.info(
                "  Actions (deg):\n%s",
                np.rad2deg(actions),
            )

    finally:
        sock.close()
        ctx.term()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dummy client for testing model_server.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--server_host", default="localhost",
                   help="Hostname of the model server.")
    p.add_argument("--server_port", type=int, default=5555,
                   help="TCP port of the model server.")
    p.add_argument("--task_id", required=True,
                   help="Task ID to look up in the server's precomputed embeddings (e.g. '1').")
    p.add_argument("--num_sampling_step", type=int, default=_NUM_SAMPLING_STEP,
                   help="Number of diffusion denoising steps.")
    p.add_argument("--stop_denoising_step", type=int, default=20,
                   help="Early-exit denoising step (0 = full denoising = --num_sampling_step).")
    p.add_argument("--zeros", action="store_true",
                   help="Use all-zero arrays instead of random noise.")
    p.add_argument("--num_requests", type=int, default=1,
                   help="Number of back-to-back requests to send.")
    p.add_argument("--recv_timeout_ms", type=int, default=300_000,
                   help="Milliseconds to wait for a server reply.")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  [test-client]  %(message)s",
        datefmt="%H:%M:%S",
    )
    run(args)
