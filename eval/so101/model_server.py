"""Model inference server for SO101 closed-loop deployment.

Loads the Video2World2Action pipeline once, then serves inference requests
from the robot controller over a ZeroMQ TCP socket (REP side).

This process runs on the **H100 brev instance** in the model environment
(numpy == 1.26.4).  The robot controller (numpy >= 2.0) connects as the
REQ side from the local machine via brev port-forwarding.

Startup sequence
----------------
1. Bind the TCP socket BEFORE loading the model — the OS queues any early
   connects from the controller so no messages are lost during model load.
2. Load and warm up the pipeline (may take 30–120 s for large checkpoints).
3. Log the bound address and enter the REP receive loop.
4. On the local machine, run::

     brev port-forward <instance-name> --port <local-port>:<remote-port>

   then start robot_controller.py (which will prompt for manual confirmation
   before connecting).

Shutdown
--------
The server exits cleanly on SIGTERM or SIGINT.  The signal handler sets a
flag; the main loop checks it after every 1-second poll so it can finish
the current inference step before exiting.

Protocol (ZeroMQ multipart, REP side)
--------------------------------------
Controller → Server  (3-part)
  Part 0  JSON header ::
            {"type": "infer",
             "img_shape": [3, 5, 480, 640], "img_dtype": "float32",
             "state_shape": [1, 5],         "state_dtype": "float32",
             "task_id": "1",
             "num_sampling_step": 35,
             "stop_denoising_step": 20,
             "seed": 0}
  Part 1  Raw float32 bytes — image array (C-contiguous)
  Part 2  Raw float32 bytes — state array (C-contiguous)

Server → Controller  (2-part, success)
  Part 0  JSON header ::
            {"type": "result",
             "actions_shape": [15, 5], "actions_dtype": "float32"}
  Part 1  Raw float32 bytes — actions array (C-contiguous)

Server → Controller  (1-part, error)
  Part 0  JSON: {"type": "error", "msg": "<traceback>"}

Usage
-----
    python eval/so101/model_server.py \\
        --video_model     /path/to/video_backbone.pt \\
        --action_model    /path/to/action_decoder.pt \\
        --stats_path      /path/to/dataset_statistics.json \\
        --embeddings_path eval/so101/task_embeddings.pt \\
        [--host 0.0.0.0] [--port 5555]

Normally started via eval/so101/start_server.sh on the brev instance.
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import signal
import sys
import traceback

import numpy as np
import torch
import zmq

# ---------------------------------------------------------------------------
# Model imports  (model/ must be on PYTHONPATH — set by eval.sh)
# ---------------------------------------------------------------------------
from cosmos_predict2.configs.config import make_config
from cosmos_predict2.data.action.utils import extract_normalization_types
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override

DEFAULT_EXPERIMENT = "w2a_lerobot_v2w_5_5k_lr1e-04_bs32_ga4"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful-shutdown flag — set by SIGTERM / SIGINT handlers.
# The flag is checked in the main poll loop so the server always finishes
# the current inference step before exiting.
# ---------------------------------------------------------------------------
_shutdown_requested: bool = False


def _request_shutdown(signum: int, frame) -> None:  # noqa: ANN001
    global _shutdown_requested
    _shutdown_requested = True
    logger.info("Shutdown requested (signal %d); will exit after current request.", signum)


# ---------------------------------------------------------------------------
# Pipeline loading  (verbatim from eval/so101/run.py)
# ---------------------------------------------------------------------------

def load_pipeline(
    video_model_path: str,
    action_model_path: str,
    stats_path: pathlib.Path,
    experiment_name: str = DEFAULT_EXPERIMENT,
    use_text_encoder: bool = True,
) -> Video2World2ActionPipeline:
    """Load the Video2World2Action pipeline and initialise the normaliser.

    When *use_text_encoder* is False the T5-11B model is not loaded at all,
    which saves ~22 GB of storage / VRAM on the deployment machine.  In that
    case the caller must supply a precomputed *prompt_embedding* tensor to
    every :func:`run_inference` call.
    """
    config = make_config()
    config = override(config, ["--", f"experiment={experiment_name}"])

    # Task descriptions for SO101 are benign; disable the text guardrail.
    config.model.config.video_pipe_config.guardrail_config.enabled = False

    video2world_pipe = Video2WorldPipeline.from_config(
        config=config.model.config.video_pipe_config,
        dit_path=video_model_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=False,
        use_text_encoder=use_text_encoder,
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
# Inference helper
# ---------------------------------------------------------------------------

def run_inference(
    model: Video2World2ActionPipeline,
    header: dict,
    img_bytes: bytes,
    state_bytes: bytes,
    embeddings: dict[str, torch.Tensor],
) -> tuple[dict, bytes]:
    """Deserialise the request, run the model, return (response_header, actions_bytes).

    Args:
        model:      Loaded pipeline in eval mode on CUDA.
        header:     Decoded JSON header from the client request.
        img_bytes:  Raw bytes of the float32 image array (C-contiguous).
        state_bytes: Raw bytes of the float32 state array (C-contiguous).
        embeddings: Dict mapping task ID strings to precomputed T5 embeddings
                    of shape (1, 512, 1024) on CUDA as bfloat16.

    Returns:
        Tuple of the JSON-serialisable response header dict and the raw
        bytes of the float32 actions array.
    """
    img_shape = tuple(header["img_shape"])
    img_dtype = np.dtype(header["img_dtype"])
    state_shape = tuple(header["state_shape"])
    state_dtype = np.dtype(header["state_dtype"])

    img_np = np.frombuffer(img_bytes, dtype=img_dtype).reshape(img_shape)
    state_np = np.frombuffer(state_bytes, dtype=state_dtype).reshape(state_shape)

    # Add batch dimension and move to CUDA as bfloat16.
    # img_np:   (3, IMG_HORIZON, H, W)  → tensor (1, 3, IMG_HORIZON, H, W)
    # state_np: (LOWDIM_HORIZON, 5)     → tensor (1, LOWDIM_HORIZON, 5)
    input_vid = torch.from_numpy(img_np).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
    state_tensor = (
        torch.from_numpy(state_np).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16)
    )

    task_id = header["task_id"]
    if task_id not in embeddings:
        available = sorted(embeddings.keys())
        raise KeyError(
            f"Task ID {task_id!r} not found in loaded embeddings. "
            f"Available: {available}"
        )
    prompt_embedding = embeddings[task_id]

    logger.debug(
        "input_vid: %s  |  state: %s  |  task_id: %s",
        tuple(input_vid.shape), tuple(state_tensor.shape), task_id,
    )

    pred_actions = model(
        input_vid=input_vid,
        state_B_HO_O=state_tensor,
        prompt="",
        prompt_embedding=prompt_embedding,
        num_sampling_step=header["num_sampling_step"],
        stop_after_step=header.get("stop_denoising_step"),  # None → full denoising
        use_cuda_graphs=True,
        seed=header["seed"],
    )

    # (1, ACTION_HORIZON, 5) → (ACTION_HORIZON, 5) float32, C-contiguous
    actions = np.ascontiguousarray(pred_actions[0].float().cpu().numpy())

    resp_header = {
        "type": "result",
        "actions_shape": list(actions.shape),
        "actions_dtype": str(actions.dtype),
    }
    return resp_header, actions.tobytes()


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------

def serve(args: argparse.Namespace) -> None:
    """Bind the socket, load the model, then serve inference requests."""
    global _shutdown_requested

    signal.signal(signal.SIGTERM, _request_shutdown)
    signal.signal(signal.SIGINT, _request_shutdown)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    # LINGER=0: close immediately without waiting to flush outgoing messages.
    sock.setsockopt(zmq.LINGER, 0)

    bind_addr = f"tcp://{args.host}:{args.port}"
    sock.bind(bind_addr)
    logger.info("ZMQ REP socket bound at %s", bind_addr)

    # ---------------------------------------------------------------------------
    # Load all precomputed task embeddings
    # ---------------------------------------------------------------------------
    embeddings_path = pathlib.Path(args.embeddings_path)
    logger.info("Loading precomputed embeddings from %s ...", embeddings_path)
    embeddings: dict[str, torch.Tensor] = {
        k: v.to(device="cuda", dtype=torch.bfloat16)
        for k, v in torch.load(embeddings_path, map_location="cpu", weights_only=True).items()
    }
    logger.info(
        "Loaded %d task embedding(s): %s",
        len(embeddings), sorted(embeddings.keys()),
    )

    # Load the model AFTER binding so the OS can queue early client messages.
    # T5 text encoder is never loaded — embeddings are always precomputed.
    logger.info("Loading Video2World2Action pipeline (use_text_encoder=False) ...")
    model = load_pipeline(
        video_model_path=args.video_model,
        action_model_path=args.action_model,
        stats_path=pathlib.Path(args.stats_path),
        experiment_name=args.experiment,
        use_text_encoder=False,
    )
    model.eval()
    logger.info("Pipeline loaded and in eval mode.")
    logger.info("Waiting for inference requests ...")

    try:
        while not _shutdown_requested:
            # Poll with a 1-second timeout so SIGTERM is handled promptly.
            if not sock.poll(timeout=1000):
                continue

            parts = sock.recv_multipart()
            header = json.loads(parts[0])
            msg_type = header.get("type")

            if msg_type == "infer":
                if len(parts) != 3:
                    err = f"'infer' requires 3 message parts, got {len(parts)}"
                    logger.error(err)
                    sock.send_multipart([json.dumps({"type": "error", "msg": err}).encode()])
                    continue

                logger.info("Inference request received (seed=%s).", header.get("seed"))
                try:
                    resp_header, actions_bytes = run_inference(
                        model, header, parts[1], parts[2],
                        embeddings=embeddings,
                    )
                    sock.send_multipart([json.dumps(resp_header).encode(), actions_bytes])
                    logger.info(
                        "Inference complete — replied with actions %s.",
                        resp_header["actions_shape"],
                    )
                except Exception:
                    tb = traceback.format_exc()
                    logger.error("Inference exception:\n%s", tb)
                    sock.send_multipart(
                        [json.dumps({"type": "error", "msg": tb}).encode()]
                    )

            else:
                err = f"Unknown message type: {msg_type!r}"
                logger.warning(err)
                sock.send_multipart([json.dumps({"type": "error", "msg": err}).encode()])

    finally:
        sock.close()
        ctx.term()
        logger.info("ZMQ context terminated.")
        logger.info("Model server shut down cleanly.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Model inference server for the SO101 Video-World-Action pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video_model", required=True,
                   help="Path to the fine-tuned video backbone .pt checkpoint.")
    p.add_argument("--action_model", required=True,
                   help="Path to the trained action decoder .pt checkpoint.")
    p.add_argument("--stats_path", required=True,
                   help="Path to dataset_statistics.json for normaliser initialisation.")
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT,
                   help="Experiment config name.")
    p.add_argument("--host", default="0.0.0.0",
                   help="IP address for the ZMQ TCP socket to bind on.")
    p.add_argument("--port", type=int, default=5555,
                   help="TCP port for the ZMQ socket.")
    p.add_argument(
        "--embeddings_path",
        required=True,
        help=(
            "Path to the precomputed task-embeddings .pt file produced by "
            "eval/so101/precompute_task_embeddings.py.  The T5 text encoder "
            "is never loaded; all task embeddings are read from this file."
        ),
    )
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_known_args()


def main() -> None:
    args, unknown = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  [model-server]  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    if unknown:
        logger.debug("Ignoring unrecognised arguments: %s", unknown)
    serve(args)


if __name__ == "__main__":
    main()
