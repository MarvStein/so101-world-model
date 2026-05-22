#!/usr/bin/env python
"""Precompute clip-level Video2World latents and store them inside each episode zarr.

Latents are written to ``episode_xxx.zarr/precomputed_video_latents`` as a dense
array of shape ``(n_valid_steps, 16, 16, 60, 80)`` float32, indexed by the same
``step_idx`` that ``ChunkReader.read_chunk`` uses internally.  This makes the
cache split-independent: the train/val split is applied only at training time.
"""

from __future__ import annotations

import argparse
import pathlib
import pickle

import hydra
import numpy as np
from omegaconf import open_dict
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
import zarr

from megatron.core import parallel_state

from cosmos_predict2.configs.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.configs.defaults.data_action import get_data_config
from cosmos_predict2.data.action.precomputed_latents_utils import PRECOMPUTED_LATENTS_KEY
from cosmos_predict2.data.action.utils import get_paths
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from imaginaire.utils import distributed as dist_utils

# Shape of one encoded latent (C, T, H, W) — matches pipe.encode output[1:]
_LATENT_SHAPE = (16, 16, 60, 80)


def ensure_episode_paths_cache(data_dir: pathlib.Path) -> None:
    paths_cache = data_dir / "paths.pkl"
    discovered_paths = get_paths(data_dir)
    if discovered_paths:
        return

    # Some local conversions use episode folders without a .zarr suffix.
    # MimicDataset only scans **/*.zarr, so build a cache explicitly.
    episode_dirs = sorted(p for p in data_dir.glob("episode_*") if (p / ".zgroup").exists())
    if not episode_dirs:
        return

    with paths_cache.open("wb") as f:
        pickle.dump(episode_dirs, f)


def load_dataset(data_config_name: str, data_dir: pathlib.Path):
    """Load a MimicDataset covering *all* episodes (no train/val split)."""
    data_config = get_data_config(data_config_name)
    ensure_episode_paths_cache(data_dir)

    dataset_cfg = data_config.dataset.dataset
    obs_policy_io = dataset_cfg.policy_io.get("obs", {})
    data_components = dataset_cfg.get("data_components", {})
    if "language_embedding" in obs_policy_io and "language_embedding" not in data_components:
        with open_dict(obs_policy_io):
            del obs_policy_io["language_embedding"]

    # Override split so that ALL episodes are included — the split is applied
    # only at training time; precomputed latents are episode-local.
    with open_dict(dataset_cfg):
        dataset_cfg.num_val_episodes = 0

    return hydra.utils.instantiate(
        dataset_cfg,
        data_dir=str(data_dir),
        train=True,
        verbose=True,
    )


def setup_distributed(num_gpus: int) -> None:
    """Initialise NCCL process group and context parallel group for multi-GPU encode."""
    dist_utils.init()
    parallel_state.initialize_model_parallel(context_parallel_size=num_gpus)
    print(f"[rank {dist.get_rank()}] context parallel group initialised with {num_gpus} GPUs")


def load_video2world_pipeline(checkpoint_path: str, device: str, enable_guardrail: bool) -> Video2WorldPipeline:
    pipe_cfg = get_cosmos_predict2_video2world_pipeline(model_size="2B", resolution="480", fps=10)
    pipe_cfg.guardrail_config.enabled = enable_guardrail

    pipe = Video2WorldPipeline.from_config(
        pipe_cfg,
        dit_path=checkpoint_path,
        use_text_encoder=False,
        device=device,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    )
    pipe.eval()
    for param in pipe.parameters():
        param.requires_grad_(False)
    return pipe


def preallocate_episode_arrays(dataset, overwrite: bool) -> None:
    """Create (or verify) the latent zarr array inside every episode zarr.

    Only rank 0 performs zarr writes; all ranks synchronise on exit via a barrier.
    """
    is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
    if is_rank0:
        skipped = 0
        for episode_path, n_valid_steps in dataset._chunk_reader.episodes:
            with zarr.open(str(episode_path), "a") as root:
                if PRECOMPUTED_LATENTS_KEY in root:
                    if not overwrite:
                        skipped += 1
                        continue
                    del root[PRECOMPUTED_LATENTS_KEY]
                root.create_dataset(
                    PRECOMPUTED_LATENTS_KEY,
                    shape=(n_valid_steps,) + _LATENT_SHAPE,
                    chunks=(1,) + _LATENT_SHAPE,
                    dtype=np.float32,
                )
        if skipped:
            print(
                f"Skipped pre-allocation for {skipped} episode(s) that already have latents. "
                "Re-run with --overwrite to replace them."
            )
    if dist.is_initialized():
        dist.barrier()


def precompute(
    *,
    pipe: Video2WorldPipeline,
    dataset,
    num_workers: int,
) -> None:
    """Encode all samples and write latents into the episode zarrs.

    With context parallelism all ranks must call ``pipe.encode`` together — the
    tokenizer splits the temporal dimension across GPUs via NCCL and returns the
    same latent on every rank.  Only rank 0 performs the zarr writes.
    """
    is_rank0 = not dist.is_initialized() or dist.get_rank() == 0

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    chunk_reader = dataset._chunk_reader
    global_idx = 0

    # Keep one zarr handle open at a time (rank 0 only); episodes arrive in order
    # because shuffle=False and ChunkReader iterates episodes sequentially.
    current_episode_path: pathlib.Path | None = None
    current_root: zarr.Group | None = None

    with torch.inference_mode():
        for batch in tqdm(dataloader, total=len(dataloader), desc="encoding", disable=not is_rank0):
            raw_state = torch.cat((batch["obs/workspace_rgb"], batch["action/workspace_rgb"]), dim=2)
            raw_state = raw_state.to(device=pipe.tensor_kwargs["device"], dtype=torch.float32)
            # All ranks participate in this call — context parallelism communicates internally.
            latents = pipe.encode(raw_state).contiguous().float().cpu().numpy()  # (B, 16, 16, 60, 80)

            if is_rank0:
                for i in range(latents.shape[0]):
                    episode_path, step_idx = chunk_reader.resolve(global_idx)

                    if episode_path != current_episode_path:
                        if current_root is not None:
                            current_root.store.close()
                        current_root = zarr.open(str(episode_path), "a")
                        current_episode_path = episode_path

                    current_root[PRECOMPUTED_LATENTS_KEY][step_idx] = latents[i]
                    global_idx += 1
            else:
                global_idx += latents.shape[0]

    if is_rank0:
        if current_root is not None:
            current_root.store.close()
        print(f"Wrote latents for {global_idx} samples across {len(chunk_reader.episodes)} episode(s).")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Precompute Video2World latents into each episode zarr."
    )
    parser.add_argument("--video_model", required=True, help="Path to the Video2World checkpoint")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset root directory")
    parser.add_argument(
        "--data_config",
        default="bridge",
        choices=["bridge", "libero", "lerobot"],
        help="Dataset config to use for loading samples",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use via context parallelism (launch with torchrun --nproc_per_node=N).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing latents")
    parser.add_argument(
        "--enable_guardrail",
        action="store_true",
        help="Enable guardrail loading (disabled by default for precompute).",
    )
    args = parser.parse_args()

    if args.num_gpus > 1:
        setup_distributed(args.num_gpus)

    # With distributed init, torch.cuda.current_device() is set to LOCAL_RANK.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = pathlib.Path(args.dataset_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    dataset = load_dataset(args.data_config, data_dir)
    pipe = load_video2world_pipeline(args.video_model, device, args.enable_guardrail)

    preallocate_episode_arrays(dataset, overwrite=args.overwrite)
    precompute(
        pipe=pipe,
        dataset=dataset,
        num_workers=args.num_workers,
    )

    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
