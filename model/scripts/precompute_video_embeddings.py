#!/usr/bin/env python
"""Precompute clip-level Video2World latents for action-decoder training."""

from __future__ import annotations

import argparse
import pathlib
import pickle
import shutil

import hydra
import numpy as np
from omegaconf import open_dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import zarr

from cosmos_predict2.configs.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.configs.defaults.data_action import get_data_config
from cosmos_predict2.data.action.utils import get_paths
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline


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


def load_dataset(data_config_name: str, data_dir: pathlib.Path, train: bool):
    data_config = get_data_config(data_config_name)
    ensure_episode_paths_cache(data_dir)

    dataset_cfg = data_config.dataset.dataset
    obs_policy_io = dataset_cfg.policy_io.get("obs", {})
    data_components = dataset_cfg.get("data_components", {})
    # TODO is this actually true that we need to remove language embedding for precomputation?
    # Copilot came up with this
    if "language_embedding" in obs_policy_io and "language_embedding" not in data_components:
        # Precompute only needs workspace RGB clips. Allow lerobot zarrs without language embeddings.
        with open_dict(obs_policy_io):
            del obs_policy_io["language_embedding"]

    return hydra.utils.instantiate(
        dataset_cfg,
        data_dir=str(data_dir),
        train=train,
        verbose=True,
    )


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


def precompute_split(
    *,
    pipe: Video2WorldPipeline,
    dataset,
    cache_path: pathlib.Path,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
) -> None:
    if cache_path.exists() and not overwrite:
        raise FileExistsError(f"Cache already exists: {cache_path}. Re-run with --overwrite to replace it.")

    if cache_path.exists() and overwrite:
        shutil.rmtree(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )

    root = zarr.open(str(cache_path), mode="a")
    latent_array = None
    write_index = 0

    with torch.inference_mode():
        for batch in tqdm(dataloader, total=len(dataloader), desc=f"precomputing {cache_path.stem}"):
            raw_state = torch.cat((batch["obs/workspace_rgb"], batch["action/workspace_rgb"]), dim=2)
            raw_state = raw_state.to(device=pipe.tensor_kwargs["device"], dtype=torch.float32)
            latent_state = pipe.encode(raw_state).contiguous().float().cpu().numpy()

            if latent_array is None:
                latent_shape = (len(dataset),) + latent_state.shape[1:]
                chunk_shape = (min(batch_size, len(dataset)),) + latent_state.shape[1:]
                latent_array = root.create_dataset(
                    "precomputed_video_latents",
                    shape=latent_shape,
                    chunks=chunk_shape,
                    dtype=np.float32,
                )
                root.attrs.update(
                    {
                        "source": "Video2WorldPipeline.encode",
                        "latent_shape": latent_shape[1:],
                        "num_samples": len(dataset),
                    }
                )

            next_index = write_index + latent_state.shape[0]
            latent_array[write_index:next_index] = latent_state
            write_index = next_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute Video2World latents into a zarr sidecar cache.")
    parser.add_argument("--video_model", required=True, help="Path to the Video2World checkpoint to use")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset root directory")
    parser.add_argument(
        "--data_config",
        default="bridge",
        choices=["bridge", "libero", "lerobot"],
        help="Dataset config to use for loading samples",
    )
    parser.add_argument(
        "--split",
        default="both",
        choices=["train", "val", "both"],
        help="Which split(s) to precompute",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for encoding")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing cache")
    parser.add_argument(
        "--enable_guardrail",
        action="store_true",
        help="Enable guardrail loading for the Video2World pipeline (disabled by default for precompute).",
    )
    args = parser.parse_args()

    data_dir = pathlib.Path(args.dataset_path)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    pipe = load_video2world_pipeline(args.video_model, args.device, args.enable_guardrail)
    splits = [True, False] if args.split == "both" else [args.split == "train"]

    for is_train in splits:
        dataset = load_dataset(args.data_config, data_dir, train=is_train)
        cache_path = data_dir / f".precomputed_video_latents_{'train' if is_train else 'val'}.zarr"
        precompute_split(
            pipe=pipe,
            dataset=dataset,
            cache_path=cache_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            overwrite=args.overwrite,
        )
        print(f"Wrote {len(dataset)} latents to {cache_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
