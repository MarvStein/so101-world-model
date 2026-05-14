"""Helpers for cache-backed precomputed Video2World latents."""

from __future__ import annotations

import pathlib
from typing import Any

import torch

PRECOMPUTED_LATENTS_KEY = "precomputed_video_latents"


def has_precomputed_latents(data_batch: dict[str, Any]) -> bool:
    return PRECOMPUTED_LATENTS_KEY in data_batch


def load_precomputed_latents(data_batch: dict[str, Any], *, device: str, dtype: torch.dtype) -> torch.Tensor | None:
    latent_state = data_batch.get(PRECOMPUTED_LATENTS_KEY)
    if latent_state is None:
        return None
    if not torch.is_tensor(latent_state):
        latent_state = torch.as_tensor(latent_state)
    return latent_state.to(device=device, dtype=dtype).contiguous()


def default_cache_path(data_dir: str | pathlib.Path, *, train: bool) -> pathlib.Path:
    data_dir = pathlib.Path(data_dir)
    return data_dir / f".precomputed_video_latents_{'train' if train else 'val'}.zarr"
