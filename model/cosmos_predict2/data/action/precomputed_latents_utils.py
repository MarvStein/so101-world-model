"""Helpers for precomputed Video2World latents stored inside episode zarrs."""

from __future__ import annotations

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
