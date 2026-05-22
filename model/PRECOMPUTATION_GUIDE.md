# Precomputing Video2World Latents

The action decoder can skip the frozen Video2World encode pass at training time by loading precomputed clip latents stored directly inside each episode zarr.

## What gets cached

For each episode, a new array is written into the existing zarr:

```
episode_000.zarr/precomputed_video_latents   shape (n_valid_steps, 16, 16, 60, 80)  float32
episode_001.zarr/precomputed_video_latents   ...
...
```

`n_valid_steps` is the number of valid anchor timesteps in that episode (as determined by the dataset config). Each entry `[step_idx]` is the encoded latent for the video clip anchored at that timestep — the concatenation of the obs window (5 frames) and the action window (56 frames).

Because latents are stored per-episode, they are **split-independent**: the train/val split is applied only at training load time and re-running precomputation is never needed just because the split changes.

## How to generate

**Single GPU:**
```bash
cd model
python scripts/precompute_video_embeddings.py \
  --video_model /path/to/video2world_checkpoint.pt \
  --dataset_path /path/to/data/lerobot \
  --data_config lerobot
```

**Two GPUs (when the model doesn't fit on one GPU):**
```bash
cd model
torchrun --nproc_per_node=2 -m scripts.precompute_video_embeddings \
  --video_model /path/to/video2world_checkpoint.pt \
  --dataset_path /path/to/data/lerobot \
  --data_config lerobot \
  --num_gpus 2
```

With `--num_gpus 2` the video tokenizer uses **context parallelism**: each sample's 61-frame clip is split temporally across both GPUs, halving peak VRAM. All ranks collaborate on encoding; only rank 0 writes zarr files.

The script always encodes one sample at a time (`batch_size=1`). Pass `--overwrite` to replace existing latents.

## How training uses it

1. `ChunkReader.read_chunk` opens the episode zarr and reads `precomputed_video_latents[step_idx]` alongside the normal data components — no second zarr open.
2. If the array is absent for an episode (e.g. partial precomputation), the key is simply not added to the batch.
3. `World2ActionModel.get_crossattn_emb()` uses the cached tensor when `use_precomputed_latents=True`; if the key is missing it falls back to on-the-fly encoding.

## Enabling the fast path

Set the model flag in your experiment config:

```python
so101_cfg["model"]["config"]["use_precomputed_latents"] = True
```

## Validation

Check that a batch contains cached latents:

```python
from cosmos_predict2.data.action.precomputed_latents_utils import has_precomputed_latents

assert has_precomputed_latents(batch)
```

Inspect a single episode's latents directly:

```python
import zarr
root = zarr.open("data/lerobot/episode_000.zarr", "r")
print(root["precomputed_video_latents"].shape)  # (n_valid_steps, 16, 16, 60, 80)
```

## Tradeoff

Reduces repeated world-model encode work at the cost of extra disk space inside the episode zarrs (~23 MB per episode at 200 steps). Partial precomputation is safe — episodes without latents fall back to on-the-fly encoding transparently.
