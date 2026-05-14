# Precomputing Video2World Latents

The action decoder can skip the frozen Video2World encode pass by loading cached clip latents from a sidecar zarr file.

## What gets cached

Each dataset split gets one cache file:

- `.precomputed_video_latents_train.zarr`
- `.precomputed_video_latents_val.zarr`

Each cache contains a single array named `precomputed_video_latents` with one latent per dataset item.

## How to generate the cache

```bash
cd model
python scripts/precompute_video_embeddings.py \
  --video_model /path/to/video2world_checkpoint.pt \
  --dataset_path /path/to/data/bridge \
  --data_config bridge \
  --split both \
  --batch_size 4
```

The script uses the same Video2World encode path as training, so the cached tensors match the model's latent format.

## How training uses it

1. `MimicDataset` looks for the split-specific cache next to the dataset root.
2. If it exists, `__getitem__` adds `precomputed_video_latents` to each batch.
3. `World2ActionModel.get_crossattn_emb()` uses that tensor when `use_precomputed_latents=True`.
4. If the key is missing, the model falls back to the original encode path.

## Enabling it

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

## Tradeoff

This reduces repeated world-model work at the cost of extra disk usage for the cache. The cached latents are deterministic for a fixed dataset, split, and checkpoint.
