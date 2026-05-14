"""Example integration for cache-backed precomputed Video2World latents.

The actual implementation lives in:
- cosmos_predict2/models/world2action_model.py
- cosmos_predict2/data/action/dataset_action.py
- scripts/precompute_video_embeddings.py
- cosmos_predict2/data/action/precomputed_latents_utils.py
"""

from cosmos_predict2.data.action.precomputed_latents_utils import (
    PRECOMPUTED_LATENTS_KEY,
    has_precomputed_latents,
    load_precomputed_latents,
)


# 1) Model config flag
# --------------------
# Add this field to World2ActionModelConfig:
#     use_precomputed_latents: bool = False
#
# Then enable it in an experiment config:
#     so101_cfg["model"]["config"]["use_precomputed_latents"] = True


# 2) Dataset cache key
# --------------------
# The dataset now looks for a sidecar zarr cache at:
#     .precomputed_video_latents_train.zarr
#     .precomputed_video_latents_val.zarr
#
# Each cache contains one array:
#     PRECOMPUTED_LATENTS_KEY == "precomputed_video_latents"


# 3) Model fast-path
# ------------------
# World2ActionModel.get_crossattn_emb() now does:
#     latent_state = data_batch.get(PRECOMPUTED_LATENTS_KEY)
#     if use_precomputed_latents and latent_state is not None:
#         ... use cached clip latent ...
#     else:
#         ... fall back to video2world_pipe.get_mimic_data_and_condition(data_batch) ...


# 4) Example usage
# ----------------
def example_batch_usage(data_batch, tensor_kwargs):
    if has_precomputed_latents(data_batch):
        latent_state = load_precomputed_latents(
            data_batch,
            device=tensor_kwargs["device"],
            dtype=tensor_kwargs["dtype"],
        )
        print("cached latent shape:", tuple(latent_state.shape) if latent_state is not None else None)
    else:
        print("batch does not contain cached latents")


if __name__ == "__main__":
    print(PRECOMPUTED_LATENTS_KEY)
