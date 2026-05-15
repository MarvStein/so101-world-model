# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import pathlib

from hydra.core.config_store import ConfigStore

from cosmos_predict2.configs.config_video2world import (
    get_cosmos_predict2_video2world_pipeline,
)
from cosmos_predict2.configs.defaults.ema import EMAConfig
from cosmos_predict2.models.world2action_model import (
    World2ActionModel as VarNoiseWorld2ActionModel,
)
from cosmos_predict2.models.world2action_model import (
    World2ActionModelConfig as VarNoiseWorld2ActionModelConfig,
)
from imaginaire.lazy_config import LazyCall as L

NON_FINETUNED: dict = {
    "trainer": {"distributed_parallelism": "ddp"},
    "model": L(VarNoiseWorld2ActionModel)(
        config=L(VarNoiseWorld2ActionModelConfig)(
            train_architecture="base",
            lora_rank=16,
            lora_alpha=16,
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            init_lora_weights=True,
            precision="bfloat16",
            loss_reduce="mean",
            loss_scale=10.0,
            ema=EMAConfig(enabled=False),
            action_dit_path="",
            video_dit_path=(
                pathlib.Path(__file__).parents[3]
                / "checkpoints"
                / "video_backbone"
                / "cosmos-predict2_v2w_480p_10fps.pt"
            ),
            pipe_config="${world2action_pipe}",
            video_pipe_config=get_cosmos_predict2_video2world_pipeline(model_size="2B", resolution="480", fps=10),
            use_precomputed_latents=False,
            fsdp_shard_size=0,
            data_config="${data_config}",
        )
    ),
}

VIDEO_MODEL_CKPT_NAMES = [
    "iter_000003000_fused", # TODO: adapt this checkpoint to final finetuned checkpoint and copy it to the video_backbone dir
    "v2w_pretrained_cosmos",
    "v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused",
    "v2w_libero_goal_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007020_fused",
    "v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused",
    "v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused",
]
VIDEO_MODEL_CKPT_DIR = pathlib.Path(__file__).parents[3] / "checkpoints" / "video_backbone"


def register_model() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="model",
        package="_global_",
        name="v2w_pretrained_cosmos",
        node=NON_FINETUNED,
    )

    for name in VIDEO_MODEL_CKPT_NAMES:
        cfg = copy.deepcopy(NON_FINETUNED)
        cfg["model"]["config"]["video_dit_path"] = str((VIDEO_MODEL_CKPT_DIR / f"{name}.pt").resolve())

        cs.store(
            group="model",
            package="_global_",
            name=name,
            node=cfg,
        )
