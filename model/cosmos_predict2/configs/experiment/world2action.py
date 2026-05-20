import copy
import itertools as it

import numpy as np
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from omegaconf import MISSING

from cosmos_predict2.configs.defaults.data_action import DATA_CONFIGS
from cosmos_predict2.configs.defaults.world2action_model import VIDEO_MODEL_CKPT_NAMES
from cosmos_predict2.configs.defaults.world2action_pipe import ACTION_DECODER_NETS
from imaginaire.lazy_config import LazyCall as L

BASE: dict = dict(
    defaults=[
        {"override /model": MISSING},
        {"override /world2action_pipe": MISSING},
        {"override /data_config": MISSING},
        {"override /optimizer": "fusedadamw"},
        {"override /ckpt_type": "standard"},
        {"override /dataloader_val": "mimic"},
        {"override /dataloader_train": "mimic"},
        {"override /scheduler": "lambdalinear"},
        "_self_",
    ],
    model=dict(
        config=dict(
            train_architecture="base",
            # video_sigma_mode="logitnormal",
            pipe_config=dict(xattn_layer_idx=MISSING),
            video_pipe_config=dict(guardrail_config=dict(enabled=False)),
        )
    ),
    optimizer=dict(
        lr=MISSING,
    ),
    scheduler=dict(
        f_max=[1],
        f_min=[0.1],
        warm_up_steps=[500],
        cycle_lengths=[50_000],
    ),
    job=dict(
        project="vam",
        group=MISSING,
        name=MISSING,
    ),
    model_parallel=dict(
        cpu_offloading_activations=False,
        cpu_offloading_weights=False,
    ),
    checkpoint=dict(save_iter=100),
    trainer=dict(
        distributed_parallelism="ddp",
        grad_accum_iter=1,
        max_iter=500_000,
        logging_iter=20,
        validation_iter=1_000,
        run_validation=True,
    ),
)

cs = ConfigStore.instance()
cs.store(name="config", node=BASE)

world2action_pipes = ACTION_DECODER_NETS.keys()
xattn_layer_idxs = [20]
lrs = np.logspace(-5, -3, 9)[[4]]
bszs = [1, 128, 256]


def get_local_batch_size(global_bsz: int) -> int:
    res = global_bsz / parallel_state.get_data_parallel_world_size()

    if not res.is_integer():
        msg = "That batch size doesn't work with the number of gpus you have."
        raise ValueError(msg)

    return int(res)


for video_ckpt, data_config, xattn_layer_idx, lr, bsz in it.product(
    VIDEO_MODEL_CKPT_NAMES, DATA_CONFIGS.keys(), xattn_layer_idxs, lrs, bszs
):
    pipes = [pipe for pipe in world2action_pipes if data_config.startswith(pipe)]
    if not pipes:
        continue
    if len(pipes) > 1:
        raise AssertionError("data_config to pipe should be n-to-1")
    pipe = pipes[0]

    exp_name = f"w2a_{data_config}_{video_ckpt}_lr{lr:.3e}_layer{xattn_layer_idx}_bsz{bsz}"

    cfg = copy.deepcopy(BASE)
    cfg["defaults"][0]["override /model"] = video_ckpt
    cfg["defaults"][1]["override /world2action_pipe"] = pipe
    cfg["defaults"][2]["override /data_config"] = data_config
    cfg["model"]["config"]["pipe_config"]["xattn_layer_idx"] = xattn_layer_idx
    cfg["optimizer"]["lr"] = lr.item()
    cfg["job"]["group"] = pipe
    cfg["job"]["name"] = exp_name
    cfg["dataloader_train"] = {"batch_size": L(get_local_batch_size)(global_bsz=bsz)}

    if "libero" in data_config:
        cfg["checkpoint"]["save_iter"] = 99999999
        cfg["trainer"]["run_validation"] = False

    cs.store(
        group="experiment",
        package="_global_",
        name=exp_name,
        node=cfg,
    )

# Add manual lerobot experiment
lerobot_cfg = copy.deepcopy(BASE)
lerobot_cfg["defaults"][0]["override /model"] = "v2w_iter_000011000_fused" # register this model in model/cosmos_predict2/configs/defaults/world2action_model.py
lerobot_cfg["defaults"][1]["override /world2action_pipe"] = "lerobot"
lerobot_cfg["defaults"][2]["override /data_config"] = "lerobot"
lerobot_cfg["model"]["config"]["pipe_config"]["xattn_layer_idx"] = 20
lerobot_cfg["optimizer"]["lr"] = 1e-4
lerobot_cfg["job"]["group"] = "lerobot"
lerobot_cfg["job"]["name"] = "w2a_lerobot_v2w_11k_lr1e-04_bs4_task1"
lerobot_cfg["dataloader_train"] = {"batch_size": L(get_local_batch_size)(global_bsz=4)}
lerobot_cfg["trainer"]["run_validation"] = False


cs.store(
    group="experiment",
    package="_global_",
    name="w2a_lerobot_v2w_11k_lr1e-04_bs4_task1",
    node=lerobot_cfg,
)

# TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 CUDA_DEVICE_MAX_CONNECTIONS=1 NVTE_FUSED_ATTN=0 torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/config.py -- experiment=...
