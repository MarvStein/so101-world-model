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

from typing import Any

import attrs

from cosmos_predict2.configs.defaults.callbacks import register_callbacks
from cosmos_predict2.configs.defaults.checkpoint import register_checkpoint
from cosmos_predict2.configs.defaults.data_action import register_training_and_val_action_data
from cosmos_predict2.configs.defaults.data_video import register_training_and_val_video_data
from cosmos_predict2.configs.defaults.optimizer import register_optimizer
from cosmos_predict2.configs.defaults.scheduler import register_scheduler
from cosmos_predict2.configs.defaults.video2world_model import register_model as register_video_model
from cosmos_predict2.configs.defaults.world2action_model import register_model as register_action_model
from cosmos_predict2.configs.defaults.world2action_pipe import register_pipe
from imaginaire import config
from imaginaire.utils import log
from imaginaire.utils.config_helper import import_all_modules_from_package


@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    # see config groups in registry.py
    defaults: list[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_config": None},
            {"video_dataset_train": None},
            {"video_dataset_val": None},
            {"dataloader_train": None},
            {"dataloader_val": None},
            {"world2action_pipe": None},
            {"optimizer": "fusedadamw"},
            {"scheduler": "constant"},
            {"model": None},
            {"callbacks": ["basic"]},
            {"net": None},
            {"ema": None},
            {"checkpoint": None},
            {"ckpt_type": None},
            # the list is with order, we need global experiment to be the last one
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(
        model=None,
        world2action_pipe=None,
        optimizer=None,
        scheduler=None,
        data_config=None,
        video_dataset_train=None,
        video_dataset_val=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Call this function to register config groups for advanced overriding. the order follows the default config groups
    register_training_and_val_action_data()
    register_training_and_val_video_data()
    register_optimizer()
    register_scheduler()
    register_action_model()
    register_video_model()
    register_pipe()

    register_checkpoint()
    register_callbacks()

    # experiment config are defined in the experiment folder
    # call import_all_modules_from_package to register them
    import_all_modules_from_package("cosmos_predict2.configs.experiment", reload=True)
    # import_all_modules_from_package("cosmos_predict2.configs.experiment.multiview", reload=True)
    # import_all_modules_from_package("cosmos_predict2.configs.action_conditioned.experiment", reload=True)
    log.critical("Finished importing config modules.")
    return c
