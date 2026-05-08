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

"""Cosmos Predict2 package configuration."""

import argparse
import enum
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Literal


def print_environment_info(args: argparse.Namespace):
    from imaginaire.utils import log

    try:
        git_branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD", shell=True, text=True).strip()
        git_revision = subprocess.check_output("git rev-parse HEAD", shell=True, text=True).strip()
        log.info(f"git.branch: {git_branch}")
        log.info(f"git.revision: {git_revision}")
    except Exception:
        pass

    # Don't print environment variables, since it can contain sensitive information.
    log.info(f"imaginaire.constants: {_args}")
    log.info(f"sys.argv: {sys.argv}")
    log.info(f"args: {args}")


class TextEncoderClass(str, enum.Enum):
    T5 = "t5"


_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument(
    "--checkpoints",
    default=str((pathlib.Path(__file__).parents[1] / "checkpoints").resolve()),
    help="Path to the checkpoints directory",
)
_args = shlex.split(os.environ.get("COSMOS_PREDICT2_ARGS", ""))
_args = _parser.parse_args(_args)


# Feature flags
TEXT_ENCODER_CLASS: TextEncoderClass = TextEncoderClass.T5

# Checkpoints
CHECKPOINTS_DIR: str = _args.checkpoints

T5_MODEL_DIR = f"{CHECKPOINTS_DIR}/text_encoder/t5-11b"

LLAMA_GUARD3_MODEL_DIR = f"{CHECKPOINTS_DIR}/meta-llama/Llama-Guard-3-8B"

COSMOS_GUARDRAIL1_MODEL_DIR = f"{CHECKPOINTS_DIR}/nvidia/Cosmos-Guardrail1"


CosmosPredict2Video2WorldModelSize = Literal["2B"]
CosmosPredict2Video2WorldResolution = Literal["480"]
CosmosPredict2Video2WorldFPS = Literal[10]
CosmosPredict2Video2WorldAspectRatio = Literal["4:3"]
CosmosPredict2Video2WorldModelType = Literal["Video2World"]


def _get_cosmos_predict2_video2world_model_dir(
    *,
    model_size: CosmosPredict2Video2WorldModelSize,
    model_type: CosmosPredict2Video2WorldModelType = "Video2World",
) -> str:
    assert model_size == "2B"
    assert model_type == "Video2World"
    return f"{CHECKPOINTS_DIR}/video_backbone"


def get_cosmos_predict2_video2world_tokenizer(
    *,
    model_size: CosmosPredict2Video2WorldModelSize,
    model_type: CosmosPredict2Video2WorldModelType = "Video2World",
) -> str:
    model_dir = _get_cosmos_predict2_video2world_model_dir(model_size=model_size, model_type=model_type)
    return f"{model_dir}/tokenizer/tokenizer.pth"


def get_cosmos_predict2_video2world_checkpoint(
    *,
    model_size: CosmosPredict2Video2WorldModelSize = "2B",
    model_type: CosmosPredict2Video2WorldModelType = "Video2World",
    resolution: CosmosPredict2Video2WorldResolution = "480",
    fps: CosmosPredict2Video2WorldFPS = 10,
    aspect_ratio: CosmosPredict2Video2WorldAspectRatio = "4:3",
) -> str:
    model_dir = _get_cosmos_predict2_video2world_model_dir(model_size=model_size, model_type=model_type)
    assert fps == 10
    assert aspect_ratio == "4:3"
    return f"{model_dir}/v2w_pretrained_cosmos.pt"
