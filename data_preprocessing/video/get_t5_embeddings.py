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

import argparse
import os
import pathlib
import pickle

import numpy as np
import tqdm

from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig
from imaginaire.constants import T5_MODEL_DIR

"""example command
python -m scripts.get_t5_embeddings --dataset_path datasets/hdvila
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute T5 embeddings for text prompts")
    parser.add_argument("--dataset_path", type=str, help="Root path to the dataset")
    parser.add_argument(
        "--max_length",
        type=int,
        help="Maximum length of the text embedding",
    )
    parser.add_argument("--cache_dir", type=str, default=T5_MODEL_DIR, help="Directory to cache the T5 model")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    metas_dir = os.path.join(args.dataset_path, "metas")
    metas_generic_dir = os.path.join(args.dataset_path, "metas_generic")
    metas_list = sorted(map(str, pathlib.Path(metas_dir).glob("*.txt")))[args.rank :: args.world_size]
    metas_generic_list = sorted(map(str, pathlib.Path(metas_generic_dir).glob("*.txt")))[args.rank :: args.world_size]

    t5_xxl_dir = os.path.join(args.dataset_path, "t5_xxl")
    t5_xxl_generic_dir = os.path.join(args.dataset_path, "t5_xxl_generic")
    os.makedirs(t5_xxl_dir, exist_ok=True)
    os.makedirs(t5_xxl_generic_dir, exist_ok=True)

    encoder_config = CosmosT5TextEncoderConfig(ckpt_path=args.cache_dir)
    encoder = CosmosT5TextEncoder(config=encoder_config)

    for meta_filename in tqdm.tqdm(metas_list, total=len(metas_list), desc="computing t5 embeddings"):
        t5_xxl_filename = os.path.join(t5_xxl_dir, os.path.basename(meta_filename).replace(".txt", ".pickle"))
        if os.path.exists(t5_xxl_filename):
            continue

        with open(meta_filename) as fp:
            prompt = fp.read().strip()

        encoded_text, mask_bool = encoder.encode_prompts(
            prompt, max_length=args.max_length, return_mask=True
        )  # list of np.ndarray in (len, embed_dim)
        attn_mask = mask_bool.long()
        lengths = attn_mask.sum(dim=1).cpu()

        encoded_text = encoded_text.cpu().numpy().astype(np.float16)

        # trim zeros to save space
        encoded_text = [encoded_text[batch_id][: lengths[batch_id]] for batch_id in range(encoded_text.shape[0])]

        with open(t5_xxl_filename, "wb") as fp:
            pickle.dump(encoded_text, fp)

    for meta_filename in tqdm.tqdm(metas_generic_list, total=len(metas_generic_list), desc="computing t5 embeddings"):
        t5_xxl_generic_filename = os.path.join(t5_xxl_generic_dir, os.path.basename(meta_filename).replace(".txt", ".pickle"))
        if os.path.exists(t5_xxl_generic_filename):
            continue

        with open(meta_filename) as fp:
            generic_prompt = fp.read().strip()

        encoded_text, mask_bool = encoder.encode_prompts(
            generic_prompt, max_length=args.max_length, return_mask=True
        )  # list of np.ndarray in (len, embed_dim)
        attn_mask = mask_bool.long()
        lengths = attn_mask.sum(dim=1).cpu()

        encoded_text = encoded_text.cpu().numpy().astype(np.float16)

        # trim zeros to save space
        encoded_text = [encoded_text[batch_id][: lengths[batch_id]] for batch_id in range(encoded_text.shape[0])]

        with open(t5_xxl_generic_filename, "wb") as fp:
            pickle.dump(encoded_text, fp)


if __name__ == "__main__":
    args = parse_args()
    main(args)
