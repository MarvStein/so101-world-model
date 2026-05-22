"""Precompute T5 text embeddings for all SO101 tasks.

Run this script once on a machine that has the T5-11B model available
(e.g. the GPU training instance).  The resulting .pt file (~6 MB) is then
copied to the deployment machine so the T5 model does not have to be
downloaded there.

Usage (model venv)
------------------
    cd /path/to/so101-world-model
    source model/.venv/bin/activate
    PYTHONPATH=model python eval/so101/precompute_task_embeddings.py \\
        --output_path eval/so101/task_embeddings.pt

The output file is a dict saved with torch.save:
    {
        "1":  torch.Tensor of shape (1, 512, 1024) float16,
        "2":  ...,
        "12": ...,
        "13": ...,
        "22": ...,
        "23": ...,
    }
Keys are the numeric suffix of the corresponding description_task{id}.txt file.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

import torch

# ---------------------------------------------------------------------------
# Make sure the model package is importable when invoked from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(_REPO_ROOT / "model"))

from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig  # noqa: E402
from imaginaire.constants import T5_MODEL_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Precompute T5 embeddings for all SO101 task descriptions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output_path",
        default=str(_REPO_ROOT / "eval" / "so101" / "task_embeddings.pt"),
        help="Where to write the output .pt file.",
    )
    p.add_argument(
        "--tasks_root",
        default=str(_REPO_ROOT),
        help=(
            "Directory containing description_task*.txt files. "
            "All matching files are processed automatically."
        ),
    )
    p.add_argument(
        "--t5_model_dir",
        default=T5_MODEL_DIR,
        help="Path to the local T5-11B model checkpoint directory.",
    )
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    tasks_root = pathlib.Path(args.tasks_root)
    task_files = sorted(tasks_root.glob("description_task*.txt"))

    if not task_files:
        print(f"[precompute] ERROR: no description_task*.txt files found in {tasks_root}")
        sys.exit(1)

    # Extract numeric IDs from filenames, e.g. "description_task13.txt" → "13"
    _id_re = re.compile(r"description_task(\d+)\.txt$")
    tasks: dict[str, str] = {}
    for f in task_files:
        m = _id_re.match(f.name)
        if m is None:
            print(f"[precompute] WARNING: skipping unexpected filename: {f.name}")
            continue
        task_id = m.group(1)
        description = f.read_text(encoding="utf-8").strip()
        tasks[task_id] = description
        print(f"[precompute] Found task {task_id!r}: {description[:80]}...")

    print(f"\n[precompute] Loading T5 encoder from {args.t5_model_dir} ...")
    encoder_config = CosmosT5TextEncoderConfig(ckpt_path=args.t5_model_dir)
    encoder = CosmosT5TextEncoder(config=encoder_config, device="cuda")

    embeddings: dict[str, torch.Tensor] = {}
    for task_id, description in tasks.items():
        print(f"[precompute] Encoding task {task_id!r} ...")
        # encode_prompts returns shape (1, max_length, 1024) when given a single string.
        # max_length defaults to NUM_EMBEDDING_PADDING_TOKENS (512).
        emb: torch.Tensor = encoder.encode_prompts(description)  # (1, 512, 1024) bfloat16
        # Store as float16 to halve storage; pipeline casts to bfloat16 on the fly.
        embeddings[task_id] = emb.cpu().to(torch.float16)
        print(f"[precompute]   → shape {tuple(emb.shape)}, dtype {embeddings[task_id].dtype}")

    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    print(f"\n[precompute] Saved embeddings for tasks {sorted(embeddings)} to {output_path}")
    print(f"[precompute] File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main(parse_args())
