"""
Smoke-test for tag-based filtering in MimicDataset.

Loads the real lerobot dataset via the standard config pipeline and verifies
that providing a ``--tags`` filter reduces the episode count correctly.

Run from the model/ directory:
    python scripts/test_tag_filtering.py
    python scripts/test_tag_filtering.py --tags task1
    python scripts/test_tag_filtering.py --tags task1 task12 task13
"""

import argparse
import pathlib

import hydra
import tqdm
import zarr

from cosmos_predict2.configs.defaults.data_action import get_data_config
from cosmos_predict2.data.action.utils import filter_paths_by_tags, get_paths


DEFAULT_DATA_DIR = pathlib.Path("/home/nvidia/projects/so101-world-model/data/action/lerobot")


def print_tag_distribution(data_dir: pathlib.Path) -> dict[str, int]:
    all_paths = get_paths(data_dir)
    print(f"Found {len(all_paths)} total episodes in {data_dir}\n")

    tag_counts: dict[str, int] = {}
    for p in tqdm.tqdm(all_paths, desc="Reading tags"):
        try:
            root = zarr.open(str(p), mode="r")
            tag = root.attrs.get("dataset_tag", "") or "<none>"
            if tag == "<none>":
                print(f"WARNING: episode {p} has no dataset_tag!")
        except Exception:
            tag = "<error>"
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print("Tag distribution:")
    col = max(len(t) for t in tag_counts) + 2
    for tag, count in sorted(tag_counts.items()):
        bar = "#" * count
        print(f"  {tag:<{col}}  {count:4d}  {bar}")

    return tag_counts


def instantiate_dataset(tags: list[str] | None, train: bool = True):
    """Instantiate MimicDataset via the standard lerobot config, with optional tag filter."""
    cfg = get_data_config("lerobot")
    return hydra.utils.instantiate(
        cfg.dataset.dataset,
        train=train,
        verbose=True,
        tags=tags,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory containing episode_*.zarr files.",
    )
    ap.add_argument(
        "--tags",
        nargs="+",
        default=None,
        metavar="TAG",
        help="Tag filter to test (e.g. --tags task1 task12).",
    )
    args = ap.parse_args()

    if not args.data_dir.exists():
        print(f"Data directory not found: {args.data_dir}")
        raise SystemExit(1)

    # 1. Show tag distribution.
    tag_counts = print_tag_distribution(args.data_dir)

    any_untagged = tag_counts.get("<none>", 0) > 0
    if any_untagged:
        print(
            f"\nWARNING: {tag_counts['<none>']} episode(s) have no dataset_tag."
            " Re-run process_lerobot.py to tag all zarr files."
        )

    # 2. Instantiate unfiltered dataset and report size.
    print("\n--- Unfiltered MimicDataset (train) ---")
    ds_all = instantiate_dataset(tags=None, train=True)
    print(f"  len(dataset) = {len(ds_all)}")

    # 3. If --tags provided, instantiate filtered dataset and compare.
    if args.tags:
        print(f"\n--- MimicDataset filtered to tags={args.tags} (train) ---")
        ds_filtered = instantiate_dataset(tags=args.tags, train=True)
        print(f"  len(dataset) = {len(ds_filtered)}")

        expected = sum(tag_counts.get(t, 0) for t in args.tags)
        print(
            f"\n  Episodes with requested tags: ~{expected} "
            f"(exact dataset len may differ due to val split and chunk counting)"
        )

        assert len(ds_filtered) <= len(ds_all), (
            f"Filtered dataset ({len(ds_filtered)}) is larger than unfiltered ({len(ds_all)})"
        )
        assert len(ds_filtered) > 0, "Filtered dataset is empty — check that the tags exist"
        print("\nPASS: filtered dataset is smaller than unfiltered and non-empty.")


if __name__ == "__main__":
    main()
