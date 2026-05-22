import pathlib
import pickle
from collections.abc import Sequence
from typing import TypeVar

import zarr

from cosmos_predict2.data.action.types import NormalizationType

T = TypeVar("T", bound=float)


def get_paths(
    data_dir: pathlib.Path,
    *,
    verbose: bool = False,
) -> list[pathlib.Path]:
    paths_cache = data_dir / "paths.pkl"

    try:
        with paths_cache.open("rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        pass

    paths = sorted(data_dir.glob("**/*.zarr"))

    with paths_cache.open("wb") as f:
        pickle.dump(paths, f)

    return paths


def filter_paths_by_tags(
    paths: list[pathlib.Path],
    tags: list[str],
    *,
    data_dir: pathlib.Path | None = None,
) -> list[pathlib.Path]:
    """Return only those episode paths whose zarr ``dataset_tag`` attribute is in *tags*.

    A tag_map.pkl cache (stored next to paths.pkl in *data_dir*) avoids re-reading
    zarr attributes on every run.  Pass *data_dir* to enable caching; omit it to
    always read attrs from disk.
    """
    cache_path = (data_dir / "tag_map.pkl") if data_dir is not None else None

    tag_map: dict[str, str] = {}
    if cache_path is not None:
        try:
            with cache_path.open("rb") as f:
                tag_map = pickle.load(f)
        except FileNotFoundError:
            pass

    # Build / extend cache for any paths not yet known.
    missing = [p for p in paths if str(p) not in tag_map]
    for p in missing:
        try:
            root = zarr.open(str(p), mode="r")
            tag_map[str(p)] = root.attrs.get("dataset_tag", "")
        except Exception:
            tag_map[str(p)] = ""

    if missing and cache_path is not None:
        with cache_path.open("wb") as f:
            pickle.dump(tag_map, f)

    tag_set = set(tags)
    return [p for p in paths if tag_map.get(str(p), "") in tag_set]


def dict_apply(x: dict, func) -> dict:
    result = {}
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def extract_normalization_types(policy_io: dict[str, dict]) -> dict[str, NormalizationType]:
    return {
        f"{category}/{key}": val["normalization_type"]
        for category, values in policy_io.items()
        for key, val in values.items()
    }


def linear_search_with_initial_guess_left(seq: Sequence[T], target: T, initial_guess: int) -> int:
    if seq[0] >= target:
        return 0
    if seq[-1] <= target:
        return len(seq) - 1

    res = max(0, min(len(seq) - 1, initial_guess))
    while True:
        if seq[res] <= target:
            if seq[res + 1] > target:
                return res
            res += 1
        else:
            res -= 1


def linear_search_with_initial_guess_right(seq: Sequence[T], target: T, initial_guess: int) -> int:
    if seq[0] >= target:
        return 0
    if seq[-1] <= target:
        return len(seq) - 1

    res = max(0, min(len(seq) - 1, initial_guess))
    while True:
        if seq[res] >= target:
            if seq[res - 1] < target:
                return res
            res -= 1
        else:
            res += 1
