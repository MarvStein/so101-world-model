"""
Convert a LeRobot v3 dataset to mimic-video format.

Each episode becomes:
  <repo>/data/task{1|2}/video/episode_NNN.mp4   — re-encoded at target FPS / resolution
  <repo>/data/task{1|2}/metas/episode_NNN.txt   — task description (for T5 embedding)

The script handles the v3 layout where many episodes are concatenated into a
single MP4 shard.  Per-episode start/end are read from the episode metadata
fields `videos/.../from_timestamp` and `videos/.../to_timestamp`.

Usage:
Task 1:
python data_preprocessing/video/process_lerobot_video.py --repo-id klucny/rl_eth --task 1 --default-lang "push the white polyhedron into the goal circle while staying in the corridor"
Task 2:
python data_preprocessing/video/process_lerobot_video.py --repo-id klucny/rl_eth_task2 --task 2 --default-lang "push the white polyhedron into the goal circle while avoiding the red obstacle"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = pathlib.Path(__file__).parents[2]

VIDEO_KEY = "observation.images.front"


def read_episodes_df(raw_dir: pathlib.Path) -> pd.DataFrame:
    dfs = [
        pd.read_parquet(p)
        for p in sorted((raw_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    ]
    return pd.concat(dfs, ignore_index=True).sort_values("episode_index").reset_index(drop=True)


def read_task_map(raw_dir: pathlib.Path) -> dict[int, str]:
    """Returns {task_index -> task_text} from tasks.jsonl or tasks.parquet."""
    tasks_jsonl = raw_dir / "meta" / "tasks.jsonl"
    tasks_parquet = raw_dir / "meta" / "tasks.parquet"

    if tasks_jsonl.exists():
        result = {}
        with open(tasks_jsonl) as fh:
            for line in fh:
                obj = json.loads(line)
                result[int(obj["task_index"])] = str(obj["task"])
        return result

    if tasks_parquet.exists():
        df = pd.read_parquet(tasks_parquet)
        # Row index = task name, column "task_index" = integer id
        return {int(row["task_index"]): str(task_name) for task_name, row in df.iterrows()}

    return {}


def episode_task_text(ep: pd.Series, task_map: dict[int, str], default_lang: str) -> str:
    if default_lang:
        return default_lang

    # ep["tasks"] is an array of task name strings (not indices)
    raw_tasks: list[str] = list(ep["tasks"])
    if not raw_tasks:
        return ""

    # If task_map maps names -> descriptions, use it; otherwise return the name as-is
    name = raw_tasks[0]
    return task_map.get(name, name)  # type: ignore[arg-type]


def extract_episode(
    src: pathlib.Path,
    dst: pathlib.Path,
    from_ts: float,
    to_ts: float,
    target_fps: int,
    width: int,
    height: int,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Add one frame's worth of duration so the last frame is included
    duration = to_ts - from_ts + 1.0 / target_fps

    # -ss before -i for fast keyframe seek; -t is relative to that seek point
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(from_ts),
        "-i", str(src),
        "-t", str(duration),
        "-vf", f"scale={width}:{height},fps={target_fps}",
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for episode (src={src}, from={from_ts}, to={to_ts}):\n"
            + result.stderr.decode(errors="replace")
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="LeRobot v3 -> mimic-video conversion")
    ap.add_argument("--repo-id", default=None, help="HuggingFace dataset repo ID")
    ap.add_argument(
        "--raw-dir",
        type=pathlib.Path,
        default=None,
        help="Local LeRobot dataset root (skips HF download)",
    )
    ap.add_argument("--task", type=int, choices=[1, 2], required=True, help="1 -> data/task1/, 2 -> data/task2/")
    ap.add_argument("--target-fps", type=int, default=10)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument(
        "--default-lang",
        type=str,
        default="",
        help="Task description written to every metas/*.txt (overrides dataset task labels).",
    )
    args = ap.parse_args()

    output_dir = REPO_ROOT / "data" / f"task{args.task}"

    if args.raw_dir is None:
        from huggingface_hub import snapshot_download

        print(f"Downloading {args.repo_id} from HuggingFace Hub...")
        local_dir = snapshot_download(repo_id=args.repo_id, repo_type="dataset")
        raw_dir = pathlib.Path(local_dir)
    else:
        raw_dir = args.raw_dir

    episodes_df = read_episodes_df(raw_dir)
    task_map = read_task_map(raw_dir)
    print(f"Found {len(episodes_df)} episodes.  Task map: {task_map}")

    (output_dir / "video").mkdir(parents=True, exist_ok=True)
    (output_dir / "metas").mkdir(parents=True, exist_ok=True)

    for _, ep in tqdm(episodes_df.iterrows(), total=len(episodes_df), desc="episodes"):
        ep_idx = int(ep["episode_index"])
        chunk_idx = int(ep[f"videos/{VIDEO_KEY}/chunk_index"])
        file_idx = int(ep[f"videos/{VIDEO_KEY}/file_index"])
        from_ts = float(ep[f"videos/{VIDEO_KEY}/from_timestamp"])
        to_ts = float(ep[f"videos/{VIDEO_KEY}/to_timestamp"])

        src_video = (
            raw_dir
            / "videos"
            / VIDEO_KEY
            / f"chunk-{chunk_idx:03d}"
            / f"file-{file_idx:03d}.mp4"
        )
        dst_video = output_dir / "video" / f"episode_{ep_idx:03d}.mp4"

        extract_episode(
            src_video,
            dst_video,
            from_ts,
            to_ts,
            args.target_fps,
            args.width,
            args.height,
        )

        task_text = episode_task_text(ep, task_map, args.default_lang)
        dst_txt = output_dir / "metas" / f"episode_{ep_idx:03d}.txt"
        dst_txt.write_text(task_text, encoding="utf-8")

    print(f"\nDone. {len(episodes_df)} episodes written to {output_dir}")

if __name__ == "__main__":
    main()
