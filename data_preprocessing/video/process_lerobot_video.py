"""
Convert a LeRobot v3 dataset to mimic-video format.

Each episode becomes:
  <repo>/data/lerobot/video/episode_NNN.mp4   — re-encoded at target FPS / resolution
  <repo>/data/lerobot/metas/episode_NNN.txt   — task description (for T5 embedding)

The script handles the v3 layout where many episodes are concatenated into a
single MP4 shard.  Per-episode start/end are read from the episode metadata
fields `videos/.../from_timestamp` and `videos/.../to_timestamp`.

Usage:
python data_preprocessing/video/process_lerobot_video.py
"""

from __future__ import annotations
from huggingface_hub import snapshot_download
from itertools import islice



import argparse
import json
import pathlib
import subprocess
import imageio_ffmpeg

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


def episode_task_text(task: int) -> str:
    txt_path = REPO_ROOT / f"description_task{task}.txt"
    return txt_path.read_text(encoding="utf-8").strip()


def extract_episode(
    src: pathlib.Path,
    dst: pathlib.Path,
    from_ts: float,
    to_ts: float,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Add one frame's worth of duration so the last frame is included
    duration = to_ts - from_ts + 1.0 / 10

    # -ss before -i for fast keyframe seek; -t is relative to that seek point
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y",
        "-ss", str(from_ts),
        "-i", str(src),
        "-t", str(duration),
        "-vf", "crop=1030:950:400:0,fps=10",
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
    output_dir = REPO_ROOT / "data" / "lerobot"

    print(f"Downloading klucny/rl_eth from HuggingFace Hub...")
    local_dir = snapshot_download(repo_id="klucny/rl_eth", repo_type="dataset")
    raw_dir1 = pathlib.Path(local_dir)

    episodes_df1 = read_episodes_df(raw_dir1)
    print(f"Found {len(episodes_df1)} episodes.")

    print(f"Downloading klucny/rl_eth_task2 from HuggingFace Hub...")
    local_dir = snapshot_download(repo_id="klucny/rl_eth_task2", repo_type="dataset")
    raw_dir2 = pathlib.Path(local_dir)

    episodes_df2 = read_episodes_df(raw_dir2)
    print(f"Found {len(episodes_df2)} episodes.")

    (output_dir / "video").mkdir(parents=True, exist_ok=True)
    (output_dir / "metas").mkdir(parents=True, exist_ok=True)

    for _, ep in tqdm(episodes_df1.iterrows(), total=len(episodes_df1), desc="episodes"):
        ep_idx = int(ep["episode_index"])
        chunk_idx = int(ep[f"videos/{VIDEO_KEY}/chunk_index"])
        file_idx = int(ep[f"videos/{VIDEO_KEY}/file_index"])
        from_ts = float(ep[f"videos/{VIDEO_KEY}/from_timestamp"])
        to_ts = float(ep[f"videos/{VIDEO_KEY}/to_timestamp"])

        src_video = (
            raw_dir1
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
        )

        task_text = episode_task_text(1)
        dst_txt = output_dir / "metas" / f"episode_{ep_idx:03d}.txt"
        dst_txt.write_text(task_text, encoding="utf-8")
    
    for _, ep in tqdm(episodes_df2.iterrows(), total=len(episodes_df2), desc="episodes"):
        ep_idx = int(ep["episode_index"])+len(episodes_df1)
        chunk_idx = int(ep[f"videos/{VIDEO_KEY}/chunk_index"])
        file_idx = int(ep[f"videos/{VIDEO_KEY}/file_index"])
        from_ts = float(ep[f"videos/{VIDEO_KEY}/from_timestamp"])
        to_ts = float(ep[f"videos/{VIDEO_KEY}/to_timestamp"])

        src_video = (
            raw_dir2
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
        )

        task_text = episode_task_text(2)
        dst_txt = output_dir / "metas" / f"episode_{ep_idx:03d}.txt"
        dst_txt.write_text(task_text, encoding="utf-8")

    print(f"\nDone. {len(episodes_df1)+len(episodes_df2)} episodes written to {output_dir}")

if __name__ == "__main__":
    main()
