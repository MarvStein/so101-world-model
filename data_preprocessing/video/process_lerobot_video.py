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
import pathlib
import subprocess
import sys
import imageio_ffmpeg

from huggingface_hub import snapshot_download

import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = pathlib.Path(__file__).parents[2]

sys.path.insert(0, str(REPO_ROOT / "data_preprocessing"))
from video_config import SO101_VIDEO_CONFIG  # noqa: E402
from dataset_specs import DATASET_SPECS  # noqa: E402

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
    duration = to_ts - from_ts + 1.0 / SO101_VIDEO_CONFIG.fps

    # -ss before -i for fast keyframe seek; -t is relative to that seek point
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y",
        "-ss", str(from_ts),
        "-i", str(src),
        "-t", str(duration),
        "-vf", SO101_VIDEO_CONFIG.ffmpeg_vf,
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
    output_dir = REPO_ROOT / "data" / "video_fine" / "lerobot"
    (output_dir / "video").mkdir(parents=True, exist_ok=True)
    (output_dir / "metas").mkdir(parents=True, exist_ok=True)

    episode_offset = 0
    for repo_id, task_id in DATASET_SPECS:
        print(f"Downloading {repo_id} from HuggingFace Hub...")
        raw_dir = pathlib.Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))

        episodes_df = read_episodes_df(raw_dir)
        print(f"Found {len(episodes_df)} episodes in {repo_id}.")

        task_text = episode_task_text(task_id)

        for _, ep in tqdm(episodes_df.iterrows(), total=len(episodes_df), desc=repo_id):
            src_ep_idx = int(ep["episode_index"])
            out_ep_idx = src_ep_idx + episode_offset

            src_video = (
                raw_dir
                / "videos"
                / VIDEO_KEY
                / f"chunk-{int(ep[f'videos/{VIDEO_KEY}/chunk_index']):03d}"
                / f"file-{int(ep[f'videos/{VIDEO_KEY}/file_index']):03d}.mp4"
            )
            dst_video = output_dir / "video" / f"episode_{out_ep_idx:03d}.mp4"

            extract_episode(
                src_video,
                dst_video,
                float(ep[f"videos/{VIDEO_KEY}/from_timestamp"]),
                float(ep[f"videos/{VIDEO_KEY}/to_timestamp"]),
            )

            dst_txt = output_dir / "metas" / f"episode_{out_ep_idx:03d}.txt"
            dst_txt.write_text(task_text, encoding="utf-8")

        episode_offset += len(episodes_df)

    print(f"\nDone. {episode_offset} episodes written to {output_dir}")


if __name__ == "__main__":
    main()
