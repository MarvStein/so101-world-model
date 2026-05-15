import huggingface_hub
from huggingface_hub import snapshot_download
import argparse
import logging
import pathlib
import pickle
import re
import sys
from typing import Literal
import pandas as pd

import numpy as np
import tqdm
import zarr
from numcodecs import Blosc
from PIL import Image
import imageio.v3 as iio
import cv2

# Make data_preprocessing/ importable so we can share VideoTransformConfig
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from video_config import SO101_VIDEO_CONFIG  # noqa: E402

S_TO_NS = 1_000_000_000
REPO_ROOT = pathlib.Path(__file__).parents[2]

# Match the transform in data_preprocessing/video/process_lerobot_video.py:
#   -vf "crop=<crop_w>:<crop_h>:<crop_x>:<crop_y>,fps=<fps>"
# All values come from SO101_VIDEO_CONFIG (data_preprocessing/video_config.py).
# IMPORTANT: If you change them there, you must re-run this script and retrain!
VIDEO_CROP_W = SO101_VIDEO_CONFIG.crop_w
VIDEO_CROP_H = SO101_VIDEO_CONFIG.crop_h
VIDEO_CROP_X = SO101_VIDEO_CONFIG.crop_x
VIDEO_CROP_Y = SO101_VIDEO_CONFIG.crop_y
VIDEO_TARGET_H = SO101_VIDEO_CONFIG.target_h
VIDEO_TARGET_W = SO101_VIDEO_CONFIG.target_w


def episode_task_text(task_id: int) -> str:
    txt_path = REPO_ROOT / f"description_task{task_id}.txt"
    return txt_path.read_text(encoding="utf-8").strip()


def extract_task_id(ep: pd.Series) -> int | None:
    if "tasks" not in ep:
        return None

    raw = ep["tasks"]
    if isinstance(raw, np.ndarray):
        candidates = raw.tolist()
    elif isinstance(raw, (list, tuple, set)):
        candidates = list(raw)
    else:
        candidates = [raw]

    for value in candidates:
        match = re.search(r"task\s*_?(\d+)", str(value), re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def read_episodes_df(raw_dir: pathlib.Path) -> pd.DataFrame:
    dfs = [
        pd.read_parquet(p)
        for p in sorted((raw_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    ]
    return pd.concat(dfs, ignore_index=True).sort_values("episode_index").reset_index(drop=True)

def process_images(
    video_path: pathlib.Path,
    start_s: float | None = None,
    end_s: float | None = None,
    target_len: int | None = None,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_idx = 0
    end_idx = total_frames

    if fps > 0 and start_s is not None:
        start_idx = int(round(start_s * fps))
    if fps > 0 and end_s is not None:
        end_idx = int(round(end_s * fps))

    start_idx = max(0, min(start_idx, total_frames))
    end_idx = max(start_idx, min(end_idx, total_frames))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    frames = []
    for _ in range(start_idx, end_idx):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Keep action preprocessing consistent with video preprocessing crop.
        h, w = frame.shape[:2]
        x0 = max(0, min(VIDEO_CROP_X, w))
        y0 = max(0, min(VIDEO_CROP_Y, h))
        x1 = max(x0, min(x0 + VIDEO_CROP_W, w))
        y1 = max(y0, min(y0 + VIDEO_CROP_H, h))
        frame = frame[y0:y1, x0:x1]
        frame = cv2.resize(frame, (VIDEO_TARGET_W, VIDEO_TARGET_H), interpolation=cv2.INTER_AREA)

        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(
            f"No frames decoded for {video_path} in range "
            f"[{start_s}, {end_s}] (frames [{start_idx}, {end_idx}))"
        )

    if target_len is not None and target_len > 0 and len(frames) != target_len:
        logging.warning(
            "Frame count mismatch for %s in [%s, %s]: decoded=%d, target=%d. "
            "Resampling frames to target length.",
            video_path,
            start_s,
            end_s,
            len(frames),
            target_len,
        )
        sample_idx = np.linspace(0, len(frames) - 1, target_len)
        sample_idx = np.round(sample_idx).astype(np.int64)
        frames = [frames[i] for i in sample_idx]

    return np.stack(frames).astype(np.uint8)


def make_zarr(
    raw_dir,
    output_dir,
    ep,
    default_lang="",
    convert_degrees_to_radians=True,
    output_episode_idx: int | None = None,
):
    data_path = (
        raw_dir
        / f"data/"
          f"chunk-{int(ep['data/chunk_index']):03d}/"
          f"file-{int(ep['data/file_index']):03d}.parquet"
    )

    data_df = pd.read_parquet(data_path)

    # Keep only this episode
    source_ep_idx = int(ep["episode_index"])
    data_df = data_df[data_df["episode_index"] == source_ep_idx]

    # Restrict to episode index range if available
    if "dataset_from_index" in ep and "dataset_to_index" in ep:
        start = int(ep["dataset_from_index"])
        end = int(ep["dataset_to_index"])

        data_df = data_df[
            (data_df["index"] >= start)
            & (data_df["index"] < end)
        ]

    data_df = data_df.sort_values("frame_index").reset_index(drop=True)

    video_path = (
        raw_dir
        / f"videos/observation.images.front/"
          f"chunk-{int(ep['videos/observation.images.front/chunk_index']):03d}/"
          f"file-{int(ep['videos/observation.images.front/file_index']):03d}.mp4"
    )

    from_ts = float(ep["videos/observation.images.front/from_timestamp"]) if "videos/observation.images.front/from_timestamp" in ep else None
    to_ts = float(ep["videos/observation.images.front/to_timestamp"]) if "videos/observation.images.front/to_timestamp" in ep else None

    images = process_images(
        video_path=video_path,
        start_s=from_ts,
        end_s=to_ts,
        target_len=len(data_df),
    )

    states = np.stack(
        data_df["observation.state"].to_numpy()
    ).astype(np.float32)

    actions = np.stack(
        data_df["action"].to_numpy()
    ).astype(np.float32)

    # SO-101 order:
    # 0 shoulder_pan.pos
    # 1 shoulder_lift.pos
    # 2 elbow_flex.pos
    # 3 wrist_flex.pos
    # 4 wrist_roll.pos
    # 5 gripper.pos

    if convert_degrees_to_radians:
        states[:, :5] = np.deg2rad(states[:, :5])
        actions[:, :5] = np.deg2rad(actions[:, :5])

    # Ignore gripper for pushing tasks
    joint_state_lowdim = states[:, :5]    # [T, 5]
    joint_action_lowdim = actions[:, :5]  # [T, 5]

    timestamps_s = data_df["timestamp"].to_numpy().astype(np.float32)

    # Match Bridge convention
    timestamps_ns = (
        (timestamps_s - timestamps_s[0]) * S_TO_NS
    ).astype(np.uint64)

    # Validate all lengths match
    assert (
        images.shape[0]
        == joint_state_lowdim.shape[0]
        == joint_action_lowdim.shape[0]
        == timestamps_ns.shape[0]
    ), (
        f"Length mismatch:\n"
        f"images={images.shape[0]}\n"
        f"states={joint_state_lowdim.shape[0]}\n"
        f"actions={joint_action_lowdim.shape[0]}\n"
        f"timestamps={timestamps_ns.shape[0]}"
    )

    print(f"Read episode {source_ep_idx} from {data_path}")
    print(f"video:   {images.shape}")

    print(
        f"states:  {joint_state_lowdim.shape}, "
        f"min={joint_state_lowdim.min():.4f}, "
        f"max={joint_state_lowdim.max():.4f}"
    )

    print(
        f"actions: {joint_action_lowdim.shape}, "
        f"min={joint_action_lowdim.min():.4f}, "
        f"max={joint_action_lowdim.max():.4f}"
    )

    print(
        f"timestamps: {timestamps_ns.shape}, "
        f"first={timestamps_ns[0]}, "
        f"last={timestamps_ns[-1]}"
    )

    out_ep_idx = source_ep_idx if output_episode_idx is None else output_episode_idx

    #make a dir inside output_dir/lerobot for current episode index if it doesn't exist
    out_path = output_dir / "lerobot" / f"episode_{out_ep_idx:03d}"
    out_path.mkdir(parents=True, exist_ok=True)

    lang = default_lang
    if not lang:
        task_id = extract_task_id(ep)
        if task_id is not None:
            try:
                lang = episode_task_text(task_id)
            except FileNotFoundError:
                logging.warning(
                    "No description file found for task%d at %s. Falling back to --default-lang.",
                    task_id,
                    REPO_ROOT / f"description_task{task_id}.txt",
                )

    t_img = min(65, timestamps_ns.shape[0])
    t_ld = min(1024, timestamps_ns.shape[0])

    root: zarr.Group
    with zarr.open(str(out_path), mode="w") as root:
        root.create_dataset(
            "workspace_rgb",
            shape=images.shape,
            dtype=np.uint8,
            chunks=(t_img, *images.shape[1:]),
            compressor=Blosc(cname="lz4", clevel=9, shuffle=Blosc.BITSHUFFLE),
        )
        root["workspace_rgb"][...] = images
        root.create_dataset(
            "workspace_rgb_timestamps",
            shape=(len(timestamps_ns),),
            dtype="uint64",
            chunks=(len(timestamps_ns),),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["workspace_rgb_timestamps"][...] = timestamps_ns.copy()

        root.create_dataset(
            "joint_state_lowdim",
            shape=joint_state_lowdim.shape,
            dtype=np.float32,
            chunks=(t_ld, *joint_state_lowdim.shape[1:]),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["joint_state_lowdim"][...] = joint_state_lowdim
        root.create_dataset(
            "joint_state_lowdim_timestamps",
            shape=(len(timestamps_ns),),
            dtype="uint64",
            chunks=(len(timestamps_ns),),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["joint_state_lowdim_timestamps"][...] = timestamps_ns.copy()

        root.create_dataset(
            "joint_action_lowdim",
            shape=joint_action_lowdim.shape,
            dtype=np.float32,
            chunks=(t_ld, *joint_action_lowdim.shape[1:]),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["joint_action_lowdim"][...] = joint_action_lowdim
        root.create_dataset(
            "joint_action_lowdim_timestamps",
            shape=(len(timestamps_ns),),
            dtype="uint64",
            chunks=(len(timestamps_ns),),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["joint_action_lowdim_timestamps"][...] = timestamps_ns.copy()

        root.create_dataset(
            "language_instruction",
            shape=(1,),
            dtype=bytes,
            chunks=(1,),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["language_instruction"][...] = np.array([lang.encode("utf-8")])
        root.create_dataset(
            "language_instruction_timestamps",
            shape=(1,),
            dtype="uint64",
            chunks=(1,),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        )
        root["language_instruction_timestamps"][...] = np.array([0], dtype=np.uint64)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-dir",
        required=False,
        type=pathlib.Path,
        help="Optional root directory where one local lerobot dataset is stored.",
    )
    ap.add_argument("--output-dir", required=True, type=pathlib.Path, help="Directory to write per-demo .zarr groups.")
    ap.add_argument(
        "--default-lang",
        type=str,
        default="",
        help="Default language instruction when label is empty or bad.",
    )

    # all sorts of race conditions if you stress the file system too much so no parallelism
    args = ap.parse_args()

    #make a dir inside output dirr called "lerobot" if it doesn't exist
    (args.output_dir / "lerobot").mkdir(parents=True, exist_ok=True)
    
    if args.raw_dir is not None:
        episodes_df = read_episodes_df(args.raw_dir)
        for _, ep in episodes_df.iterrows():
            print(ep["episode_index"])
            make_zarr(args.raw_dir, args.output_dir, ep, default_lang=args.default_lang)
        return

    # Mirror process_lerobot_video.py task mapping by dataset source.
    dataset_specs = [
        ("klucny/rl_eth", 1),
        ("klucny/rl_eth_task2", 2),
    ]

    episode_offset = 0
    for repo_id, task_id in dataset_specs:
        print(f"Downloading {repo_id} from HuggingFace Hub...")
        local_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
        print(local_dir)
        raw_dir = pathlib.Path(local_dir)

        episodes_df = read_episodes_df(raw_dir)
        print(f"Found {len(episodes_df)} episodes in {repo_id}.")

        try:
            task_lang = episode_task_text(task_id)
        except FileNotFoundError:
            task_lang = args.default_lang
            logging.warning(
                "No description file found for task%d at %s. Falling back to --default-lang.",
                task_id,
                REPO_ROOT / f"description_task{task_id}.txt",
            )
        
        for _, ep in episodes_df.iterrows():
            source_ep_idx = int(ep["episode_index"])
            out_ep_idx = source_ep_idx + episode_offset
            print(f"{repo_id}: source_episode={source_ep_idx} -> output_episode={out_ep_idx}")
            make_zarr(
                raw_dir,
                args.output_dir,
                ep,
                default_lang=task_lang,
                output_episode_idx=out_ep_idx,
            )
            
        episode_offset += len(episodes_df)

if __name__ == "__main__":
    main()