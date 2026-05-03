import huggingface_hub
from huggingface_hub import snapshot_download
import argparse
import logging
import pathlib
import pickle
import re
from typing import Literal
import pandas as pd

import numpy as np
import tqdm
import zarr
from numcodecs import Blosc
from PIL import Image
import imageio.v3 as iio
import cv2

# snapshot_download(
#     repo_id="jere-erej/rl_eth",
#     repo_type="dataset",
#     local_dir="/Users/dragos/ETH/FS2026/RL/robot-learning-fs26/data"
# )

S_TO_NS = 1_000_000_000

def process_images(video_path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resize: (width, height)
        size = (640, 480)
        frame = cv2.resize(frame, size)
        frames.append(frame)
    cap.release()
    return np.stack(frames).astype(np.uint8)


def make_zarr(raw_dir, output_dir, ep, convert_degrees_to_radians=True):
    video_path = (
        raw_dir
        / f"videos/observation.images.front/"
          f"chunk-{int(ep['videos/observation.images.front/chunk_index']):03d}/"
          f"file-{int(ep['videos/observation.images.front/file_index']):03d}.mp4"
    )

    images = process_images(video_path)

    data_path = (
        raw_dir
        / f"data/"
          f"chunk-{int(ep['data/chunk_index']):03d}/"
          f"file-{int(ep['data/file_index']):03d}.parquet"
    )

    data_df = pd.read_parquet(data_path)

    # Keep only this episode
    ep_idx = int(ep["episode_index"])
    data_df = data_df[data_df["episode_index"] == ep_idx]

    # Restrict to episode index range if available
    if "dataset_from_index" in ep and "dataset_to_index" in ep:
        start = int(ep["dataset_from_index"])
        end = int(ep["dataset_to_index"])

        data_df = data_df[
            (data_df["index"] >= start)
            & (data_df["index"] < end)
        ]

    data_df = data_df.sort_values("frame_index").reset_index(drop=True)

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

    print(f"Read episode {ep_idx} from {data_path}")
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

    #make a dir inside output_dir/lerobot for current episode index if it doesn't exist
    out_path = output_dir / "lerobot" / f"episode_{ep_idx:03d}"
    out_path.mkdir(parents=True, exist_ok=True)

    t_img = min(65, timestamps_ns.shape[0])
    t_ld = min(1024, timestamps_ns.shape[0])

    root: zarr.Group
    with zarr.open(str(out_path), mode="w") as root:
        root.create_dataset(
            "workspace_rgb",
            shape=images.shape,
            dtype=np.uint8,
            chunks=(t_img, *images.shape[1:]),
            compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
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

        # root.create_dataset(
        #     "language_instruction",
        #     shape=(1,),
        #     dtype=bytes,
        #     chunks=(1,),
        #     compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        # )
        # root["language_instruction"][...] = np.array([lang.encode()])
        # root.create_dataset(
        #     "language_instruction_timestamps",
        #     shape=(1,),
        #     dtype="uint64",
        #     chunks=(1,),
        #     compressor=Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE),
        # )
        # root["language_instruction_timestamps"][...] = np.array([0], dtype=np.uint64)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-dir", required=True, type=pathlib.Path, help="Root directory where lerobot data is stored."
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

    #read all parquet files and concatenate into one dataframe
    dfs = []
    for p in sorted((args.raw_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        dfs.append(pd.read_parquet(p))
    episodes_df = pd.concat(dfs, ignore_index=True)

    #process each episode and write to zarr
    for _, ep in episodes_df.iterrows():
        print(ep['episode_index'])
        make_zarr(args.raw_dir, args.output_dir, ep)

if __name__ == "__main__":
    main()