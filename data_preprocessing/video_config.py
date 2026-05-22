"""Shared SO101 camera transform parameters.

All video preprocessing (recording → zarr, recording → mp4, deployment)
must use identical crop / resize / fps values so the model sees the same
pixel distribution at training time and at inference time.

IMPORTANT: If you change any value here you must:
  1. Re-run data_preprocessing/video/process_lerobot_video.py
  2. Re-run data_preprocessing/action/process_lerobot.py
  3. Retrain all models that depend on this data
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class VideoTransformConfig:
    """All transform parameters for one camera setup."""

    # ---- crop applied to the raw frame (pixels) ----------------------------
    # Matches the ffmpeg filter used during data collection:
    #   -vf "crop=<crop_w>:<crop_h>:<crop_x>:<crop_y>,fps=<fps>"
    crop_x: int = 326
    crop_y: int = 0
    crop_w: int = 1268
    crop_h: int = 951

    # ---- target resolution expected by the model tokenizer -----------------
    # Must match cosmos_predict2's Video2WorldPipeline (480p, 4:3 aspect ratio)
    target_h: int = 480
    target_w: int = 640

    # ---- frame rate --------------------------------------------------------
    fps: int = 10

    @property
    def ffmpeg_vf(self) -> str:
        """ffmpeg -vf filter string that crops and re-samples to target fps."""
        return (
            f"crop={self.crop_w}:{self.crop_h}:{self.crop_x}:{self.crop_y}"
            f",fps={self.fps}"
        )


# ---------------------------------------------------------------------------
# Project-wide singleton — import this everywhere instead of defining
# separate constants in each script.
# ---------------------------------------------------------------------------
SO101_VIDEO_CONFIG = VideoTransformConfig()
