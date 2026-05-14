#!/usr/bin/env bash
set -euo pipefail

# Run Video2World inference with 5 conditional frames.
# Optional start-frame logic:
# - If --start-frame-index is provided (0-based), this script trims the source video
#   to the first N frames (N=start-frame-index), so the model conditions on frames
#   [N-5, ..., N-1] and prediction effectively starts at frame N.

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_video2world_5cond.sh \
    --dit-path /abs/path/to/iter_000001000_fused.pt \
    --input-video /abs/path/to/input.mp4 \
    --prompt "robot reaches and grasps object" \
    --output-video /abs/path/to/output.mp4 \
    [--start-frame-index 120] \
    [--seed 0] \
    [--guidance 7] \
    [--num-gpus 1] \
    [--load-ema] \
    [--enable-guardrail]

Arguments:
  --dit-path             Path to DiT checkpoint (.pt/.safetensors). For LoRA runs,
                         pass a fused checkpoint (e.g. *_fused.pt).
  --input-video          Input conditioning video (.mp4).
  --prompt               Text prompt.
  --output-video         Output generated video path (.mp4).

Optional:
  --start-frame-index    0-based frame index where prediction should start.
                         Must be >= 5 for 5-frame conditioning.
  --seed                 RNG seed (default: 0).
  --guidance             CFG guidance (default: 7).
  --num-gpus             Context-parallel GPU count (default: 1).
  --load-ema             Use EMA weights in checkpoint (passes --load_ema).
  --enable-guardrail     Keep guardrail enabled (default is disabled in this helper).
  -h, --help             Show this help.

Notes:
  - This script expects to be run from anywhere inside the repo and uses
    "python -m scripts.run_video2world" from the model directory.
  - Requires ffmpeg only when --start-frame-index is used.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DIT_PATH=""
INPUT_VIDEO=""
PROMPT=""
OUTPUT_VIDEO=""
START_FRAME_INDEX=""
SEED="0"
GUIDANCE="7"
NUM_GPUS="1"
LOAD_EMA="0"
ENABLE_GUARDRAIL="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dit-path)
      DIT_PATH="$2"
      shift 2
      ;;
    --input-video)
      INPUT_VIDEO="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    --output-video)
      OUTPUT_VIDEO="$2"
      shift 2
      ;;
    --start-frame-index)
      START_FRAME_INDEX="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --guidance)
      GUIDANCE="$2"
      shift 2
      ;;
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --load-ema)
      LOAD_EMA="1"
      shift
      ;;
    --enable-guardrail)
      ENABLE_GUARDRAIL="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$DIT_PATH" || -z "$INPUT_VIDEO" || -z "$PROMPT" || -z "$OUTPUT_VIDEO" ]]; then
  echo "Missing required arguments." >&2
  usage
  exit 1
fi

if [[ ! -f "$DIT_PATH" ]]; then
  echo "Checkpoint not found: $DIT_PATH" >&2
  exit 1
fi

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Input video not found: $INPUT_VIDEO" >&2
  exit 1
fi

INPUT_FOR_INFERENCE="$INPUT_VIDEO"
TMP_DIR=""

if [[ -n "$START_FRAME_INDEX" ]]; then
  if ! [[ "$START_FRAME_INDEX" =~ ^[0-9]+$ ]]; then
    echo "--start-frame-index must be a non-negative integer, got: $START_FRAME_INDEX" >&2
    exit 1
  fi

  if [[ "$START_FRAME_INDEX" -lt 5 ]]; then
    echo "For 5-frame conditioning, --start-frame-index must be >= 5 (got: $START_FRAME_INDEX)." >&2
    exit 1
  fi

  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg is required when using --start-frame-index but was not found in PATH." >&2
    exit 1
  fi

  TMP_DIR="$(mktemp -d)"
  INPUT_FOR_INFERENCE="${TMP_DIR}/conditioning_prefix.mp4"

  # Keep only the first N frames. The pipeline then uses the last 5 frames of this
  # clip as conditioning, making prediction start at frame index N.
  ffmpeg -y -i "$INPUT_VIDEO" -frames:v "$START_FRAME_INDEX" -an "$INPUT_FOR_INFERENCE" >/dev/null 2>&1
fi

cleanup() {
  if [[ -n "$TMP_DIR" && -d "$TMP_DIR" ]]; then
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT

mkdir -p "$(dirname "$OUTPUT_VIDEO")"

cd "$MODEL_DIR"

CMD=(
  python -m scripts.run_video2world
  --dit_path "$DIT_PATH"
  --input_path "$INPUT_FOR_INFERENCE"
  --prompt "$PROMPT"
  --num_conditional_frames 5
  --guidance "$GUIDANCE"
  --seed "$SEED"
  --num_gpus "$NUM_GPUS"
  --save_path "$OUTPUT_VIDEO"
)

if [[ "$LOAD_EMA" == "1" ]]; then
  CMD+=(--load_ema)
fi

if [[ "$ENABLE_GUARDRAIL" != "1" ]]; then
  CMD+=(--disable_guardrail)
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
