#!/bin/bash
# Setup script for so101-world-model on a new machine.
# Run from the root of the repository.
#
# Usage:
#   bash setup_new_machine.sh [--data-dir /path/to/data] [--skip-checkpoints] [--skip-data]
#
# Options:
#   --data-dir DIR        Where to write processed data (default: ./data)
#   --skip-checkpoints    Skip downloading model checkpoints
#   --skip-data           Skip all data preprocessing steps
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="{$REPO_ROOT}/data"
SKIP_CHECKPOINTS=false
SKIP_DATA=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --skip-checkpoints) SKIP_CHECKPOINTS=true; shift ;;
        --skip-data) SKIP_DATA=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "======================================================================"
echo " so101-world-model machine setup"
echo " Repo root : $REPO_ROOT"
echo " Data dir  : $DATA_DIR"
echo "======================================================================"

# ---------------------------------------------------------------------------
# 1. Root venv — for data preprocessing scripts (lerobot, zarr, etc.)
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Setting up root venv (lerobot + data-preprocessing deps)..."

cd "$REPO_ROOT"

if ! command -v uv &>/dev/null; then
    echo "  uv not found. Installing via pip..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
    source ~/.bashrc
fi

if [[ ! -d ".venv" ]]; then
    uv venv --python 3.10
fi
uv pip install -r requirements.txt   # lerobot[feetech] + any extras

echo "  Root venv ready: $REPO_ROOT/.venv"

# ---------------------------------------------------------------------------
# 2. Model venv — cosmos-predict2 (Python 3.10, CUDA 12.6)
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Setting up model venv (cosmos-predict2, cu126)..."

cd "$REPO_ROOT/model"

if [[ ! -d ".venv" ]]; then
    uv venv --python 3.10
fi
uv sync --extra cu126
uv pip install pyzmq

echo "  Model venv ready: $REPO_ROOT/model/.venv"

# ---------------------------------------------------------------------------
# 3. Download model checkpoints (video backbone + tokenizer + text encoder)
# ---------------------------------------------------------------------------
if [[ "$SKIP_CHECKPOINTS" == false ]]; then
    echo ""
    echo "[3/5] Downloading model checkpoints..."
    cd "$REPO_ROOT/model"
    uv run python scripts/download_checkpoints.py \
        --checkpoint-dir checkpoints
else
    echo ""
    echo "[3/5] Skipping checkpoint download (--skip-checkpoints)."
fi

# ---------------------------------------------------------------------------
# 4. Data preprocessing
# ---------------------------------------------------------------------------
if [[ "$SKIP_DATA" == false ]]; then

    # -- 4a. Process lerobot video (mp4 + metas/ for video finetuning) --------
    echo ""
    echo "[4a/5] Processing LeRobot video dataset (mp4 + T5 metas)..."
    cd "$REPO_ROOT"
    source .venv/bin/activate
    python data_preprocessing/video/process_lerobot_video.py 
    deactivate

    # -- 4b. Compute T5 embeddings for video finetuning -----------------------
    echo ""
    echo "[4b/5] Computing T5 embeddings for video finetuning..."
    LEROBOT_VIDEO_DIR="$DATA_DIR/video_fine/lerobot"
    cd "$REPO_ROOT/model"
    source .venv/bin/activate
    python ../data_preprocessing/video/get_t5_embeddings.py \
        --dataset_path "$LEROBOT_VIDEO_DIR"
    deactivate
    # -- 4c. Process lerobot action data (zarr for action decoder) ------------
    echo ""
    echo "[4c/5] Processing LeRobot action data to zarr..."
    ACTION_ZARR_DIR="$DATA_DIR/action"
    cd "$REPO_ROOT"
    source .venv/bin/activate
    python data_preprocessing/action/process_lerobot.py \
        --output-dir "$ACTION_ZARR_DIR"
    deactivate

    # -- 4d. Compute T5 language embeddings for action zarrs ------------------
    echo ""
    echo "[4d/5] Computing T5 language embeddings for action zarrs..."
    cd "$REPO_ROOT/model"
    source .venv/bin/activate
    python ../data_preprocessing/action/precompute_t5.py \
        --dataset-path "$ACTION_ZARR_DIR/lerobot"
    deactivate

    bash "$REPO_ROOT/model/fix_cuda_libs.sh"

    echo ""
    echo "  Data preprocessing complete."
    echo "  ┌── Action zarrs : $ACTION_ZARR_DIR/lerobot"
    echo "  └── Video data   : $LEROBOT_VIDEO_DIR"

    # -- 4e. Remind the user to update the dataset path in the config ---------
    CONFIG_YAML="$REPO_ROOT/model/cosmos_predict2/configs/dataloading/dataset/lerobot.yaml"
    echo ""
    echo "  ⚠  ACTION REQUIRED: Update data_dir in:"
    echo "     $CONFIG_YAML"
    echo "     Set: data_dir: $ACTION_ZARR_DIR/lerobot"

else
    echo ""
    echo "[4/5] Skipping data preprocessing (--skip-data)."
fi

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
echo ""
echo "======================================================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo ""
echo " A) Video model finetuning:"
echo "    cd $REPO_ROOT/model"
echo "    uv run torchrun --nproc_per_node=<NUM_GPUS> -m scripts.train \\"
echo "        --config=cosmos_predict2/configs/config.py \\"
echo "        -- experiment=v2w_lerobot-so101_custom"
echo ""
echo " B) (Optional) Precompute video latents for faster action decoder training:"
echo "    cd $REPO_ROOT/model"
echo "    uv run python scripts/precompute_video_embeddings.py \\"
echo "        --video_model checkpoints/posttraining/video2world/<YOUR_CKPT>.pt \\"
echo "        --dataset_path $DATA_DIR/action/processed/lerobot \\"
echo "        --data_config lerobot \\"
echo "        --split both \\"
echo "        --batch_size 4"
echo ""
echo " C) Action decoder training:"
echo "    cd $REPO_ROOT/model"
echo "    uv run torchrun --nproc_per_node=<NUM_GPUS> -m scripts.train \\"
echo "        --config=cosmos_predict2/configs/config.py \\"
echo "        -- experiment=w2a_lerobot_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1"
echo "======================================================================"
