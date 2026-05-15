#!/usr/bin/env bash
# Closed-loop SO101 deployment with the Video-World-Action pipeline.
#
# Usage:
#   bash eval/so101/eval.sh \
#       --video_model  /path/to/video_backbone.pt \
#       --action_model /path/to/action_decoder.pt \
#       --stats_path   /path/to/dataset_statistics.json
#
# All extra arguments are forwarded verbatim to run.py (see --help).
#
# Required: one CUDA GPU.  Set CUDA_VISIBLE_DEVICES before calling this script
# to select a specific device, e.g.:
#   CUDA_VISIBLE_DEVICES=0 bash eval/so101/eval.sh ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Activate virtual environment (optional — skip if already active)
# ---------------------------------------------------------------------------
if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    for venv_dir in \
        "${REPO_ROOT}/.venv" \
        "${REPO_ROOT}/venv" \
        "${HOME}/.venvs/so101-world-model"
    do
        if [[ -f "${venv_dir}/bin/activate" ]]; then
            # shellcheck source=/dev/null
            source "${venv_dir}/bin/activate"
            echo "[eval.sh] Activated venv: ${venv_dir}"
            break
        fi
    done
fi

# ---------------------------------------------------------------------------
# PYTHONPATH: expose both lerobot/src (robot control) and model/ (VAM pipeline)
# ---------------------------------------------------------------------------
export PYTHONPATH="${REPO_ROOT}/lerobot/src:${REPO_ROOT}/model:${PYTHONPATH:-}"

# Suppress HuggingFace tokenizer parallelism warnings (no tokenizer in this pipeline)
export TOKENIZERS_PARALLELISM=false

# Use a single GPU by default; override with CUDA_VISIBLE_DEVICES in the caller
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[eval.sh] REPO_ROOT:           ${REPO_ROOT}"
echo "[eval.sh] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "[eval.sh] PYTHONPATH:           ${PYTHONPATH}"
echo ""

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
python "${SCRIPT_DIR}/run.py" "$@"
