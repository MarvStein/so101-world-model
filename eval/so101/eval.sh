#!/usr/bin/env bash
# Closed-loop SO101 deployment with the Video-World-Action pipeline.
#
# The model server runs on a remote H100 brev instance.  This script only
# starts the robot controller on the local machine (lerobot environment).
#
# Prerequisites — run these BEFORE this script:
#
#   1. On the brev instance, start the model server:
#        bash eval/so101/start_server.sh \
#            --video_model  /path/to/video_backbone.pt \
#            --action_model /path/to/action_decoder.pt \
#            --stats_path   /path/to/dataset_statistics.json \
#            [--port 5555]
#
#   2. On the local machine, open a port-forwarding tunnel:
#        brev port-forward <instance-name> --port 5555:5555
#
# Usage:
#   bash eval/so101/eval.sh \
#       [--server_host localhost] [--server_port 5555] \
#       [--target_hz 20] \
#       [--task 1] \
#       [--robot_port /dev/ttyACM1] \
#       [--task_description "Push ..."] \
#       [see robot_controller.py --help for all options]
#
# Environment variables (optional):
#   LEROBOT_PYTHON  Explicit path to the lerobot-env Python interpreter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Python interpreter discovery
# ---------------------------------------------------------------------------
_find_python() {
    local label="$1"; shift
    for candidate in "$@"; do
        if [[ -x "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    echo "[eval.sh] ERROR: Cannot find a Python interpreter for ${label}." >&2
    echo "[eval.sh]        Set the ${label} environment variable to the" >&2
    echo "[eval.sh]        absolute path of the correct interpreter, e.g.:" >&2
    echo "[eval.sh]          export ${label}=/path/to/venv/bin/python" >&2
    exit 1
}

if [[ -z "${LEROBOT_PYTHON:-}" ]]; then
    LEROBOT_PYTHON="$(_find_python LEROBOT_PYTHON \
        "${REPO_ROOT}/.venv-lerobot/bin/python" \
        "${REPO_ROOT}/.venv/bin/python" \
        "${HOME}/.venvs/lerobot/bin/python")"
fi

# ---------------------------------------------------------------------------
# Misc environment
# ---------------------------------------------------------------------------
export TOKENIZERS_PARALLELISM=false

echo "[eval.sh] REPO_ROOT:      ${REPO_ROOT}"
echo "[eval.sh] LEROBOT_PYTHON: ${LEROBOT_PYTHON}"
echo ""

# ---------------------------------------------------------------------------
# Start the robot controller (foreground — blocks until episode finishes)
# ---------------------------------------------------------------------------
echo "[eval.sh] Starting robot controller ..."
PYTHONPATH="${REPO_ROOT}/lerobot/src:${REPO_ROOT}/data_preprocessing${PYTHONPATH:+:${PYTHONPATH}}" \
TOKENIZERS_PARALLELISM=false \
    "${LEROBOT_PYTHON}" "${SCRIPT_DIR}/robot_controller.py" \
        "$@"

echo "[eval.sh] Robot controller exited."
