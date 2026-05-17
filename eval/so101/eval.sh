#!/usr/bin/env bash
# Closed-loop SO101 deployment with the Video-World-Action pipeline.
#
# Launches two separate processes to work around the numpy 1.x / 2.x
# incompatibility between the model environment and the lerobot environment:
#
#   1. model_server.py   — model env  (numpy == 1.26.4)  — GPU inference
#   2. robot_controller.py — lerobot env (numpy >= 2.0)  — robot control
#
# The two processes communicate over a ZeroMQ IPC socket whose path is
# generated from this script's PID to avoid collisions between parallel runs.
#
# Usage:
#   bash eval/so101/eval.sh \
#       --video_model  /path/to/video_backbone.pt \
#       --action_model /path/to/action_decoder.pt \
#       --stats_path   /path/to/dataset_statistics.json \
#       [--robot_port /dev/ttyACM1] \
#       [--task_description "Push ..."] \
#       [see model_server.py --help and robot_controller.py --help for all options]
#
# All arguments are forwarded verbatim to BOTH Python scripts; each one
# silently ignores arguments it does not own (via argparse.parse_known_args).
# Do NOT pass --socket_path or --ready_file — they are set automatically.
#
# Environment variables (all optional):
#   MODEL_PYTHON    Explicit path to the model-env Python interpreter.
#   LEROBOT_PYTHON  Explicit path to the lerobot-env Python interpreter.
#   CUDA_VISIBLE_DEVICES  GPU index (default: 0).
#
# If MODEL_PYTHON / LEROBOT_PYTHON are not set, the script searches for venvs
# in the standard locations listed in the discovery section below.
#
# Required: one CUDA GPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ---------------------------------------------------------------------------
# Python interpreter discovery
# ---------------------------------------------------------------------------
# Returns the first existing executable from the candidate list, or exits with
# a helpful error message if none are found.
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

if [[ -z "${MODEL_PYTHON:-}" ]]; then
    MODEL_PYTHON="$(_find_python MODEL_PYTHON \
        "${REPO_ROOT}/.venv-model/bin/python" \
        "${REPO_ROOT}/.venv/bin/python" \
        "${HOME}/.venvs/so101-world-model/bin/python" \
        "${HOME}/.venvs/cosmos/bin/python")"
fi

if [[ -z "${LEROBOT_PYTHON:-}" ]]; then
    LEROBOT_PYTHON="$(_find_python LEROBOT_PYTHON \
        "${REPO_ROOT}/.venv-lerobot/bin/python" \
        "${REPO_ROOT}/.venv/bin/python" \
        "${HOME}/.venvs/lerobot/bin/python")"
fi

# ---------------------------------------------------------------------------
# Unique IPC paths for this invocation (shell PID avoids collisions)
# ---------------------------------------------------------------------------
SOCKET_PATH="/tmp/vam_ipc_$$"
READY_FILE="/tmp/vam_ready_$$"

# ---------------------------------------------------------------------------
# ldconfig shim — no-sudo workaround for transformer_engine libnvrtc discovery
#
# transformer_engine runs  `ldconfig -p | grep 'libnvrtc'`  to locate the
# CUDA NVRTC library.  The library lives inside the model venv's nvidia
# packages but is not registered in the system ldconfig cache (registering
# requires root).  We create a temporary shim script named `ldconfig` that
# calls the real one and then appends the venv library entries in the
# expected format.  The shim directory is prepended to PATH only for the
# model server process, so nothing else on the system is affected.
# ---------------------------------------------------------------------------
_MODEL_SITE_PKG="$("${MODEL_PYTHON}" -c \
    "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
_NVRTC_LIB_DIR="${_MODEL_SITE_PKG}/nvidia/cuda_nvrtc/lib"
_CUDART_LIB_DIR="${_MODEL_SITE_PKG}/nvidia/cuda_runtime/lib"
_CUBLAS_LIB_DIR="${_MODEL_SITE_PKG}/nvidia/cublas/lib"

# Find the real ldconfig before the shim shadows it.
_REAL_LDCONFIG="$(command -v ldconfig 2>/dev/null || echo /sbin/ldconfig)"

_SHIM_DIR="$(mktemp -d)"
cat > "${_SHIM_DIR}/ldconfig" << SHIM_SCRIPT
#!/bin/bash
# Shim: run the real ldconfig (read-only; sudo not required for -p),
# then append entries for CUDA libs bundled in the model Python venv.
"${_REAL_LDCONFIG}" "\$@" 2>/dev/null || true
for _f in "${_NVRTC_LIB_DIR}"/libnvrtc*.so*; do
    [[ -f "\$_f" ]] || continue
    printf "\\t%s (libc6,x86-64) => %s\\n" "\$(basename "\$_f")" "\$_f"
done
SHIM_SCRIPT
chmod +x "${_SHIM_DIR}/ldconfig"

# LD_LIBRARY_PATH so ctypes can resolve transitive CUDA dependencies.
_MODEL_LD_LIBRARY_PATH="${_NVRTC_LIB_DIR}:${_CUDART_LIB_DIR}:${_CUBLAS_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# ---------------------------------------------------------------------------
# Cleanup on exit / interrupt / error
# ---------------------------------------------------------------------------
SERVER_PID=""

cleanup() {
    local exit_code=$?
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[eval.sh] Sending SIGTERM to model server (PID ${SERVER_PID}) ..."
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        # Give the server up to 10 s to finish the current inference step and exit.
        local _i
        for _i in $(seq 1 10); do
            kill -0 "${SERVER_PID}" 2>/dev/null || break
            sleep 1
        done
        # Force-kill if still alive after the grace period.
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[eval.sh] Force-killing model server ..."
            kill -KILL "${SERVER_PID}" 2>/dev/null || true
        fi
    fi
    rm -f "${SOCKET_PATH}" "${READY_FILE}"
    rm -rf "${_SHIM_DIR}"
    exit "${exit_code}"
}

trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Misc environment
# ---------------------------------------------------------------------------
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[eval.sh] REPO_ROOT:            ${REPO_ROOT}"
echo "[eval.sh] MODEL_PYTHON:         ${MODEL_PYTHON}"
echo "[eval.sh] LEROBOT_PYTHON:       ${LEROBOT_PYTHON}"
echo "[eval.sh] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "[eval.sh] SOCKET_PATH:          ${SOCKET_PATH}"
echo ""

# ---------------------------------------------------------------------------
# 1. Start the model server in the background (model env)
# ---------------------------------------------------------------------------
echo "[eval.sh] Starting model server ..."
echo "[eval.sh] ldconfig shim:        ${_SHIM_DIR}/ldconfig"
echo "[eval.sh] NVRTC lib dir:        ${_NVRTC_LIB_DIR}"
PATH="${_SHIM_DIR}:${PATH}" \
LD_LIBRARY_PATH="${_MODEL_LD_LIBRARY_PATH}" \
PYTHONPATH="${REPO_ROOT}/model:${REPO_ROOT}/data_preprocessing${PYTHONPATH:+:${PYTHONPATH}}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
TOKENIZERS_PARALLELISM=false \
    "${MODEL_PYTHON}" "${SCRIPT_DIR}/model_server.py" \
        --socket_path "${SOCKET_PATH}" \
        --ready_file  "${READY_FILE}" \
        "$@" &
SERVER_PID=$!
echo "[eval.sh] Model server PID: ${SERVER_PID}"

# ---------------------------------------------------------------------------
# 2. Wait for the model server to signal readiness
#
#    Poll for the ready file every 1 s; also check that the server process
#    is still alive each iteration to avoid waiting the full timeout when
#    the server crashes during model loading.
#
#    Timeout: 300 s — generous for large checkpoints on first load.
# ---------------------------------------------------------------------------
echo "[eval.sh] Waiting for model server to load (up to 300 s) ..."
_elapsed=0
for _i in $(seq 1 300); do
    if [[ -f "${READY_FILE}" ]]; then
        echo "[eval.sh] Model server ready (${_i} s elapsed)."
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[eval.sh] ERROR: Model server (PID ${SERVER_PID}) exited before becoming ready." >&2
        exit 1
    fi
    sleep 1
    _elapsed=$((_elapsed + 1))
done

if [[ ! -f "${READY_FILE}" ]]; then
    echo "[eval.sh] ERROR: Model server did not become ready within 300 s." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Start the robot controller in the foreground (lerobot env)
#
#    eval.sh blocks here until the episode finishes (or the controller errors).
#    The EXIT trap then fires cleanup(), which sends SIGTERM to the server.
# ---------------------------------------------------------------------------
echo "[eval.sh] Starting robot controller ..."
PYTHONPATH="${REPO_ROOT}/lerobot/src:${REPO_ROOT}/data_preprocessing${PYTHONPATH:+:${PYTHONPATH}}" \
TOKENIZERS_PARALLELISM=false \
    "${LEROBOT_PYTHON}" "${SCRIPT_DIR}/robot_controller.py" \
        --socket_path "${SOCKET_PATH}" \
        "$@"

echo "[eval.sh] Robot controller exited — running cleanup ..."
# cleanup() fires via the EXIT trap.
