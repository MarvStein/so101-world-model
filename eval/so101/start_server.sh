#!/usr/bin/env bash
# Model inference server startup script — runs on the H100 brev instance.
#
# Starts model_server.py with the correct PYTHONPATH and ldconfig shim so
# transformer_engine can locate libnvrtc inside the model venv without root.
#
# Workflow (three terminals):
#
#   Terminal A — brev instance:
#     bash eval/so101/start_server.sh \
#         --video_model  /path/to/video_backbone.pt \
#         --action_model /path/to/action_decoder.pt \
#         --stats_path   /path/to/dataset_statistics.json \
#         [--port 5555]
#
#   Terminal B — local machine:
#     brev port-forward <instance-name> --port 5555:5555
#
#   Terminal C — local machine:
#     bash eval/so101/eval.sh \
#         --server_host localhost --server_port 5555 \
#         --target_hz 20 \
#         --task 1 \
#         [see robot_controller.py --help for all options]
#
# Environment variables (all optional):
#   MODEL_PYTHON         Explicit path to the model-env Python interpreter.
#   CUDA_VISIBLE_DEVICES GPU index (default: 0).

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
    echo "[start_server.sh] ERROR: Cannot find a Python interpreter for ${label}." >&2
    echo "[start_server.sh]        Set the ${label} environment variable to the" >&2
    echo "[start_server.sh]        absolute path of the correct interpreter, e.g.:" >&2
    echo "[start_server.sh]          export ${label}=/path/to/venv/bin/python" >&2
    exit 1
}

if [[ -z "${MODEL_PYTHON:-}" ]]; then
    MODEL_PYTHON="$(_find_python MODEL_PYTHON \
        "${REPO_ROOT}/model/.venv/bin/python" \
        "${REPO_ROOT}/.venv-model/bin/python" \
        "${REPO_ROOT}/.venv/bin/python" \
        "${HOME}/.venvs/so101-world-model/bin/python" \
        "${HOME}/.venvs/cosmos/bin/python")"
fi

# ---------------------------------------------------------------------------
# ldconfig shim — no-sudo workaround for transformer_engine libnvrtc discovery
#
# transformer_engine runs  `ldconfig -p | grep 'libnvrtc'`  to locate the
# CUDA NVRTC library.  The library lives inside the model venv's nvidia
# packages but is not registered in the system ldconfig cache (registering
# requires root).  We create a temporary shim script named `ldconfig` that
# calls the real one and then appends the venv library entries in the
# expected format.  The shim directory is prepended to PATH only for the
# model server process.
# ---------------------------------------------------------------------------
_MODEL_SITE_PKG="$("${MODEL_PYTHON}" -c \
    "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
_NVRTC_LIB_DIR="${_MODEL_SITE_PKG}/nvidia/cuda_nvrtc/lib"
_CUDART_LIB_DIR="${_MODEL_SITE_PKG}/nvidia/cuda_runtime/lib"
_CUBLAS_LIB_DIR="${_MODEL_SITE_PKG}/nvidia/cublas/lib"

_REAL_LDCONFIG="$(command -v ldconfig 2>/dev/null || echo /sbin/ldconfig)"

_SHIM_DIR="$(mktemp -d)"
cat > "${_SHIM_DIR}/ldconfig" << SHIM_SCRIPT
#!/bin/bash
"${_REAL_LDCONFIG}" "\$@" 2>/dev/null || true
for _f in "${_NVRTC_LIB_DIR}"/libnvrtc*.so*; do
    [[ -f "\$_f" ]] || continue
    printf "\\t%s (libc6,x86-64) => %s\\n" "\$(basename "\$_f")" "\$_f"
done
SHIM_SCRIPT
chmod +x "${_SHIM_DIR}/ldconfig"

_MODEL_LD_LIBRARY_PATH="${_NVRTC_LIB_DIR}:${_CUDART_LIB_DIR}:${_CUBLAS_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# ---------------------------------------------------------------------------
# Cleanup shim dir on exit
# ---------------------------------------------------------------------------
cleanup() {
    rm -rf "${_SHIM_DIR}"
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[start_server.sh] REPO_ROOT:            ${REPO_ROOT}"
echo "[start_server.sh] MODEL_PYTHON:         ${MODEL_PYTHON}"
echo "[start_server.sh] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "[start_server.sh] ldconfig shim:        ${_SHIM_DIR}/ldconfig"
echo "[start_server.sh] NVRTC lib dir:        ${_NVRTC_LIB_DIR}"
echo ""

# ---------------------------------------------------------------------------
# Start model server (foreground — Ctrl-C or SIGTERM to stop)
# ---------------------------------------------------------------------------
echo "[start_server.sh] Starting model server ..."
PATH="${_SHIM_DIR}:${PATH}" \
LD_LIBRARY_PATH="${_MODEL_LD_LIBRARY_PATH}" \
PYTHONPATH="${REPO_ROOT}/model:${REPO_ROOT}/data_preprocessing${PYTHONPATH:+:${PYTHONPATH}}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
TOKENIZERS_PARALLELISM=false \
    "${MODEL_PYTHON}" "${SCRIPT_DIR}/model_server.py" \
        "$@"
