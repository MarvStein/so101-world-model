#!/usr/bin/env bash
set -e

VENV_SITE_PACKAGES="$(dirname "$0")/.venv/lib/python3.10/site-packages"

NVRTC_DIR="$VENV_SITE_PACKAGES/nvidia/cuda_nvrtc/lib"
CUDART_DIR="$VENV_SITE_PACKAGES/nvidia/cuda_runtime/lib"

echo "Registering libnvrtc..."
echo "$NVRTC_DIR" | sudo tee /etc/ld.so.conf.d/cuda-nvrtc-venv.conf

echo "Registering libcudart..."
echo "$CUDART_DIR" | sudo tee /etc/ld.so.conf.d/cuda-runtime-venv.conf

echo "Creating libcudart.so symlink..."
sudo ln -sf "$CUDART_DIR/libcudart.so.12" "$CUDART_DIR/libcudart.so"

echo "Refreshing ldconfig..."
sudo ldconfig

echo "Done. Verifying:"
ldconfig -p | grep -E "libnvrtc|libcudart"
