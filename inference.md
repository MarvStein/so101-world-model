# Inference Setup (conda, RTX 5090 / Blackwell)

This guide replaces the uv-based Blackwell setup in TODO.md with a conda workflow.
The `uv sync --extra blackwell` path fails because `apex` is not available as a
pre-built wheel on the cosmos-cu128 index for all Blackwell configurations.
Using conda to install PyTorch first, then pip for the rest, sidesteps this.

---

## Prerequisites

- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html) installed
- RTX 5090 (SM 1.2, CUDA 12.8)
- The repo cloned to a local path (referred to as `REPO_ROOT` below)
- Checkpoints and `task_embeddings.pt` already copied to the machine (see TODO.md §"One-time: precompute T5 embeddings")

---

## 1. Model environment (`so101-model`)

This environment runs [model_server.py](eval/so101/model_server.py) on the GPU.
It requires **Python 3.10** and **numpy 1.26.4** exactly.

```bash
cd /path/to/so101-world-model   # your REPO_ROOT

# Create the env
conda create -n so101-model python=3.10 -y
conda activate so101-model

# Install PyTorch for Blackwell (CUDA 12.8) via the nightly conda channel.
# This is the step that avoids the uv/apex wheel problem.
conda install pytorch torchvision pytorch-cuda=12.8 \
    -c pytorch-nightly -c nvidia -y

# Install apex from source (no pre-built Blackwell wheel exists on PyPI).
# Requires a working CUDA 12.8 toolkit — provided by the pytorch-nightly install above.
pip install packaging
pip install git+https://github.com/NVIDIA/apex.git --no-build-isolation

# Install the remaining Blackwell-specific packages
# flash-attn-4, natten, and transformer-engine from the cosmos nightly index
pip install flash-attn-4 natten transformer-engine \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128

# Install all base cosmos-predict2 dependencies (excludes torch/torchvision/apex/etc.
# which are already installed above)
pip install \
    attrs==25.1.0 \
    better-profanity==0.7.0 \
    boto3==1.38.31 \
    decord==0.6.0 \
    diffusers==0.33.1 \
    einops==0.8.1 \
    ftfy==6.3.1 \
    fvcore==0.1.5.post20221221 \
    h11==0.16.0 \
    hydra-core==1.3.2 \
    "imageio[pyav,ffmpeg]==2.37.0" \
    iopath==0.1.10 \
    ipdb==0.13.13 \
    loguru==0.7.3 \
    mediapy==1.2.4 \
    megatron-core==0.12.1 \
    modelscope==1.26.0 \
    nltk==3.9.1 \
    "numpy==1.26.4" \
    nvidia-ml-py==12.535.133 \
    omegaconf==2.3.0 \
    "opencv-python==4.11.0.86" \
    pandas==2.2.3 \
    peft==0.15.2 \
    pillow==11.1.0 \
    "protobuf==4.25.3" \
    pycocotools==2.0.10 \
    pynvml==12.0.0 \
    pyyaml==6.0.2 \
    "qwen-vl-utils[decord]==0.0.11" \
    retinaface-py==0.0.2 \
    safetensors==0.5.3 \
    scikit-image==0.25.2 \
    sentencepiece==0.2.0 \
    setuptools==80.9.0 \
    termcolor==3.1.0 \
    threadpoolctl==3.6.0 \
    tqdm==4.66.5 \
    transformers==4.51.3 \
    triton==3.2.0 \
    tyro==1.0.8 \
    webdataset==0.2.111

# Install pyzmq for the ZeroMQ IPC socket used by eval.sh
pip install pyzmq

# Install the cosmos-predict2 package itself (editable, no extras)
cd model
pip install -e . --no-deps
cd ..
```

---

## 2. LeRobot environment (`so101-lerobot`)

This environment runs [robot_controller.py](eval/so101/robot_controller.py).
It needs **numpy 2.x** (lerobot requirement) which is incompatible with the model
env above — that is why they are separate processes talking over ZeroMQ.

```bash
conda create -n so101-lerobot python=3.10 -y
conda activate so101-lerobot

pip install "lerobot[feetech]" zarr pyzmq
```

---

## 3. Running inference

Find your camera device:
```bash
v4l2-ctl --list-devices
```

Locate the conda Python binaries — typically:
```
MODEL_PYTHON  = $CONDA_PREFIX/../so101-model/bin/python
LEROBOT_PYTHON = $CONDA_PREFIX/../so101-lerobot/bin/python

# or with the full path, e.g.:
# /home/team01/miniconda3/envs/so101-model/bin/python
# /home/team01/miniconda3/envs/so101-lerobot/bin/python
```

Run inference with precomputed T5 embeddings (recommended — skips loading the 22 GB T5 model):

```bash
MODEL_PYTHON=/home/team01/miniconda3/envs/so101-model/bin/python \
LEROBOT_PYTHON=/home/team01/miniconda3/envs/so101-lerobot/bin/python \
bash eval/so101/eval.sh \
    --video_model  /home/team01/projects/so101-world-model/model/checkpoints/v2w_11000_fused.pt \
    --action_model /home/team01/projects/so101-world-model/model/checkpoints/w2a_000003250.pt \
    --stats_path   /home/team01/projects/so101-world-model/data/action/lerobot/.statistics_cache/cf89be487e1fc98411666c8fb142a6e0f73086fe4e45c39a71fdaffe48cb03dc.json \
    --task 13 \
    --embeddings_path /home/team01/projects/so101-world-model/eval/so101/task_embeddings.pt \
    --camera_index 0 \
    --robot_port /dev/ttyACM0 \
    --fps 10 \
    --num_execute_actions 8 \
    --max_steps 20
```

Adapt `--task` and paths to your setup. Available task IDs:

| ID  | Object           | Variant         |
|-----|------------------|-----------------|
| 1   | white polyhedron | straight line   |
| 2   | white polyhedron | around obstacle |
| 12  | blue cube        | straight line   |
| 13  | orange cube      | straight line   |
| 22  | blue cube        | around obstacle |
| 23  | orange cube      | around obstacle |

### Optional parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--stop_denoising_step` | 20 | Early-stop video denoising (35 = full quality, slower) |
| `--num_execute_actions` | 8 | Actions to execute per model call |
| `--max_steps` | 20 | Max control steps per episode |
| `--still_threshold` | 3.0 | Grayscale diff threshold for motion detection |
| `--max_wait_s` | 10.0 | Max seconds to wait for scene to settle |
| `--recv_timeout_ms` | 300000 | Controller timeout waiting for model reply |
| `--log_level` | INFO | DEBUG / INFO / WARNING / ERROR |

---

## Troubleshooting

**`apex` build fails:** Make sure the CUDA toolkit version matches PyTorch.
After `conda install pytorch ... pytorch-cuda=12.8`, run `python -c "import torch; print(torch.version.cuda)"` — it should print `12.8`.
Then retry `pip install git+https://github.com/NVIDIA/apex.git --no-build-isolation`.

**`flash-attn-4` / `natten` not found on the nightly index:** The cosmos nightly
index occasionally lags. Try pinning the nightly date:
```bash
pip install flash-attn-4 natten \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
    --pre
```

**`transformer-engine` import error:** Ensure `libcudart.so` is visible.
With conda the CUDA runtime is bundled alongside PyTorch, so add:
```bash
export LD_LIBRARY_PATH="$(python -c 'import torch; import os; print(os.path.dirname(torch.__file__))')/lib:${LD_LIBRARY_PATH}"
```

**`numpy` version conflict after install:** The model env must stay on 1.26.4.
Check with `python -c "import numpy; print(numpy.__version__)"` inside `so101-model`.
If something upgraded it, pin it back: `pip install "numpy==1.26.4" --force-reinstall`.

**`MODEL_PYTHON` / `LEROBOT_PYTHON` not found by eval.sh:** Set them explicitly
as shown above. The auto-discovery in eval.sh looks for venv paths (`.venv-model/`,
`.venv-lerobot/`) which don't exist with conda.
