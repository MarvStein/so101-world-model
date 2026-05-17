# TODO

We have to do Video Model Finetuning with the LeRobot dataset and then Action Decoder Pretraining. For that, we also need to setup brev completely to be able to run everything on the H100.

## First time setup on a new **training** GPU instance
Please ignore if you want to do inference.
Run this in the root directory of the repo. Create a ./data/ folder and provide the path

```bash
bash setup_new_machine.sh --data-dir ./data
```

## Inference

[Jere, 17.5.26, 22:48]
Tomorrow try to install uv: 
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then install the base environment in the root of the repo:
```bash
uv venv --python 3.10
uv pip install -r requirements.txt
```

Deactivate the base env and create the model env:
```bash
deactivate
cd ./model
```

**On a Blackwell GPU (RTX 5090):** `pyproject_gemini.toml` contains the
cu128 index and the `blackwell` optional-dependency group needed for
`apex`, `flash-attn-4`, `natten`, `transformer-engine`, and the matching
`torch`/`torchvision` builds.  `uv sync` (not `uv pip install`) is required
so that the `[tool.uv]` keys (`no-build-package`, conditional `sources`) are
respected — they are silently ignored by plain pip.

```bash
# Replace pyproject.toml with the Blackwell-compatible version
cp pyproject_gemini.toml pyproject.toml
# Create the venv and install everything including the blackwell extras
uv venv --python 3.10
uv sync --extra blackwell
# Install pyzmq for ZeroMQ IPC with the robot controller
uv pip install pyzmq
```

**On any other GPU (H100, A100, …):** use the original `pyproject.toml`
as-is:

```bash
uv venv --python 3.10
uv pip install pyzmq
```

The model environment uses **numpy 1.26.4** but lerobot requires **numpy 2.x**, so the
pipeline is split into two processes that talk over a ZeroMQ IPC socket:
- [model_server.py](eval/so101/model_server.py) — model env, GPU inference
- [robot_controller.py](eval/so101/robot_controller.py) — lerobot env, robot control
- [eval.sh](eval/so101/eval.sh) — launches both, wires them together, cleans up

The original monolithic [run.py](eval/so101/run.py) is kept for reference but **cannot
run** without resolving the numpy conflict.


### To run inference:

First, find your camera device:
```bash
v4l2-ctl --list-devices
```

#### One-time: precompute T5 embeddings (run on training instance, not on 5090)
(Already done!!)
The RTX 5090 deployment machine only has 50 GB of storage, so we cannot
download the T5-11B model there.  Instead, precompute the embeddings for all
tasks once on any machine that has the model env set up, then copy the ~6 MB
output file to the deployment machine.

```bash
# On the training / H100 instance (model venv):
cd /path/to/so101-world-model
source model/.venv/bin/activate
PYTHONPATH=model python eval/so101/precompute_task_embeddings.py \
    --output_path eval/so101/task_embeddings.pt
```

This produces `eval/so101/task_embeddings.pt` — a dict keyed by task ID
(`"1"`, `"2"`, `"12"`, `"13"`, `"22"`, `"23"`), each value a
`(1, 512, 1024)` float16 tensor.  Copy this file to the deployment machine.

#### Running inference with precomputed embeddings (recommended for 5090)

Pass `--task <id>` and `--embeddings_path` to skip loading the T5 model
entirely (~22 GB saved).  Available task IDs:

| ID  | Object        | Variant              |
|-----|---------------|----------------------|
| 1   | white polyhedron | straight line     |
| 2   | white polyhedron | around obstacle  |
| 12  | blue cube     | straight line        |
| 13  | orange cube   | straight line        |
| 22  | blue cube     | around obstacle      |
| 23  | orange cube   | around obstacle      |

Adapt the task tag accordingly!!!
```bash
MODEL_PYTHON=/home/team01/projects/so101-world-model/model/.venv/bin/python \
LEROBOT_PYTHON=/home/team01/.conda/envs/so101/bin/python \
bash eval/so101/eval.sh \
    --video_model /home/team01/projects/so101-world-model/model/checkpoints/v2w_11000_fused.pt \
    --action_model /home/team01/projects/so101-world-model/model/checkpoints/w2a_000003250.pt \
    --stats_path /home/team01/projects/so101-world-model/data/action/lerobot/.statistics_cache/cf89be487e1fc98411666c8fb142a6e0f73086fe4e45c39a71fdaffe48cb03dc.json \
    --task 13 \
    --embeddings_path /home/team01/projects/so101-world-model/eval/so101/task_embeddings.pt \
    --camera_index 0 \
    --robot_port /dev/ttyACM0 \
    --fps 10 \
    --num_execute_actions 8 \
    --max_steps 20
```

#### Running inference without precomputed embeddings (requires T5 model)
(Ignore)
If the T5 model is available locally, you can pass the full task description
directly (T5 will be loaded at startup):

```bash
MODEL_PYTHON=/home/team01/projects/so101-world-model/model/.venv/bin/python \
LEROBOT_PYTHON=/home/team01/.conda/envs/so101/bin/python \
bash eval/so101/eval.sh \
    --video_model /home/team01/projects/so101-world-model/model/checkpoints/v2w_11000_fused.pt \
    --action_model /home/team01/projects/so101-world-model/model/checkpoints/w2a_000003250.pt \
    --stats_path /home/team01/projects/so101-world-model/data/action/lerobot/.statistics_cache/cf89be487e1fc98411666c8fb142a6e0f73086fe4e45c39a71fdaffe48cb03dc.json \
    --camera_index 0 \
    --robot_port /dev/ttyACM0 \
    --fps 10 \
    --num_execute_actions 8 \
    --max_steps 20 \
    --task_description "Push the target object, in this case the orange cube, in a straight line to the goal position which is the smaller of the two white circles seen on the left. The target object is not allowed to leave the area bounded by the two parallel straight white lines at anytime."
```

`eval.sh` auto-discovers the two interpreters from standard venv locations
(`.venv-model/`, `.venv-lerobot/`, `~/.venvs/…`) — set `MODEL_PYTHON` /
`LEROBOT_PYTHON` explicitly if yours are somewhere else.

Optional parameters (forwarded to both scripts; each ignores what it doesn't own):
- `--task <id>`: Task ID shorthand — reads description from `description_task{id}.txt`
- `--embeddings_path`: Path to precomputed `.pt` file; skips T5 loading (requires `--task`)
- `--experiment`: Experiment config name (default: w2a_lerobot_v2w_11k_lr1e-04_bs16)
- `--camera_key`: Observation dict key for front camera (default: "front")
- `--stop_denoising_step`: Early-stop video denoising (default: 20; pass 35 for full quality)
- `--still_threshold`: Motion detection threshold in grayscale diff (default: 3.0)
- `--max_wait_s`: Max time to wait for scene to settle (default: 10.0)
- `--recv_timeout_ms`: How long the controller waits for a model reply before erroring (default: 300000)
- `--log_level`: DEBUG, INFO, WARNING, ERROR (default: INFO)

## Video Model Finetuning

- [-] Look into pre-computing the embedded space @jeremiasbaur
      This is implemented but not tested yet because I first had to adapt the process_lerobot.py script to extract the action sequences. However, I ran into storage problems on the instance if I tried to script on the whole dataset of Konstantin.

- [ ] Run finetuning on an instance with 8 H100s


### To run finetuning:
Add / adapt the datasets in [dataset_specs.py](data_preprocessing/dataset_specs.py).

```bash
# get dataset (only do once)
source .venv/bin/activate # activate the so-101-world-model venv from the root of the repo
python data_preprocessing/video/process_lerobot_video.py

# compute t5 embeddings for the dataset (only do once)
deactivate
cd model
source .venv/bin/activate # activate the cosmos-predict2 venv for running the rest
cd ../data_preprocessing/video/
python get_t5_embeddings.py --dataset_path ../../data/lerobot
```

Finetune
Change the video model and action model register order in [config.py](model/cosmos_predict2/configs/config.py). The video model should be registered after action model.
Adapt path here: [data_video.py](model/cosmos_predict2/configs/defaults/data_video.py)
```bash
torchrun --nproc_per_node=<NUM_GPUS> -m scripts.train --config=cosmos_predict2/configs/config.py -- experiment="v2w_lerobot-so101_custom"
```


## Action Decoder Pretraining

- [x] data_preprocessing/action/process_lerobot.py already written to convert LeRobot to zarr
  - [ ] DO NOT FORGET: Change path to run it yourself in model/cosmos_predict2/configs/dataloading/dataset/lerobot.yaml
- [-] Make sure the data is adapted to remove the 6th DOF of the open/close gripper because we don't need it
    I think this is done right?
- [x] Fix the in_channels / out_channels number
- [x] Reduce the action decoder by a factor of 10
    Can adapt it here: model/cosmos_predict2/configs/defaults/world2action_pipe.py
- [ ] Check if the language embeddings are correctly produced and passed to the action decoder / video model.
- [ ] Figure out why video embedding precomp is so slow

To run preprocessing for action decoder training:

```bash
source .venv/bin/activate
python /home/ubuntu/workspace/so101-world-model/data_preprocessing/action/process_lerobot.py --output-dir /home/ubuntu/workspace/so101-world-model/data/action/processed
# Language embedding computation
deactivate
cd ./model
source .venv/bin/activate
python ../data_preprocessing/action/precompute_t5.py --dataset-path ../data/action/lerobot
```

Then run precomputation pipeline:
Adapt paths! This doesn't work correctly yet so ignore for now
```bash
deactivate
cd model
source .venv/bin/activate
python scripts/precompute_video_embeddings.py \
  --video_model /home/jbaur/projects/so101-world-model/model/checkpoints/video_backbone/iter_000003000_fused.pt \
  --dataset_path /home/jbaur/projects/so101-world-model/data/action/lerobot \
  --data_config lerobot 
```

Training the action decoder:
1. Register the fine-tuned video model here: [world2action_model.py](model/cosmos_predict2/configs/defaults/world2action_model.py)
2. Setup experiment and everything in [world2action.py](model/cosmos_predict2/configs/experiment/world2action.py)
3. Change the video model and action model register order in [config.py](model/cosmos_predict2/configs/config.py). The video model should be registered first.
4. Adapt the paths and supply the relevant **task tags** for which you want to train the action decoder in [lerobot.yaml](model/cosmos_predict2/configs/dataloading/dataset/lerobot.yaml)

Available tags, can also be supplied as list with [task1 task13]:

none: all tasks
task1: just task1
task2: just task2
task12: task1 with blue cube
task13: task1 with red cube
task22: task2 with blue cube
task23: task2 with red cube


You can test task filtering with [test_tag_filtering.py](projects/so101-world-model/model/scripts/test_tag_filtering.py)
```bash
python ./model/scripts/test_tag_filtering.py --tags task1 task12 task13 --data-dir pathto/action/lerobot
```

Run:
```bash
deactivate
cd model
source .venv/bin/activate
torchrun --nproc_per_node=2 -m scripts.train --config=cosmos_predict2/configs/config.py -- experiment="v2w_lerobot-so101_bs4_10fps"
```

# BACKLOG

...


# DONE

## Video Model Finetuning

- [x] Finish data_preprocessing/video/process_lerobot_video.py to download the HF datasets and structure it under a dataset folder in the way that the data_preprocessing/video/get_t5_embeddings.py needs it -> put the mp4 files under video/ and write task text under metas/*.txt
  - the script is DONE and the following commands need to be run to build the dataset:
  ```bash
  python data_preprocessing/video/process_lerobot_video.py
  ```
  Then you will have this structure:
  ```
  <this-repo>
  ├── data
  │   ├── lerobot
  │   │   ├── video
  │   │   │   ├── episode_000.mp4
  │   │   │   ├── ...
  │   │   ├── metas
  │   │   │   ├── episode_000.txt
  │   │   │   ├── ...
  ```
  - [x] Once this is done, you should be able to run data_preprocessing/video/get_t5_embeddings.py with the dataset path pointing to where data_preprocessing/video/process_lerobot_video.py built it 
  ```bash
    cd data_preprocessing/video/
    python get_t5_embeddings.py --dataset_path /path/to/dataset/
  ```
- [x] Create the video finetuning config by adding a lerobot entry to the train_datasets in model/cosmos_predict2/configs/defaults/data_video.py with the right dataset directory path and make sure it's adapted since we used 10 fps. @klucny
  - [x] We might have to add the hyperparameters manually to model/cosmos_predict2/configs/experiment/video2world.py for the SO101 and taking into acount the 10fps
