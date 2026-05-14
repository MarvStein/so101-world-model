# TODO

We have to do Video Model Finetuning with the LeRobot dataset and then Action Decoder Pretraining. For that, we also need to setup brev completely to be able to run everything on the H100.


## Inference
- [ ] Write the inference pipeline that runs on a 5090


## Video Model Finetuning

- [-] Look into pre-computing the embedded space @jeremiasbaur
      This is implemented but not tested yet because I first had to adapt the process_lerobot.py script to extract the action sequences. However, I ran into storage problems on the instance if I tried to script on the whole dataset of Konstantin.

- [ ] Run finetuning on an instance with 8 H100s


### To run finetuning:
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

# finetune
torchrun --nproc_per_node=<NUM_GPUS> -m scripts.train --config=cosmos_predict2/configs/config.py -- experiment="v2w_lerobot-so101_custom"
```


## Action Decoder Pretraining

- [x] data_preprocessing/action/process_lerobot.py already written to convert LeRobot to zarr
  - [ ] DO NOT FORGET: Change path to run it yourself in model/cosmos_predict2/configs/dataloading/dataset/lerobot.yaml
- [ ] Make sure the data is adapted to remove the 6th DOF of the open/close gripper because we don't need it
    I think this is done right?
- [ ] Fix the in_channels / out_channels number
- [ ] Reduce the action decoder by a factor of 10
- [ ] Check if the language embeddings are correctly produced and passed to the action decoder / video model.

To run preprocessing for action decoder training:

```bash
source .venv/bin/activate
python /home/ubuntu/workspace/so101-world-model/data_preprocessing/action/process_lerobot.py --output-dir /home/ubuntu/workspace/so101-world-model/data/action/processed
```

Then run precomputation pipeline:
Adapt paths!
```bash
deactivate
cd model
source .venv/bin/activate
python scripts/precompute_video_embeddings.py \
  --video_model /home/ubuntu/workspace/so101-world-model/model/checkpoints/posttraining/video2world/v2w_lerobot-so101_custom/checkpoints/model/iter_000004000_fused.pt \
  --dataset_path /home/ubuntu/workspace/so101-world-model/data/action/processed/lerobot \
  --data_config lerobot \
  --split both \
  --batch_size 4
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
