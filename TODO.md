# TODO

We have to do Video Model Finetuning with the LeRobot dataset and then Action Decoder Pretraining. For that, we also need to setup brev completely to be able to run everything on the H100.

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
  - [ ] Once this is done, you should be able to run data_preprocessing/video/get_t5_embeddings.py with the dataset path pointing to where data_preprocessing/video/process_lerobot_video.py built it 
  ```bash
    cd data_preprocessing/video/
    python get_t5_embeddings.py --dataset_path /path/to/dataset/
  ```
- [x] Create the video finetuning config by adding a lerobot entry to the train_datasets in model/cosmos_predict2/configs/defaults/data_video.py with the right dataset directory path and make sure it's adapted since we used 10 fps. @klucny
  - [x] We might have to add the hyperparameters manually to model/cosmos_predict2/configs/experiment/video2world.py for the SO101 and taking into acount the 10fps

For the finetuning, once all the TODOs are completed, we have to finetune by runing:
```bash
torchrun -m scripts.train --config=cosmos_predict2/configs/config.py -- experiment=...
```

## Action Decoder Pretraining

- [x] data_preprocessing/action/process_lerobot.py already written to convert LeRobot to zarr
  - [ ] DO NOT FORGET: Change path to run it yourself in model/cosmos_predict2/configs/dataloading/dataset/lerobot.yaml
- [ ] Make sure the data is adapted to remove the 6th DOF of the open/close gripper because we don't need it
- [ ] Figure out what the TAs mean by "pre-compute the embedded space"?? In the dataloader??
- [ ] Figure out how to reduce the action decoder by a factor of 10??

# BACKLOG

...


# DONE

...
