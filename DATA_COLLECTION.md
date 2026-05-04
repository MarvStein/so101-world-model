# Dataset Recording Tutorial
This is a tutorial on how to do data collection for the SO101 robot.

1. Create a huggingface account if you don't have one yet.
2. Go to https://huggingface.co/new-dataset and create a new dataset, remember the dataset name!
3. Go to https://huggingface.co/settings/tokens and create an access token that has read/write access to the above created dataset repo. Save the access token because you will only see it once.
4. Open a terminal and run `hf auth login`, paste your access token.
5. Run `export HF_USER=your-username`
6. Run this command and adapt the dataset.repo with the dataset name from above, also adapt the ports.
```
lerobot-record --robot.type=so101_follower \
    --robot.calibration_dir=./calibration \
    --teleop.calibration_dir=./calibration \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 1920, height: 1080, fps: 30, fourcc: MJPG}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/rl_eth \
    --dataset.num_episodes=1 \
    --dataset.single_task="task1" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.root=./data \
    --dataset.vcodec=h264
    --resume=false
```


## MISC commands

`lerobot-find-port`


### Calibration command

`lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=follower`

`lerobot-calibrate     --teleop.type=so101_leader     --teleop.port=/dev/ttyACM0 --teleop.id=leader`


### Teleoperation command

```
lerobot-teleoperate \
    --robot.calibration_dir=./calibration \
    --teleop.calibration_dir=./calibration \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader
```


