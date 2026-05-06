# Dataset Recording Tutorial
This is a tutorial on how to do data collection for the SO101 robot.

1. Create a [HuggingFace](https://huggingface.co/) account if you don't have one yet.
2. Go to https://huggingface.co/new-dataset and create a new dataset, remember the dataset name!
3. Go to https://huggingface.co/settings/tokens and create an access token that has read/write access to the above created dataset repo. Save the access token because you will only see it once!
4. Open a terminal and run `hf auth login`, paste your access token.
5. Run `export HF_USER=your-username`
6. Run this command and adapt the `dataset.repo` with the dataset name from above, also adapt the ports! Adapt fps to the same number in `robot.cameras` and `dataset.fps`! We are using 10fps.

```
lerobot-record --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 1920, height: 1080, fps: 10, fourcc: MJPG, warmup_s: 5}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader \
    --display_data=false \
    --dataset.repo_id=jere-erej/rl_eth \
    --dataset.num_episodes=2 \
    --dataset.single_task="task1" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264 \
    --dataset.fps=10 \
    --dataset.reset_time_s=10 \
    --dataset.episode_time_s=20 \
    --dataset.root=./data/try0 \
    --resume=true
```

delete episode: 7, 9
maybe deleteé 11, 15,  (close to the edge of the goal circle)
goal overshoot: 17, 22, 23


## MISC commands

`lerobot-find-port`

### Calibration command

`lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=follower`

`lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=leader`


### Teleoperation command

```
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader
```
