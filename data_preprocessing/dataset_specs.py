# Shared dataset specifications for LeRobot preprocessing.
# Both process_lerobot_video.py and process_lerobot.py import from here.
# Each entry is (repo_id, task_id).
# Extend this list to add more datasets; re-run both preprocessing scripts afterwards.

DATASET_SPECS: list[tuple[str, int]] = [
    ("klucny/rl_eth", 1),
    ("klucny/rl_eth_task2", 2),
    ("klucny/rl_eth_task1_blue_cube", 12),
    ("klucny/rl_eth_task1_red_cube", 13),
]
