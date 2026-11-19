import os
import shlex
import sys

from swarm_rl.runs.obstacles.quads_multi_obstacles import OBSTACLE_MODEL_CLI

arguments = [
    "--device=gpu",
    "--seed=0",
    "--quads_num_agents=100",
    "--quads_mode=o_random",
    "--quads_obst_spawn_area", "12", "12",
    "--quads_room_dims", "14", "14", "10",
    "--num_workers=16",
    "--num_envs_per_worker=2",
    "--batched_sampling=True",
    "--batch_size=1280",
    "--with_wandb=False",
    "--replay_buffer_sample_prob=0",
    "--train_for_seconds=120",
    "--save_every_sec=60",
    "--train_dir=/workspace/quad-swarm-runs",
    "--experiment=quads100-gpu-pilot-b1280",
]
command = [sys.executable, *shlex.split(OBSTACLE_MODEL_CLI)[1:], *arguments]
os.execv(sys.executable, command)
