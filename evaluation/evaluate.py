import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from numba import njit
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from gym_art.quadrotor_multi.quad_utils import QUADS_OBSTACLE_OBS_TYPE
from swarm_rl.train import register_swarm_components


CONDITIONS = (("clean", 0.0), ("dropout_0p5s", 0.5), ("dropout_2s", 2.0))
SEED_BASE = 50000


@njit(cache=True)
def seed_numba(seed):
    np.random.seed(seed)


class Evaluation:
    def __init__(self, config, agents):
        cfg = AttrDict(json.loads(Path(config).read_text()))
        cfg.update(device="cpu", quads_num_agents=agents, quads_render=False, eval_env_frameskip=1)
        self.env = make_env_func_batched(
            cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=None
        )
        self.core = self.env.unwrapped
        self.env_info = extract_env_info(self.env, cfg)
        self.model = create_actor_critic(cfg, self.env.observation_space, self.env.action_space)
        self.model.model_to_device(torch.device("cpu"))
        self.model.eval()
        self.current_checkpoint = None
        self.agents = agents
        self.starting_boxes = [drone.box for drone in self.core.envs]
        self.rnn_size = get_rnn_size(cfg)
        self.obstacle_dims = QUADS_OBSTACLE_OBS_TYPE[cfg.quads_obstacle_obs_type]
        if self.obstacle_dims != 9 or cfg.use_rnn or cfg.env_frameskip != 1 or cfg.replay_buffer_sample_prob:
            raise ValueError("This evaluation requires the feed-forward octomap baseline without replay or frame skipping")
        self.capture_terminal = False
        self.terminal_success = None
        original_reset = self.core.reset

        def reset(*args, **kwargs):
            if self.capture_terminal:
                self.terminal_success = (
                    np.asarray(self.core.reached_goal, dtype=bool)
                    & np.asarray(self.core.agent_col_agent, dtype=bool)
                    & np.asarray(self.core.agent_col_obst, dtype=bool)
                )
            return original_reset(*args, **kwargs)

        self.core.reset = reset

    def load(self, checkpoint):
        if self.current_checkpoint != checkpoint:
            scalar_types = [np.core.multiarray.scalar, np.dtype]
            scalar_types.extend(type(np.dtype(kind)) for kind in (np.float64, np.float32, np.int64))
            with torch.serialization.safe_globals(scalar_types):
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            expected_steps = int(Path(checkpoint).stem.split("_")[2])
            if state["env_steps"] != expected_steps:
                raise RuntimeError(f"Checkpoint step mismatch for {checkpoint}")
            self.model.load_state_dict(state["model"], strict=True)
            self.current_checkpoint = checkpoint
        return int(Path(checkpoint).stem.split("_")[2])

    @torch.inference_mode()
    def episode(self, checkpoint, seed, condition, fault_seconds):
        steps = self.load(checkpoint)
        random.seed(seed)
        np.random.seed(seed)
        seed_numba(seed)
        torch.manual_seed(seed)
        for index, (drone, box) in enumerate(zip(self.core.envs, self.starting_boxes)):
            drone._seed(seed * self.agents + index)
            drone.box = box
        obs, _ = self.env.reset()
        layout = hashlib.sha256()
        for item in (self.core.obst_map, self.core.pos, self.core.scenario.goals):
            layout.update(np.ascontiguousarray(item).tobytes())
        scene_id = layout.hexdigest()[:16]

        fault_rng = np.random.default_rng(seed)
        affected = np.sort(fault_rng.choice(self.agents, math.ceil(self.agents / 4), replace=False))
        control_freq = self.core.control_freq
        start = round(fault_rng.uniform(3, 8) * control_freq)
        outage_ticks = round(fault_seconds * control_freq)
        last_observed = None
        rnn_states = torch.zeros(self.agents, self.rnn_size, dtype=torch.float32)
        reward = 0.0
        tick = 0
        self.terminal_success = None
        self.capture_terminal = True
        try:
            while True:
                if outage_ticks:
                    if tick == start - 1:
                        last_observed = obs["obs"][affected, -self.obstacle_dims:].clone()
                    elif start <= tick < start + outage_ticks:
                        obs["obs"][affected, -self.obstacle_dims:] = last_observed
                model_input = prepare_and_normalize_obs(self.model, obs)
                output = self.model(model_input, rnn_states)
                actions = argmax_actions(self.model.action_distribution())
                actions = preprocess_actions(self.env_info, actions)
                rnn_states = output["new_rnn_states"]
                obs, rewards, terminated, truncated, infos = self.env.step(actions)
                reward += float(rewards.mean())
                tick += 1
                if bool((terminated | truncated).all()):
                    break
                if tick > self.core.control_freq * 16:
                    raise RuntimeError("Episode exceeded expected 15 s horizon")
        finally:
            self.capture_terminal = False

        stats = infos[0]["episode_extra_stats"]
        success = self.terminal_success
        if success is None or tick < start + outage_ticks:
            raise RuntimeError("Incomplete episode or outage")
        if not np.isclose(success.mean(), stats["metric/agent_success_rate"]):
            raise RuntimeError("Per-agent outcomes disagree with QuadSwarm success metric")

        return {
            "checkpoint": Path(checkpoint).name,
            "trained_steps": steps,
            "agents": self.agents,
            "condition": condition,
            "seed": seed,
            "scene_id": scene_id,
            "ticks": tick,
            "fault_start_tick": start if outage_ticks else "",
            "fault_ticks": outage_ticks,
            "agent_success": float(success.mean()),
            "selected_agent_success": float(success[affected].mean()),
            "swarm_success": int(success.all()),
            "agent_obstacle_collision": stats["metric/agent_obst_col_rate"],
            "agent_neighbor_collision": stats["metric/agent_neighbor_col_rate"],
            "room_collision_events": stats["num_collisions_with_room"],
            "mean_episode_reward": reward,
        }


def initialize_worker(config, agents):
    global evaluation
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    register_swarm_components()
    evaluation = Evaluation(config, agents)


def evaluate_batch(checkpoint, first, last, conditions):
    return [
        evaluation.episode(checkpoint, seed, condition, seconds)
        for seed in range(first, last)
        for condition, seconds in conditions
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--agents", type=int, default=24)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    checkpoints = Path(args.manifest).read_text().splitlines()
    if not checkpoints or len(checkpoints) != len(set(checkpoints)):
        raise ValueError("Manifest is empty or contains duplicate checkpoints")

    started = time.monotonic()
    seed_scenes = {}

    def save_rows(rows, file, writer):
        if writer is None:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
        for row in rows:
            scene = (row["agents"], row["seed"])
            if seed_scenes.setdefault(scene, row["scene_id"]) != row["scene_id"]:
                raise RuntimeError(f"Mismatched initial maps, states or goals for {scene}")
        writer.writerows(rows)
        file.flush()
        return writer

    with Path(args.output).open("x", newline="") as file:
        if args.smoke:
            initialize_worker(args.config, args.agents)
            rows = evaluate_batch(checkpoints[-1], SEED_BASE, SEED_BASE + 1, CONDITIONS)
            save_rows(rows, file, None)
            for row in rows:
                print(row, flush=True)
            return

        batch = 5
        stop = SEED_BASE + args.episodes
        tasks = [
            (checkpoint, first, min(first + batch, stop), CONDITIONS)
            for first in range(SEED_BASE, stop, batch)
            for checkpoint in checkpoints
        ]
        print(
            f"{len(checkpoints)} checkpoints, {args.agents} agents, {args.episodes} seeds, "
            f"{len(CONDITIONS)} conditions, {args.workers} workers",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(args.config, args.agents),
        ) as pool:
            pending = [pool.submit(evaluate_batch, *task) for task in tasks]
            writer = None
            completed = 0
            for future in as_completed(pending):
                rows = future.result()
                writer = save_rows(rows, file, writer)
                completed += len(rows)
                print(
                    f"{completed}/{len(checkpoints) * args.episodes * len(CONDITIONS)} episodes "
                    f"in {time.monotonic() - started:.1f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
