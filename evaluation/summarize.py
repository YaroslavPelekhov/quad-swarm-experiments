"""Validate the paired evaluation and create one summary row per checkpoint and condition."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean

CONDITIONS = {"clean": 0, "dropout_0p5s": 50, "dropout_2s": 200}
METRICS = (
    "agent_success",
    "selected_agent_success",
    "swarm_success",
    "agent_obstacle_collision",
    "agent_neighbor_collision",
    "room_collision_events",
)
STARTING_SEED = 50000


def summarize(source, manifest, output, episodes):
    checkpoints = {Path(path).name for path in manifest.read_text().splitlines()}
    paired = {}
    scenes = {}
    grouped = defaultdict(list)
    with source.open(newline="") as file:
        for row in csv.DictReader(file):
            checkpoint = row["checkpoint"]
            condition = row["condition"]
            agents = int(row["agents"])
            seed = int(row["seed"])
            if checkpoint not in checkpoints or condition not in CONDITIONS:
                raise ValueError(f"Unexpected checkpoint or condition: {checkpoint}/{condition}")
            if int(row["fault_ticks"]) != CONDITIONS[condition]:
                raise ValueError(f"Unexpected outage duration for {checkpoint}/{condition}/{seed}")
            scene_key = (agents, seed)
            scene_id = row["scene_id"]
            if scene_id != scenes.setdefault(scene_key, scene_id):
                raise ValueError(f"Different initial scenes for {scene_key}")
            episode_key = (checkpoint, agents, seed, condition)
            if episode_key in paired:
                raise ValueError(f"Duplicate episode: {episode_key}")
            paired[episode_key] = row
            grouped[checkpoint, agents, condition].append(row)

    agent_counts = {agents for _, agents, _ in grouped}
    expected_seeds = set(range(STARTING_SEED, STARTING_SEED + episodes))
    if not agent_counts:
        raise ValueError("No episodes to summarize")
    for checkpoint in checkpoints:
        for agents in agent_counts:
            for condition in CONDITIONS:
                rows = grouped[checkpoint, agents, condition]
                if {int(row["seed"]) for row in rows} != expected_seeds:
                    raise ValueError(f"Missing or unexpected episodes: {checkpoint}/{agents}/{condition}")
                if len({row["trained_steps"] for row in rows}) != 1:
                    raise ValueError(f"Inconsistent training step count in {checkpoint}")

    fields = ["checkpoint", "trained_steps", "agents", "condition", "episodes"]
    fields += [f"mean_{metric}" for metric in METRICS]
    fields += ["selected_agent_success_delta_pp_vs_clean"]
    groups = sorted(
        grouped.items(),
        key=lambda entry: (int(entry[1][0]["trained_steps"]), entry[0][1], CONDITIONS[entry[0][2]]),
    )
    with output.open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for (checkpoint, agents, condition), rows in groups:
            means = {f"mean_{metric}": fmean(float(row[metric]) for row in rows) for metric in METRICS}
            clean = (paired[checkpoint, agents, int(row["seed"]), "clean"] for row in rows)
            clean_mean = fmean(float(row["selected_agent_success"]) for row in clean)
            writer.writerow(
                {
                    "checkpoint": checkpoint,
                    "trained_steps": rows[0]["trained_steps"],
                    "agents": agents,
                    "condition": condition,
                    "episodes": len(rows),
                    **{key: round(value, 8) for key, value in means.items()},
                    "selected_agent_success_delta_pp_vs_clean": round(
                        100 * (means["mean_selected_agent_success"] - clean_mean), 6
                    ),
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes_csv", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    summarize(args.episodes_csv, args.manifest, args.output, args.episodes)


if __name__ == "__main__":
    main()
