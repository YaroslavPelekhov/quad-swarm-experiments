#!/usr/bin/env bash
set -u
run=/workspace/quad-swarm-runs/quads100-100m-w16
cd /workspace/quad-swarm-rl || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m swarm_rl.train \
    --env=quadrotor_multi --device=gpu --restart_behavior=resume \
    --experiment=quads100-100m-w16 --train_dir=/workspace/quad-swarm-runs \
    --train_for_env_steps=100000000 --train_for_seconds=10000000000 \
    --num_workers=16 --num_envs_per_worker=2 --batch_size=1280 \
    --with_wandb=False --save_every_sec=300 --keep_checkpoints=3 &
child=$!
printf '%s\n' "$child" > "$run/trainer.pid"
trap 'kill -INT "$child" 2>/dev/null || true' INT TERM
wait "$child"
status=$?
printf '%s\n' "$status" > "$run/exit_status.tmp"
mv "$run/exit_status.tmp" "$run/exit_status"
exit "$status"
