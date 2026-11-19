#!/bin/bash
set -euo pipefail
root=/workspace/quad-swarm-runs
eval_dir="$root/quads100-eval-24"
cd /workspace/quad-swarm-rl
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .venv/bin/python -u "$eval_dir/evaluate.py" --config "$root/quads100-1b-w16/config.json" --manifest "$eval_dir/manifest.txt" --output "$eval_dir/episodes.csv" --agents 24 --episodes 100 --workers 16
