#!/usr/bin/env bash
# Collect STT training rollouts with the ground-truth-box baseline agent.
set -euo pipefail
# Habitat configs and data/ paths are relative to the repository root.
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
runtime="${OMTRACK_PYTHON:-$PWD/.venv/bin/python}"
CHUNKS=30
NUM_PARALLEL=2
SEED=101
SAVE_PATH="sim_data/stt_train/seed_${SEED}"
IDX=0

while [ $IDX -lt $CHUNKS ]; do
    for ((i = 0; i < NUM_PARALLEL && IDX < CHUNKS; i++)); do
        echo "Launching job IDX=$IDX on GPU=$((IDX % NUM_PARALLEL))"
        CUDA_VISIBLE_DEVICES=$((i)) SAVE_VIDEO=1 PYTHONPATH="$PWD/habitat-lab:$PWD" "$runtime" -m sim.run \
            --split-num $CHUNKS \
            --split-id $IDX \
            --exp-config 'habitat-lab/habitat/config/benchmark/nav/track/track_train_stt.yaml' \
            --run-type 'eval' \
            --save-path $SAVE_PATH \
            habitat.simulator.seed=${SEED} &
        IDX=$((IDX + 1))  # not ((IDX++)): it returns status 1 at 0 under set -e
    done
    wait
done
