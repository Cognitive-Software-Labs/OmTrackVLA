#!/usr/bin/env bash
# Run evaluation splits sequentially on one selected GPU.
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
chunks="${CHUNKS:-30}"
output="${SAVE_PATH:-sim_data/eval/stt}"
config="${EXP_CONFIG:-habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PWD/habitat-lab:$PWD"
runtime="${OMTRACK_PYTHON:-$PWD/.venv/bin/python}"
child=""
stop() { if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; fi; }
trap stop EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for ((idx=0; idx<chunks; idx++)); do
    "$runtime" run_eval.py --split-num "$chunks" --split-id "$idx" \
        --exp-config "$config" --run-type eval --save-path "$output" &
    child=$!
    wait "$child"
    child=""
done
