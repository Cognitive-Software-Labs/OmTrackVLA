#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# Keep the machine's ROS Python packages out of this Python 3.9 environment.
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$PWD/.local/huggingface}"
export HF_MODEL_DIR="${HF_MODEL_DIR:-$PWD/models/OmTrackVLA-0.6B}"
export DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-$PWD/models/dinov3-vits16-pretrain-lvd1689m}"
export SIGLIP_MODEL_PATH="${SIGLIP_MODEL_PATH:-$PWD/models/siglip-so400m-patch14-384}"
if [[ $# -eq 0 ]]; then
    set -- -m sim.run_demo
elif [[ "$1" == "--benchmark" ]]; then
    shift
    set -- -m sim.run_eval --run-type eval --split-id 0 --split-num 30 \
        --exp-config habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml \
        --save-path sim_data/eval/stt "$@"
    export HF_MODEL_ID="${HF_MODEL_ID:-omlab/OmTrackVLA-0.6B}"
fi
runtime_python="${OMTRACK_PYTHON:-$PWD/.venv/bin/python}"
[[ -x "$runtime_python" ]] || { echo 'Run bash setup.sh or set OMTRACK_PYTHON.' >&2; exit 1; }
export PYTHONPATH="$PWD/habitat-lab:$PWD"
export PATH="$(dirname "$runtime_python"):$PATH"
exec "$runtime_python" -u "$@"
