#!/usr/bin/env bash
# Run an OmTrackVLA script in the repository environment (default: infer_sample.py).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Keep the machine's ROS Python packages out of this environment.
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$PWD/.local/huggingface}"
export DINOV3_MODEL_PATH="${DINOV3_MODEL_PATH:-$PWD/models/dinov3-vits16-pretrain-lvd1689m}"
export SIGLIP_MODEL_PATH="${SIGLIP_MODEL_PATH:-$PWD/models/siglip-so400m-patch14-384}"
if [[ $# -eq 0 || "$1" == --* ]]; then
    set -- methods/omtrack_vla/infer_sample.py "$@"
fi
runtime_python="${OMTRACK_PYTHON:-$PWD/.venv/bin/python}"
[[ -x "$runtime_python" ]] || { echo 'Run bash setup.sh or set OMTRACK_PYTHON.' >&2; exit 1; }
export PYTHONPATH="$PWD"
export PATH="$(dirname "$runtime_python"):$PATH"
exec "$runtime_python" -u "$@"
