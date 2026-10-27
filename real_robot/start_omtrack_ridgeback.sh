#!/usr/bin/env bash
# Run this checkout's OmTrackVLA model locally; no DyNAMO, ROS or motion output.
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"
runtime="${OMTRACK_PYTHON:-$project/.venv/bin/python}"
port="${OMTRACK_PORT:-18781}"
[[ -x "$runtime" ]] || { echo "CUDA Python not found: $runtime" >&2; exit 1; }
# Keep ROS Python overlays out of the CUDA environment.
unset PYTHONPATH
export PYTHONPATH="$project"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
/usr/bin/python3 - "$port" <<'PY'
import socket,subprocess,sys
port=int(sys.argv[1])
with socket.socket() as s:
    try:s.bind(('127.0.0.1',port))
    except OSError:raise SystemExit(f'Port {port} is occupied; inspect the existing service before starting another.')
r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True)
if r.stdout.strip():raise SystemExit('GPU compute jobs already running; inspect before launching:\n'+r.stdout)
PY
exec "$runtime" -u real_robot/inference_server.py \
    --host 127.0.0.1 --allowed-client 127.0.0.1 --port "$port" \
    --assets "${OMTRACK_ASSETS:-$project}" --mode model_only --device cuda --planner-rate 2 --legacy-vision-preprocess
