#!/usr/bin/env bash
# Core inference setup; model licenses/downloads are handled separately.
set -euo pipefail
# Avoid importing ROS/user overlays while creating the inference environment.
unset PYTHONPATH
export PYTHONNOUSERSITE=1
cd "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${PYTHON:-python3}"
venv="${OMTRACK_VENV:-$PWD/.venv}"
index="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
"$python_bin" -c 'import sys; assert (3,9) <= sys.version_info[:2] <= (3,12), "Use Python 3.9–3.12"'
"$python_bin" -m venv "$venv"
"$venv/bin/python" -m pip install 'pip==25.2'
"$venv/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url "$index"
"$venv/bin/python" -m pip install -r requirements/inference.txt
"$venv/bin/python" -m pip check
"$venv/bin/python" scripts/doctor.py
printf '\nSetup complete. Next: %s/bin/python scripts/download_models.py\n' "$venv"
