#!/usr/bin/env bash
# Subscribe to the existing Ridgeback camera and display model outputs only.
set -euo pipefail
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project"
set +u
source /opt/ros/jazzy/setup.bash
set -u
export PYTHONPATH="$project:${PYTHONPATH:-}"
exec /usr/bin/python3 -u -m real_robot.live_omtrack_preview "$@"
