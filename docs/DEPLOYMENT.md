# Deployment reference

## Processes and dependencies

| Process | Python | Responsibility |
|---|---|---|
| `inference_server.py --mode model_only` | `.venv` CUDA Python | RGB/text/history to waypoints |
| `live_omtrack_preview.py` | ROS Jazzy system Python | Camera subscriber, local HTML/JSON display |

The socket protocol avoids mixing Python/ROS binary environments. ROS is not
needed for offline inference. `start_omtrack_ridgeback.sh` uses
`OMTRACK_PYTHON` or `.venv/bin/python`; `OMTRACK_ASSETS` names an asset root with
`models/`. `OMTRACK_PORT` selects its port. Pass the same `--model-port` to the
preview if changing it. The preview accepts `--http-port`, `--topic`, `--prompt`
and `--output`. A non-default HTTP port also changes the browser URL.

ROS defaults are sourced from `/opt/ros/jazzy/setup.bash`. Existing DDS/domain
settings from the shell are retained. The default domain must match the camera.
Use `ros2 topic info TOPIC` to check the existing publisher before previewing.
The subscriber keeps only the latest compressed image, processes it outside the
callback and resets history after an input gap. No ROS publisher is created.
No camera driver, robot service or network profile is installed or changed.

The browser displays the model input at inference cadence, not a 30 Hz camera
stream. The configured 2 Hz model cache and client pacing result in approximately
1.4 fresh predictions/s. Inference time and update rate are different quantities.
Stale predictions are cleared in the display after two seconds. The plot is a
robot-relative diagram, not a calibrated image projection or navigation map.

## Standalone service and one-image query

```bash
PYTHONPATH=. .venv/bin/python -m real_robot.inference_server \
  --mode model_only --port 18781 --legacy-vision-preprocess
PYTHONPATH=. .venv/bin/python -m real_robot.query_omtrack /path/to/image.jpg \
  --port 18781 --output outputs/query.json
```

The service defaults to loopback. A remote bind requires explicit `--host` and
`--allowed-client` IPv4 arguments. It has a source-address allowlist, not TLS or
user authentication; keep it on a trusted, restricted network. No public
inference endpoint is created by the setup.

## Failure checks

- Missing `.venv`: run setup or select `OMTRACK_PYTHON`.
- Missing weights/access: run the downloader and `scripts/doctor.py --require-models`.
- CUDA unavailable: check the NVIDIA driver and compatible PyTorch build.
- Busy GPU/port: inspect the existing owner; launchers do not kill unknown jobs.
- Camera stale: check its publisher, ROS domain, topic and DDS configuration.
- Unexpected waypoints: inspect the actual prompt and image; finite predictions
  do not certify target presence or trajectory correctness.

Output files and logs stay under ignored `outputs/`. Stop foreground processes
with Ctrl-C. The optional PID-based stop helper is only for a separately saved
`outputs/ridgeback_omtrack/runtime.json`; a fresh clone has no such PID manifest.
