# OmTrackVLA — Ridgeback deployment

Reproducible OmTrackVLA inference, training/evaluation source, and an optional
Ridgeback camera/waypoint preview. Based on
[om-ai-lab/OmTrackVLA](https://github.com/om-ai-lab/OmTrackVLA), upstream revision
`e9cb1fbd57f8dbcf98c460914251cbcbe20f2f57`.

RGB image history and a text instruction are encoded by DINOv3/SiGLIP and a
Qwen-based planner to produce eight robot-relative `[forward, left, yaw]`
waypoints. The library is the `methods.omtrack_vla` package: network, training
data, vision features and the Hugging Face loader. Person identity, distance
estimation, Nav2 control and robot motion are not part of this repository; the
DyNAMO-Ridgeback repository runs them and calls this model as a service.

All commands run from the repository root with `PYTHONPATH=.`. The Python
environment (`.venv`), weights (`models/`) and outputs (`outputs/`) live there;
`methods/omtrack_vla/` holds the code, docs and two sample rollouts (`samples/`).
`robot/` holds the small Ridgeback helpers the deploy tools share.

## 1. Clone and install inference

Supported setup target: Linux x86_64, NVIDIA GPU/driver compatible with CUDA
12.8, Python 3.9–3.12, and the OS `python3-venv` package. The measured Ridgeback
stack is Python 3.9, PyTorch 2.8.0+cu128, TorchVision 0.23.0+cu128 and
Transformers 4.57.6. GPU inference was measured on an RTX 5060 (8 GB).

```bash
git clone https://github.com/Cognitive-Software-Labs/OmTrackVLA.git
cd OmTrackVLA
bash setup.sh
```

`setup.sh` creates `.venv`, installs pinned inference dependencies and performs
a runtime check. `PYTHON=/path/to/python bash setup.sh` selects another Python.
For CPU-only debugging, use
`TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash setup.sh`; real-time
CPU performance is not qualified. ARM64/Jetson requires its vendor-compatible
PyTorch/TorchVision build; do not install the x86 CUDA recipe there.

## 2. Download the pretrained weights

Request and accept access to
[facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)
on Hugging Face. Authenticate locally using an account with that access:

```bash
.venv/bin/hf auth login
.venv/bin/python methods/omtrack_vla/scripts/download_models.py
.venv/bin/python methods/omtrack_vla/scripts/doctor.py --require-models
```

The downloader uses the exact revisions in `requirements/model-revisions.json`
and writes a download manifest. Weights are downloaded into `models/`, not stored
in Git. They retain their publishers' licenses. Core assets are OmTrackVLA-0.6B,
Qwen3-0.6B, DINOv3 ViT-S/16 and SiGLIP SO400M. Model downloads and a compatible
GPU driver are prerequisites: cloning alone cannot supply them.

## 3. Verify inference without a robot or simulator

A small sample video is included in Git:

```bash
methods/omtrack_vla/run_local.sh
```

This decodes a real sample frame and saves eight predicted waypoints in
`outputs/sample-inference/prediction.json`. To use your own image:

```bash
methods/omtrack_vla/run_local.sh --image /path/to/image.jpg \
  --instruction 'Follow the person wearing a black T-shirt.'
```

`OMTRACK_PYTHON=/path/to/cuda/python` selects an existing environment.
`OMTRACK_ASSETS=/path/to/asset-root` selects a root containing `models/` without
copying or changing shared weights. Defaults always refer to this checkout.

## 4. Ridgeback live camera and waypoint preview

Prerequisite: Ubuntu 24.04 / ROS 2 Jazzy with the existing robot camera publisher
running. System Python needs `rclpy`, `sensor_msgs` and `python3-opencv`.
No hardware driver is bundled or automatically started by this repo.

Terminal 1:

```bash
bash methods/omtrack_vla/deploy/start_omtrack_ridgeback.sh
```

Terminal 2, after the model prints `ready`:

```bash
bash methods/omtrack_vla/deploy/start_omtrack_preview.sh
```

Open **http://127.0.0.1:18881** on Ridgeback. The browser shows timestamp-paired
camera inputs, eight model waypoints and inference timing. Model service:
loopback port 18781. Camera topic:
`/r100_0160/sensors/camera_0/color/image/compressed`.

```bash
# Different namespace/topic or instruction:
bash methods/omtrack_vla/deploy/start_omtrack_preview.sh \
  --topic /robot/sensors/camera/color/image/compressed \
  --prompt 'Follow the person wearing a blue jacket.'
```

Stop the preview with Ctrl-C, then stop the model with Ctrl-C in its terminal.
Neither launcher starts automatically on boot. See
[deployment details](methods/omtrack_vla/docs/DEPLOYMENT.md).

**Output interpretation:** model-only inference does not return a verified
person identity or no-person confidence. It can output nonzero paths in an
empty scene. `trajectory_valid` in model-only mode checks numerical finiteness;
it does not establish correct target choice, obstacle clearance or path accuracy.
The preview never executes the displayed paths.

For bounded, timestamped camera recordings paired with model outputs and an
offline prompt comparison, see [real-camera evaluation](methods/omtrack_vla/docs/REAL_CAMERA_EVALUATION.md).
Local camera recordings stay under ignored `outputs/`; they are not published.

## 5. Compute benchmarks

Create reproducible RGB fixtures from an available video:

```bash
.venv/bin/python methods/omtrack_vla/scripts/prepare_benchmark.py \
  --video methods/omtrack_vla/samples/seed_100/17DRP5sb8fy/4.mp4
PYTHONPATH=. .venv/bin/python -m methods.omtrack_vla.deploy.benchmark_omtrack_only direct \
  --output outputs/benchmark/local
```

The benchmark requires an idle GPU, warms the model, resets history at sequence
boundaries and rejects cached/nonfinite measured outputs. Saved reports contain
input/model/source hashes, raw waypoints, latency percentiles and memory data.
Original private lab camera recordings and local run logs are not published.
See [benchmark protocol and verified measurements](methods/omtrack_vla/docs/BENCHMARKS.md).

## 6. Planner mode for an external identity stage

`--mode planner` (the server default) plans only for a target the robot's own
identity stage has already locked: each request carries the target state and
its bounding box (`target` in the request), and the server returns zero
velocity unless that target is `LOCKED`. Crowd leader following uses it this
way: the people tracking and leader lock live in the DyNAMO-Ridgeback
repository (branch `leader-following`), whose client node calls this server.

## 7. Training

The upstream model, trainer, converter and data tooling are retained. The
Habitat simulator stack was removed; simulation uses the DyNAMO-Ridgeback
simulators (Gazebo, Isaac Sim) instead. See [training instructions](methods/omtrack_vla/docs/TRAINING.md).

## 8. Tests

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest discover -s methods/omtrack_vla/tests -t . -v
```

Tests need no models, ROS or GPU. GPU inference and real-camera verification are
separate checks; their tested scope is documented in [docs/VALIDATION.md](methods/omtrack_vla/docs/VALIDATION.md).

## Source attribution

See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream source, bundled licenses and
model/data access requirements. This deployment does not claim new model
accuracy or reproduce the authors' simulation benchmark scores.
