# Verified scope

Validated on 2026-10-27, Ubuntu 24.04 Ridgeback / RTX 5060 (8 GB):

- A new isolated Python 3.9 environment installed the pinned PyTorch 2.8 CUDA
  12.8 and inference dependencies through `setup.sh`; `pip check` passed.
- The publication source was exported from Git's staged tree to a separate
  directory without local outputs, model symlinks, environments or private data.
- From that isolated source, `infer-sample.py` decoded the bundled sample video
  and produced a fresh finite 8×3 waypoint array on CUDA.
- Nine protocol/repository contract tests and Python/shell syntax checks passed
  from the source snapshot without weights or ROS dependencies.

Model files for GPU validation were reused read-only from an existing local
cache to avoid redundant multi-gigabyte downloads. Their downloaded revisions
match `requirements/model-revisions.json`. The model downloader's gated-access
flow was not re-executed using a new Hugging Face account. A new operator must
obtain DINOv3 access and download their own licensed weights as documented.

`requirements/lock-ridgeback-py39-cu128.txt` records the verified dependency
closure for Linux x86_64/Python 3.9. It includes platform-specific CUDA packages;
do not use it as a universal ARM64 or Python 3.12 lock. The normal setup has
Python-version markers for NumPy and uses the specified PyTorch wheel index.
Other Python/OS combinations require independent runtime verification.

The full simulator benchmark and fine-tuning were not run during packaging.
They require the separate scene/avatar assets and procedures in TRAINING.md.
No claim of identity accuracy, collision safety or physical following is made.

The isolated publication source and new dependency environment were also tested
against the existing Ridgeback compressed-camera topic. At least 3 fresh
waypoint responses were observed; example inference time 166.8 ms.
The browser JSON output contained eight finite waypoints paired to camera
source stamps. Both owned validation processes were stopped after the bounded
check. No camera driver was installed or robot movement publisher created.

After initial publication, a separate shallow clone of
`https://github.com/Cognitive-Software-Labs/OmTrackVLA.git` also passed all nine
contract tests and real CUDA sample inference using the new dependency
installation and read-only matching weight cache. The eight predicted waypoints
matched the isolated staged-source validation. The public GitHub Actions workflow
runs the weight-free contracts and syntax checks; it does not run GPU inference.
