# Local camera and waypoint tools

See [deployment instructions](../docs/DEPLOYMENT.md) and the root README.
This directory is deployment tooling added to the original model, not part of
its released neural architecture.

| Files | Responsibility |
|---|---|
| `inference_server.py`, `protocol.py`, `perf_log.py` | Inference service, bounded JSON protocol, optional timing |
| `start_omtrack_ridgeback.sh` | CUDA model-only localhost launcher |
| `live_omtrack_preview.py`, `start_omtrack_preview.sh` | Existing ROS camera subscriber and browser preview |
| `query_omtrack.py` | One-image request with saved waypoint response |
| `stop_omtrack_preview.py` | Optional identity-checked stop using a local runtime manifest |
| `target_perception.py` | Optional Grounding DINO/YOLO/appearance identity wrapper |
| `benchmark_omtrack_only.py`, `benchmark_native_modes.py` | Fixed-input compute measurements |
| `run_remote_omtrack_case.py` | Explicitly configured remote serving benchmark |
| `report_omtrack_only.py`, `report_native_modes.py` | Historical lab report format readers |
| `sample_compute_host.py` | Read-only CPU/RAM telemetry |
| `record_live_model_test.py` | Bounded camera recording paired with fresh model predictions |
| `record_prompt_clip.py`, `evaluate_prompt_swap.py` | Camera-only capture and offline A/B/B/A prompt replay |
| `waypoint_data.py` | Explicit timestamped pose-to-waypoint label validation |
| `rviz/live_mapping.yaml`, `rviz/ridgeback_live_map.rviz` | Optional read-only SLAM map and RViz display settings |

No ROS movement publisher, Nav2 or DyNAMO integration is included. Mode
`model_only` does not establish target presence, identity or path correctness.
The optional wrapper requires separate tracking dependencies/model assets.

The historical report readers expect the documented lab folder/sample layout;
for new datasets, use each benchmark's `report.json`/`outputs.json` directly.
No private recordings, archived session logs or machine-specific process IDs
are shipped with the source.
