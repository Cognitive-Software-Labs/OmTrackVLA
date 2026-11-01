# OmTrackVLA on the Ridgeback

See [deployment instructions](../docs/DEPLOYMENT.md) and the [method README](../README.md).
This directory is deployment tooling added to the original model, not part of
its released neural architecture.

| Files | Responsibility |
|---|---|
| `inference_server.py` | Inference service (protocol and timing from `robot/protocol.py`, `robot/perf_log.py`) |
| `model_client.py` | Client side of the protocol: JPEG request, validated waypoint reply |
| `defaults.py` | Ports and default prompt shared by every tool (camera topic: `robot/topics.py`) |
| `start_omtrack_ridgeback.sh` | CUDA model-only localhost launcher |
| `live_omtrack_preview.py`, `start_omtrack_preview.sh` | Existing ROS camera subscriber and browser preview |
| `query_omtrack.py` | One-image request with saved waypoint response |
| `grounding.py` | Optional prompt-grounded target: Grounding DINO + colour signature on the shared camera tracker and leader lock |
| `benchmark_omtrack_only.py`, `benchmark_native_modes.py` | Fixed-input compute measurements |
| `run_remote_omtrack_case.py` | Explicitly configured remote serving benchmark |
| `record_live_model_test.py` | Bounded camera recording paired with fresh model predictions |
| `record_prompt_clip.py`, `evaluate_prompt_swap.py` | Camera-only capture and offline A/B/B/A prompt replay |
| `waypoint_data.py` | Explicit timestamped pose-to-waypoint label validation |

No ROS movement publisher, Nav2 or DyNAMO integration is included. Mode
`model_only` does not establish target presence, identity or path correctness.
The optional wrapper requires separate tracking dependencies/model assets.

Each benchmark writes its results to `report.json` and `outputs.json` in its
output folder.
No private recordings, archived session logs or machine-specific process IDs
are shipped with the source.
