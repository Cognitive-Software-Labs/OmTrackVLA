# robot

Small Ridgeback helpers shared by the OmTrackVLA deploy tools.
- `topics.py` — Ridgeback ROS topic names
- `ros_camera.py` — latest-frame QoS, stamp and JPEG helpers
- `owned_processes.py` — start and stop only processes a run owns; idle-GPU check
- `files.py`, `latency.py`, `perf_log.py` — hashes, fresh output paths, latency percentiles, stage timing
- `protocol.py` — length-prefixed JSON socket messages
- `sample_compute_host.py` — host CPU, RAM and process-memory telemetry
- `rviz/` — live map and lidar view configs
