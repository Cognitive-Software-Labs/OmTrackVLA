# perception

People seen by each sensor, before fusion.

- `camera_people.py` — YOLO person boxes associated over frames by BoT-SORT →
  `CameraTrack[]`. The appearance embedding is injected (colour signature
  today, OSNet-AIN planned).
- The merge of both 270° lidars into one 360° scan already exists in the
  DyNAMO-Ridgeback repo; reuse it.
- Planned: lidar person detection (leg detector or DR-SPAAM) → `LidarPerson[]`.
