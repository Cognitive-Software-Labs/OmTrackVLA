# perception

People seen by each sensor, before fusion.

- `camera_people.py` — YOLO person boxes associated over frames by BoT-SORT →
  `CameraTrack[]`. The appearance embedding is injected (colour signature
  today, OSNet-AIN planned).
- Planned: scan merge of both 270° lidars into one 360° scan, and lidar person
  detection (leg detector or DR-SPAAM) → `LidarPerson[]`.
