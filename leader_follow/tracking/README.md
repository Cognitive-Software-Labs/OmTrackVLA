# tracking

Planned. Fuses `LidarPerson[]` and `CameraTrack[]` into `PersonTrack[]` in the
`odom` frame: one constant-velocity Kalman filter per person, camera tracks
matched to lidar tracks by bearing. Lidar carries a person through camera
occlusion; the camera adds the appearance embedding when the person is in view.
