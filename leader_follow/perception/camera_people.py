"""Every person in the camera image, detected by YOLO and associated over time by BoT-SORT.

The appearance embedding is injected (``embed``), so the same tracker serves the
hand-crafted colour signature used by the OmTrackVLA wrapper and a learned
re-identification model such as OSNet.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List

import numpy as np

from leader_follow.interfaces import CameraTrack

# (bgr image, N x 4 xyxy boxes) -> N x D embeddings
Embedder = Callable[[np.ndarray, np.ndarray], np.ndarray]


class CameraPeopleTracker:
    def __init__(self, detector_path: Path, embed: Embedder, device: str = "cuda",
                 person_threshold: float = 0.25) -> None:
        from ultralytics import YOLO
        from ultralytics.trackers.bot_sort import BOTSORT

        self.device = str(device)
        self.person_threshold = float(person_threshold)
        self.embed = embed
        self.person_model = YOLO(str(detector_path))
        tracker_args = SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=0.25,
            track_low_thresh=0.10,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.80,
            fuse_score=True,
            gmc_method="sparseOptFlow",
            proximity_thresh=0.50,
            appearance_thresh=0.80,
            with_reid=True,
            model="auto",
            device=self.device,
        )
        self.tracker = BOTSORT(tracker_args)

    def reset(self) -> None:
        self.tracker.reset()

    def update(self, bgr: np.ndarray, stage_timer=None) -> List[CameraTrack]:
        """Detect every person (not only the leader) and associate with BoT-SORT."""
        stage_timer = stage_timer or (lambda _name: nullcontext())
        with stage_timer("yolo_detect"):
            result = self.person_model.predict(
                source=bgr,
                classes=[0],
                conf=self.person_threshold,
                verbose=False,
                device=self.device,
                imgsz=416,
            )[0]
            detections = result.boxes.cpu().numpy()
        with stage_timer("appearance_botsort"):
            features = self.embed(bgr, detections.xyxy)
            tracks_raw = self.tracker.update(detections, img=bgr, feats=features)
            return parse_tracks(tracks_raw, features)


def parse_tracks(rows: np.ndarray, features: np.ndarray) -> List[CameraTrack]:
    """Turn BoT-SORT rows (xyxy, id, score, class, detection index) into tracks."""
    tracks: List[CameraTrack] = []
    for row in np.asarray(rows):
        if len(row) < 8:
            continue
        detection_index = int(round(float(row[7])))
        if detection_index < 0 or detection_index >= len(features):
            continue
        tracks.append(CameraTrack(
            track_id=int(round(float(row[4]))),
            bbox=np.asarray(row[:4], dtype=np.float32),
            score=float(row[5]),
            feature=features[detection_index],
        ))
    return tracks
