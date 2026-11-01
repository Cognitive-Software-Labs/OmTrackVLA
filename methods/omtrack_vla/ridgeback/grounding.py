"""Prompt-grounded target for the OmTrackVLA Ridgeback service.

Grounding DINO is used only to initialize or re-ground a target from the
prompt.  Between grounding passes, identity is kept by the shared leader-follow
stages: leader_follow.perception.camera_people (YOLO + BoT-SORT) and
leader_follow.identity.lock_on (state machine), fed with the hand-crafted
part-based clothing colour signature below.  The signature is not a learned
person-ReID model; its similarities are uncalibrated and must be tuned on
recorded footage.  This module never generates robot motion; it only returns a
fail-closed target state to the OmTrackVLA bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import math
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from leader_follow.identity.lock_on import LeaderLock
from leader_follow.interfaces import CameraTrack
from leader_follow.perception.camera_people import CameraPeopleTracker


Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class PerceptionResult:
    query: str
    state: str
    valid: bool
    reason: str
    track_id: Optional[int]
    bbox: Optional[List[float]]
    grounding_score: Optional[float]
    person_score: Optional[float]
    reid_similarity: Optional[float]
    people_count: int
    annotated_bgr: np.ndarray
    debug: Optional[dict] = None


def target_query(instruction: str) -> str:
    """Extract the visual target description from a navigation instruction."""
    query = instruction.strip()
    query = re.sub(r"^(?:please\s+)?follow\s+", "", query, flags=re.IGNORECASE)
    query = re.split(
        r"\.(?:\s+)(?:maintain|keep|avoid|do not|without|stop)\b",
        query,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return query.strip().rstrip(".!") or instruction.strip()


SUBJECT_WORDS = {"person", "people", "man", "woman", "boy", "girl", "human", "guy", "lady", "leader", "one"}
FILLER_WORDS = {
    "the", "a", "an", "who", "that", "which", "is", "are", "was", "visibly", "clearly", "currently",
    "holding", "carrying", "wearing", "with", "in", "has", "having", "dressed", "and", "of", "his", "her",
    "their", "something", "color", "colour", "colored", "coloured",
}
# OpenCV HSV (H 0-180): list of (h_low, h_high, s_min, s_max, v_min, v_max) ranges per colour word.
COLOR_RANGES = {
    "red": [(0, 10, 80, 255, 50, 255), (170, 180, 80, 255, 50, 255)],
    "orange": [(10, 22, 80, 255, 80, 255)],
    "yellow": [(22, 35, 80, 255, 80, 255)],
    "green": [(35, 85, 60, 255, 40, 255)],
    "blue": [(85, 130, 70, 255, 40, 255)],
    "purple": [(130, 160, 60, 255, 40, 255)],
    "pink": [(145, 175, 40, 255, 120, 255)],
    "brown": [(5, 20, 80, 255, 30, 150)],
    "black": [(0, 180, 0, 255, 0, 60)],
    "white": [(0, 180, 0, 40, 180, 255)],
    "gray": [(0, 180, 0, 40, 60, 180)],
    "grey": [(0, 180, 0, 40, 60, 180)],
}


def target_attribute(query: str) -> Optional[str]:
    """Return the visual attribute phrase of a person description, if any.

    ``the person who is visibly holding a blue basket`` -> ``blue basket``;
    ``the person directly in front of you`` -> ``None`` (no visual attribute).
    Grounding the attribute separately matters: in a full sentence, Grounding
    DINO scores every human highly on the word "person" alone.
    """
    words = re.findall(r"[a-z]+", query.lower())
    if not any(word in SUBJECT_WORDS for word in words):
        return None
    start = next(index for index, word in enumerate(words) if word in SUBJECT_WORDS) + 1
    remainder = [word for word in words[start:] if word not in FILLER_WORDS]
    spatial = {"front", "directly", "ahead", "you", "me", "us", "behind", "left", "right", "near", "closest", "nearest"}
    if not remainder or all(word in spatial for word in remainder):
        return None
    attribute = [word for word in remainder if word not in spatial]
    return " ".join(attribute) if attribute else None


def attribute_color(attribute: Optional[str]) -> Optional[str]:
    if not attribute:
        return None
    return next((word for word in attribute.split() if word in COLOR_RANGES), None)


def color_fraction(bgr: np.ndarray, box: Sequence[float], color: str) -> float:
    """Fraction of pixels in ``box`` that fall in the HSV range of ``color``."""
    height, width = bgr.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    left, top = max(0, int(x1)), max(0, int(y1))
    right, bottom = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
    if right <= left or bottom <= top:
        return 0.0
    hsv = cv2.cvtColor(bgr[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=bool)
    for h_low, h_high, s_min, s_max, v_min, v_max in COLOR_RANGES[color]:
        h, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        mask |= (h >= h_low) & (h <= h_high) & (sat >= s_min) & (sat <= s_max) & (val >= v_min) & (val <= v_max)
    return float(mask.mean())


def box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def attribute_on_person(attribute_box: Sequence[float], person_box: Sequence[float]) -> float:
    """Fraction of the attribute box inside the person box, or 0 if implausibly large.

    A held object or garment must lie mostly within the person's box and be
    smaller than the person; a spurious scene-sized box is rejected.
    """
    attribute_area = box_area(attribute_box)
    if attribute_area <= 0.0 or attribute_area > 0.9 * box_area(person_box):
        return 0.0
    ax1, ay1, ax2, ay2 = [float(value) for value in attribute_box]
    px1, py1, px2, py2 = [float(value) for value in person_box]
    intersection = max(0.0, min(ax2, px2) - max(ax1, px1)) * max(0.0, min(ay2, py2) - max(ay1, py1))
    return intersection / attribute_area


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


APPEARANCE_STRIPES = ((0.15, 0.45), (0.45, 0.75), (0.75, 1.00))
APPEARANCE_BINS = (8, 4, 4)


def appearance_descriptor(bgr: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """Return an L2-normalized upper/middle/lower-body HSV colour signature.

    Each horizontal stripe of the central person crop is summarized by a
    square-rooted HSV histogram, so the cosine similarity of two descriptors is
    the mean Bhattacharyya coefficient of their stripes.  Value bins are kept so
    black and white clothing, which share hue, remain separable.
    """
    height, width = bgr.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    margin = 0.2 * max(0.0, x2 - x1)
    left = max(0, min(width - 1, int(math.floor(x1 + margin))))
    right = max(left + 1, min(width, int(math.ceil(x2 - margin))))
    top = max(0, min(height - 1, int(math.floor(y1))))
    bottom = max(top + 1, min(height, int(math.ceil(y2))))
    hsv = cv2.cvtColor(bgr[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    crop_height = hsv.shape[0]
    stripes = []
    for start, end in APPEARANCE_STRIPES:
        first = min(crop_height - 1, int(start * crop_height))
        last = max(first + 1, int(end * crop_height))
        hist = cv2.calcHist([hsv[first:last]], [0, 1, 2], None, list(APPEARANCE_BINS), [0, 180, 0, 256, 0, 256])
        hist = np.sqrt(hist.ravel() / max(float(hist.sum()), 1.0))
        stripes.append(hist / max(float(np.linalg.norm(hist)), 1e-8))
    descriptor = np.concatenate(stripes).astype(np.float32)
    return descriptor / max(float(np.linalg.norm(descriptor)), 1e-8)


def trajectory_is_directionally_consistent(
    bbox: Optional[Sequence[float]],
    image_width: int,
    trajectory: Sequence[Sequence[float]],
    target_deadband: float = 0.18,
    opposite_tolerance: float = 0.08,
) -> Tuple[bool, str]:
    """Reject only a strong left/right contradiction between target and path.

    This RGB-only check cannot validate metric following distance.  It is kept
    deliberately coarse so obstacle-avoidance curvature is not rejected merely
    because the path does not point straight at the target.
    """
    if bbox is None or image_width <= 0 or not trajectory:
        return False, "target_or_trajectory_missing"
    center_x = 0.5 * (float(bbox[0]) + float(bbox[2]))
    target_direction = (0.5 * image_width - center_x) / (0.5 * image_width)
    if abs(target_direction) <= target_deadband:
        return True, "target_centered"
    waypoint = trajectory[min(1, len(trajectory) - 1)]
    if len(waypoint) < 2:
        return False, "trajectory_waypoint_invalid"
    forward, lateral = float(waypoint[0]), float(waypoint[1])
    yaw = float(waypoint[2]) if len(waypoint) >= 3 else 0.0
    path_direction = math.atan2(lateral, max(0.03, abs(forward))) + 0.5 * yaw
    if target_direction * path_direction < 0.0 and abs(path_direction) > opposite_tolerance:
        return False, "trajectory_points_away_from_target"
    return True, "direction_consistent"


def appearance_features(bgr: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0, int(np.prod(APPEARANCE_BINS)) * len(APPEARANCE_STRIPES)), dtype=np.float32)
    return np.stack([appearance_descriptor(bgr, box) for box in boxes])


class TargetPerception:
    """Grounding DINO prompt target on top of the shared camera tracker and leader lock.

    A lost target is never replaced by a different person: recovery requires
    either the original track ID with matching appearance, or a new track that
    matches the stored appearance gallery and is independently confirmed by
    Grounding DINO, for ``acquire_frames`` consecutive frames.
    """

    def __init__(
        self,
        project_dir: Path,
        device: str = "cuda",
        grounding_box_threshold: float = 0.35,
        grounding_text_threshold: float = 0.25,
        person_threshold: float = 0.25,
        grounding_interval: int = 8,
        acquire_frames: int = 3,
        lost_frames: int = 8,
        ambiguity_margin: float = 0.08,
        phrase_overlap_threshold: float = 0.50,
        reid_match_threshold: float = 0.72,
        gallery_update_threshold: float = 0.85,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.device = str(device)
        self.grounding_box_threshold = float(grounding_box_threshold)
        self.grounding_text_threshold = float(grounding_text_threshold)
        self.ambiguity_margin = float(ambiguity_margin)
        self.phrase_overlap_threshold = float(phrase_overlap_threshold)
        self.attribute_threshold = float(grounding_box_threshold)
        self.color_fraction_threshold = 0.15

        grounding_path = self.project_dir / "models" / "grounding-dino-tiny"
        person_path = self.project_dir / "models" / "yolo11n.pt"
        required = [grounding_path / "model.safetensors", person_path]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing target-perception models:\n  " + "\n  ".join(missing))

        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.grounding_processor = AutoProcessor.from_pretrained(str(grounding_path), local_files_only=True)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(grounding_path), local_files_only=True
        ).eval().to(self.device)
        self.people = CameraPeopleTracker(person_path, appearance_features, self.device, person_threshold)
        self.lock = LeaderLock(
            acquire_interval=grounding_interval,
            acquire_frames=acquire_frames,
            lost_frames=lost_frames,
            match_threshold=reid_match_threshold,
            gallery_update_threshold=gallery_update_threshold,
        )
        self.reset()

    def reset(self) -> None:
        """Forget the target identity; the next frame starts a fresh prompt search."""
        self.people.reset()
        self.lock.reset()
        self.prompt: Optional[str] = None
        self.query = ""
        self.attribute: Optional[str] = None
        self.color: Optional[str] = None
        self.grounding_score: Optional[float] = None
        self.last_grounding: List[Dict[str, object]] = []
        self.last_candidates: List[Dict[str, object]] = []

    @torch.inference_mode()
    def update(self, rgb: np.ndarray, prompt: str, stage_timer=None) -> PerceptionResult:
        self._stage_timer = stage_timer or (lambda _name: nullcontext())
        if prompt != self.prompt:
            self.reset()
            self.prompt = prompt
            self.query = target_query(prompt)
            self.attribute = target_attribute(self.query)
            self.color = attribute_color(self.attribute)
        self.last_grounding = []
        self.last_candidates = []
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tracks = self.people.update(bgr, self._stage_timer)
        lock = self.lock.update(tracks, acquire=lambda people: self._select_grounded_track(rgb, people))
        selected = lock.selected
        valid = lock.valid(self.lock.state)
        with self._stage_timer("target_annotation"):
            annotated = self._annotate(bgr, tracks, selected, valid)
        return PerceptionResult(
            query=self.query,
            state=self.lock.state,
            valid=valid,
            reason=self.lock.reason,
            track_id=selected.track_id if selected is not None else self.lock.target_track_id,
            bbox=[float(value) for value in selected.bbox] if selected is not None else None,
            grounding_score=self.grounding_score,
            person_score=selected.score if selected is not None else None,
            reid_similarity=lock.similarity,
            people_count=len(tracks),
            annotated_bgr=annotated,
            debug={
                "frame_index": self.lock.frame_index,
                "people": [
                    {
                        "track_id": track.track_id,
                        "score": round(track.score, 3),
                        "bbox": [round(float(value), 1) for value in track.bbox],
                        "gallery_similarity": (
                            round(float(self.lock.similarity(track.feature)), 3) if self.lock.gallery else None
                        ),
                    }
                    for track in tracks
                ],
                "grounding": self.last_grounding,
                "candidates": self.last_candidates,
                "attribute": self.attribute,
                "attribute_color": self.color,
                "gallery_size": len(self.lock.gallery),
                "missing_frames": self.lock.missing_frames,
            },
        )

    def _select_grounded_track(
        self, rgb: np.ndarray, tracks: List[CameraTrack]
    ) -> Tuple[Optional[CameraTrack], str]:
        with self._stage_timer("grounding_total"):
            return self._select_grounded_track_impl(rgb, tracks)

    def _select_grounded_track_impl(
        self, rgb: np.ndarray, tracks: List[CameraTrack]
    ) -> Tuple[Optional[CameraTrack], str]:
        if not tracks:
            self.grounding_score = None
            return None, "no_person_detected"
        from PIL import Image

        # Ground the person and the attribute as separate phrases, then require the
        # attribute to be found on a person. A full-sentence caption lets the word
        # "person" alone match every human.
        caption = "person." + (f" {self.attribute}." if self.attribute else "")
        inputs = self.grounding_processor(images=Image.fromarray(rgb), text=caption, return_tensors="pt").to(self.device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.device.startswith("cuda"),
        ):
            outputs = self.grounding_model(**inputs)
        grounded = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.grounding_box_threshold,
            text_threshold=self.grounding_text_threshold,
            target_sizes=[rgb.shape[:2]],
        )[0]
        labels = grounded["text_labels"] if "text_labels" in grounded else grounded.get("labels", [])
        attribute_words = set(self.attribute.split()) if self.attribute else set()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if self.color else None
        candidates: Dict[int, Tuple[float, CameraTrack]] = {}
        for index, (box_tensor, score_tensor) in enumerate(zip(grounded["boxes"], grounded["scores"])):
            box = box_tensor.detach().float().cpu().tolist()
            score = float(score_tensor.detach().float().cpu())
            label = str(labels[index]) if index < len(labels) else ""
            is_attribute = bool(attribute_words & set(label.lower().split()))
            entry = {"label": label, "score": round(score, 3), "bbox": [round(value, 1) for value in box]}
            if is_attribute and self.color:
                entry["color_fraction"] = round(color_fraction(bgr, box, self.color), 3)
            self.last_grounding.append(entry)
            if self.attribute and not is_attribute:
                continue
            if is_attribute and self.color and entry["color_fraction"] < self.color_fraction_threshold:
                continue
            for track in tracks:
                if is_attribute:
                    overlap = attribute_on_person(box, track.bbox)
                    if overlap < self.phrase_overlap_threshold or score < self.attribute_threshold:
                        continue
                else:
                    overlap = box_iou(box, track.bbox)
                    if overlap < self.phrase_overlap_threshold:
                        continue
                combined = score * (0.5 + 0.5 * overlap) * track.score
                if track.track_id not in candidates or combined > candidates[track.track_id][0]:
                    candidates[track.track_id] = (combined, track)
        ranked = sorted(candidates.values(), key=lambda item: item[0], reverse=True)
        self.last_candidates = [
            {"track_id": track.track_id, "combined": round(float(score), 3)} for score, track in ranked
        ]
        if not ranked:
            self.grounding_score = None
            if self.attribute:
                return None, f"no_person_with:{self.attribute}"
            return None, "prompt_match_not_on_a_person"
        best_score, best_track = ranked[0]
        self.grounding_score = float(best_score)
        if len(ranked) > 1 and best_score - ranked[1][0] < self.ambiguity_margin:
            return None, "multiple_plausible_people"
        return best_track, "prompt_match_found"

    def _annotate(
        self,
        bgr: np.ndarray,
        tracks: List[CameraTrack],
        selected: Optional[CameraTrack],
        valid: bool,
    ) -> np.ndarray:
        annotated = bgr.copy()
        selected_id = selected.track_id if selected is not None else None
        for track in tracks:
            x1, y1, x2, y2 = [int(round(float(value))) for value in track.bbox]
            is_selected = track.track_id == selected_id
            color = (0, 255, 0) if is_selected and valid else ((0, 165, 255) if is_selected else (150, 150, 150))
            thickness = 3 if is_selected else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                annotated,
                f"person id={track.track_id} {track.score:.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
            )
        for grounding in self.last_grounding:
            x1, y1, x2, y2 = [int(round(float(value))) for value in grounding["bbox"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(
                annotated,
                f"{grounding['label'][:28]} {grounding['score']:.2f}",
                (x1 + 2, min(annotated.shape[0] - 6, y2 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
            )
        state = self.lock.state
        state_color = (0, 255, 0) if valid else ((0, 165, 255) if state in ("SEARCHING", "UNCERTAIN") else (0, 0, 255))
        label = f"TARGET {state}: {self.lock.reason}"
        cv2.rectangle(annotated, (0, 0), (min(annotated.shape[1] - 1, 760), 34), (0, 0, 0), -1)
        cv2.putText(annotated, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, state_color, 2)
        return annotated
