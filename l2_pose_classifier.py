import json
import math
import os
from typing import Dict, Optional, Tuple


Point = Optional[Tuple[int, int]]


class PoseClassifier:
    """
    L2 yoga pose classifier with two modes:
    1) Prototype mode: nearest prototype loaded from JSON (recommended for Roboflow dataset).
    2) Fallback mode: map L1 heuristic asana labels.

    Expected JSON format:
    {
      "tree_pose": {"nose": [0.51, 0.13], "left_shoulder": [0.44, 0.29], ...},
      "warrior_2": {...}
    }
    Coordinates must be normalized to [0, 1] in person-centric crop space.
    """

    KEYPOINT_ORDER = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    FALLBACK_MAP = {
        "mountain": "mountain_pose",
        "raised_hands": "raised_hands_pose",
        "t_pose": "warrior_2_like",
        "seated": "seated_pose",
        "unknown": "unknown",
    }

    def __init__(self, prototypes_path: str = "pose_prototypes.json", min_visible_keypoints: int = 6):
        self.prototypes_path = prototypes_path
        self.min_visible_keypoints = min_visible_keypoints
        self.prototypes = self._load_prototypes(prototypes_path)

    def _load_prototypes(self, path: str) -> Dict[str, Dict[str, Tuple[float, float]]]:
        if not path or not os.path.exists(path):
            return {}

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        parsed = {}
        for label, point_map in raw.items():
            parsed[label] = {}
            for k in self.KEYPOINT_ORDER:
                if k in point_map and point_map[k] is not None:
                    x, y = point_map[k]
                    parsed[label][k] = (float(x), float(y))
        return parsed

    def _normalize_body_parts(self, body_parts: Dict[str, Point]) -> Dict[str, Tuple[float, float]]:
        pts = [(p[0], p[1]) for p in body_parts.values() if p is not None]
        if len(pts) < self.min_visible_keypoints:
            return {}

        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        w = max(1.0, float(max_x - min_x))
        h = max(1.0, float(max_y - min_y))

        norm = {}
        for k in self.KEYPOINT_ORDER:
            p = body_parts.get(k)
            if p is None:
                continue
            norm[k] = ((p[0] - min_x) / w, (p[1] - min_y) / h)
        return norm

    def _prototype_distance(self, norm_points: Dict[str, Tuple[float, float]], prototype: Dict[str, Tuple[float, float]]) -> float:
        common = [k for k in self.KEYPOINT_ORDER if k in norm_points and k in prototype]
        if len(common) < self.min_visible_keypoints:
            return float("inf")

        s = 0.0
        for k in common:
            dx = norm_points[k][0] - prototype[k][0]
            dy = norm_points[k][1] - prototype[k][1]
            s += dx * dx + dy * dy
        return math.sqrt(s / len(common))

    def classify(self, body_parts: Dict[str, Point], heuristic_asana: str) -> Tuple[str, float, str]:
        norm_points = self._normalize_body_parts(body_parts)

        if self.prototypes and norm_points:
            best_label = "unknown"
            best_dist = float("inf")

            for label, prototype in self.prototypes.items():
                dist = self._prototype_distance(norm_points, prototype)
                if dist < best_dist:
                    best_dist = dist
                    best_label = label

            if math.isfinite(best_dist):
                confidence = max(0.0, min(1.0, 1.0 - best_dist))
                return best_label, confidence, "prototype"

        fallback = self.FALLBACK_MAP.get(heuristic_asana, "unknown")
        confidence = 0.45 if fallback != "unknown" else 0.2
        return fallback, confidence, "heuristic"
