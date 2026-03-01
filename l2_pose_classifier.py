import json
import math
import os
from typing import Dict, Optional, Tuple


Point = Optional[Tuple[int, int]]


class PoseClassifier:
    """
    L2 yoga pose classifier with two modes:
    1) Prototype mode: nearest angle-prototype loaded from JSON (recommended for Roboflow dataset).
    2) Fallback mode: map L1 heuristic asana labels.

    Expected JSON format:
    {
      "tree_pose": {
        "left_elbow": 168.5,
        "right_elbow": 172.0,
        "left_knee": 48.2,
        "right_knee": 177.1
      },
      "warrior_2": {...}
    }

    Angles are in degrees [0..180].
    """

    ANGLE_TRIPLETS = {
        "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
        "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
        "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
        "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
        "left_hip": ("left_shoulder", "left_hip", "left_knee"),
        "right_hip": ("right_shoulder", "right_hip", "right_knee"),
        "left_knee": ("left_hip", "left_knee", "left_ankle"),
        "right_knee": ("right_hip", "right_knee", "right_ankle"),
    }

    KEYPOINT_NAMES = {
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
    }

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

    def _load_prototypes(self, path: str) -> Dict[str, Dict[str, float]]:
        if not path or not os.path.exists(path):
            return {}

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        parsed: Dict[str, Dict[str, float]] = {}
        for label, prototype_data in raw.items():
            parsed[label] = self._parse_single_prototype(prototype_data)

        return parsed

    def _parse_single_prototype(self, prototype_data: Dict[str, object]) -> Dict[str, float]:
        if not isinstance(prototype_data, dict):
            return {}

        # Format A (new): {"left_knee": 92.0, ...}
        angle_like = all(
            (k in self.ANGLE_TRIPLETS) or (k in self.KEYPOINT_NAMES)
            for k in prototype_data.keys()
        )

        parsed_angles: Dict[str, float] = {}
        for angle_name in self.ANGLE_TRIPLETS:
            value = prototype_data.get(angle_name)
            if isinstance(value, (int, float)):
                parsed_angles[angle_name] = float(value)

        # If we already have explicit angles, use them directly.
        if parsed_angles:
            return parsed_angles

        # Format B (legacy): {"left_shoulder": [x,y], ...}
        # Convert coordinate prototype to angle prototype for backward compatibility.
        if angle_like:
            keypoint_map: Dict[str, Point] = {}
            for name in self.KEYPOINT_NAMES:
                point = prototype_data.get(name)
                if (
                    isinstance(point, (list, tuple))
                    and len(point) >= 2
                    and isinstance(point[0], (int, float))
                    and isinstance(point[1], (int, float))
                ):
                    keypoint_map[name] = (int(point[0] * 1000), int(point[1] * 1000))
                else:
                    keypoint_map[name] = None

            return self._extract_angles(keypoint_map)

        return {}

    def _visible_points_count(self, body_parts: Dict[str, Point]) -> int:
        return sum(1 for p in body_parts.values() if p is not None)

    def _joint_angle(self, a: Point, b: Point, c: Point) -> Optional[float]:
        if a is None or b is None or c is None:
            return None

        bax = float(a[0] - b[0])
        bay = float(a[1] - b[1])
        bcx = float(c[0] - b[0])
        bcy = float(c[1] - b[1])

        norm_ba = math.hypot(bax, bay)
        norm_bc = math.hypot(bcx, bcy)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return None

        cos_theta = (bax * bcx + bay * bcy) / (norm_ba * norm_bc)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta = math.degrees(math.acos(cos_theta))
        return theta

    def _extract_angles(self, body_parts: Dict[str, Point]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for angle_name, (a_name, b_name, c_name) in self.ANGLE_TRIPLETS.items():
            angle = self._joint_angle(
                body_parts.get(a_name),
                body_parts.get(b_name),
                body_parts.get(c_name),
            )
            if angle is not None:
                out[angle_name] = angle
        return out

    def _angle_distance(self, observed: Dict[str, float], prototype: Dict[str, float]) -> float:
        common = [k for k in self.ANGLE_TRIPLETS if k in observed and k in prototype]
        if len(common) < 3:
            return float("inf")

        s = 0.0
        for k in common:
            diff = abs(observed[k] - prototype[k])
            diff = min(diff, 360.0 - diff)
            s += (diff / 180.0) ** 2
        return math.sqrt(s / len(common))

    def classify(self, body_parts: Dict[str, Point], heuristic_asana: str) -> Tuple[str, float, str]:
        if self._visible_points_count(body_parts) >= self.min_visible_keypoints:
            observed_angles = self._extract_angles(body_parts)
        else:
            observed_angles = {}

        if self.prototypes and observed_angles:
            best_label = "unknown"
            best_dist = float("inf")

            for label, prototype in self.prototypes.items():
                dist = self._angle_distance(observed_angles, prototype)
                if dist < best_dist:
                    best_dist = dist
                    best_label = label

            if math.isfinite(best_dist):
                confidence = max(0.0, min(1.0, 1.0 - best_dist))
                return best_label, confidence, "prototype_angles"

        fallback = self.FALLBACK_MAP.get(heuristic_asana, "unknown")
        confidence = 0.45 if fallback != "unknown" else 0.2
        return fallback, confidence, "heuristic"
