import json
import math
import os
from typing import Dict, List, Optional, Tuple


Point = Optional[Tuple[int, int]]


class PoseClassifier:
    """
    L2 yoga pose classifier with three levels:
    1) MLP model (`pose_mlp.pt`) on 17 normalized keypoints (x, y).
    2) Prototype angle matcher (mirror-aware).
    3) L1 heuristic fallback map.
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

    FALLBACK_MAP = {
        "mountain": "mountain_pose",
        "raised_hands": "raised_hands_pose",
        "t_pose": "warrior_2_like",
        "seated": "seated_pose",
        "unknown": "unknown",
    }

    def __init__(
        self,
        prototypes_path: str = "pose_prototypes.json",
        min_visible_keypoints: int = 6,
        mlp_model_path: str = "pose_mlp.pt",
        mlp_labels_path: str = "pose_labels.json",
    ):
        self.prototypes_path = prototypes_path
        self.min_visible_keypoints = min_visible_keypoints
        self.prototypes = self._load_prototypes(prototypes_path)

        self.mlp_model_path = mlp_model_path
        self.mlp_labels = self._load_mlp_labels(mlp_labels_path)
        self.mlp_model = None
        self.mlp_backend = None
        self._load_mlp_model(mlp_model_path)

    def _load_mlp_labels(self, path: str) -> List[str]:
        if not path or not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(v) for v in data]
            if isinstance(data, dict) and "labels" in data and isinstance(data["labels"], list):
                return [str(v) for v in data["labels"]]
        except Exception:
            return []
        return []

    def _load_mlp_model(self, model_path: str) -> None:
        if not model_path or not os.path.exists(model_path):
            return

        try:
            import torch
        except Exception:
            return

        # Prefer TorchScript for portability
        try:
            model = torch.jit.load(model_path, map_location="cpu")
            model.eval()
            self.mlp_model = model
            self.mlp_backend = "torchscript"
            return
        except Exception:
            pass

        try:
            model = torch.load(model_path, map_location="cpu")
            if hasattr(model, "eval"):
                model.eval()
            self.mlp_model = model
            self.mlp_backend = "torch"
        except Exception:
            self.mlp_model = None
            self.mlp_backend = None

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

        parsed_angles: Dict[str, float] = {}
        for angle_name in self.ANGLE_TRIPLETS:
            value = prototype_data.get(angle_name)
            if isinstance(value, (int, float)):
                parsed_angles[angle_name] = float(value)

        if parsed_angles:
            return parsed_angles

        keypoint_map: Dict[str, Point] = {}
        for name in self.KEYPOINT_ORDER:
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

    def _visible_points_count(self, body_parts: Dict[str, Point]) -> int:
        return sum(1 for p in body_parts.values() if p is not None)

    def _normalize_keypoints_xy(self, body_parts: Dict[str, Point]) -> Optional[List[float]]:
        points = [body_parts.get(k) for k in self.KEYPOINT_ORDER]
        visible = [p for p in points if p is not None]
        if len(visible) < self.min_visible_keypoints:
            return None

        xs = [p[0] for p in visible]
        ys = [p[1] for p in visible]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        w = max(1.0, float(max_x - min_x))
        h = max(1.0, float(max_y - min_y))

        out: List[float] = []
        for p in points:
            if p is None:
                out.extend([0.0, 0.0])
            else:
                out.extend([(p[0] - min_x) / w, (p[1] - min_y) / h])
        return out

    def _classify_mlp(self, body_parts: Dict[str, Point]) -> Optional[Tuple[str, float, str]]:
        if self.mlp_model is None:
            return None

        features = self._normalize_keypoints_xy(body_parts)
        if features is None:
            return None

        try:
            import torch
            x = torch.tensor([features], dtype=torch.float32)
            with torch.no_grad():
                logits = self.mlp_model(x)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                probs = torch.softmax(logits, dim=-1)
                conf, idx = torch.max(probs, dim=-1)
                idx_val = int(idx.item())
                conf_val = float(conf.item())

            if self.mlp_labels and 0 <= idx_val < len(self.mlp_labels):
                label = self.mlp_labels[idx_val]
            else:
                label = f"class_{idx_val}"

            return label, conf_val, "mlp_17xy"
        except Exception:
            return None

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

    def _mirror_angles(self, angles: Dict[str, float]) -> Dict[str, float]:
        mirrored: Dict[str, float] = {}
        for name, value in angles.items():
            if name.startswith("left_"):
                mirrored["right_" + name[5:]] = value
            elif name.startswith("right_"):
                mirrored["left_" + name[6:]] = value
            else:
                mirrored[name] = value
        return mirrored

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

    def distance_map(self, body_parts: Dict[str, Point]) -> Dict[str, float]:
        if not self.prototypes:
            return {}

        if self._visible_points_count(body_parts) < self.min_visible_keypoints:
            return {}

        observed_angles = self._extract_angles(body_parts)
        if not observed_angles:
            return {}

        mirrored_angles = self._mirror_angles(observed_angles)

        distances: Dict[str, float] = {}
        for label, prototype in self.prototypes.items():
            d_normal = self._angle_distance(observed_angles, prototype)
            d_mirror = self._angle_distance(mirrored_angles, prototype)
            best = min(d_normal, d_mirror)
            if math.isfinite(best):
                distances[label] = best

        return distances

    def classify(self, body_parts: Dict[str, Point], heuristic_asana: str) -> Tuple[str, float, str]:
        mlp_result = self._classify_mlp(body_parts)
        if mlp_result is not None:
            return mlp_result

        distances = self.distance_map(body_parts)
        if distances:
            best_label, best_dist = min(distances.items(), key=lambda x: x[1])
            confidence = max(0.0, min(1.0, 1.0 - best_dist))
            return best_label, confidence, "prototype_angles_mirror"

        fallback = self.FALLBACK_MAP.get(heuristic_asana, "unknown")
        confidence = 0.45 if fallback != "unknown" else 0.2
        return fallback, confidence, "heuristic"
