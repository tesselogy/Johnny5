# l1/perception.py

import cv2
import uuid
import time
import numpy as np
from typing import Dict, Optional, Tuple

from l1.detector import PersonDetector
from l1.face_encoder import FaceEncoder
from l1.faiss_index import IdentityIndex
from models import IdentityMatch
from identity_store import IdentityStore


RECOGNITION_THRESHOLD = 0.75
LOW_THRESHOLD = 0.60
SINGLE_PERSON_FALLBACK = 0.55
STICKINESS_THRESHOLD = 0.40
ENROLLMENT_TIME = 3.0
EMA_ALPHA = 0.35
ACCUMULANCE_STEP_UP = 1.0
ACCUMULANCE_STEP_DOWN = 0.5
ACCUMULANCE_RECOGNIZE = 3.0
TRACK_STALE_SEC = 2.0
PROFILE_UPDATE_THRESHOLD = 0.82
PROFILE_UPDATE_INTERVAL_SEC = 8.0


COCO_KEYPOINTS = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}


class Perception:

    def __init__(self):

        self.detector = PersonDetector()
        self.encoder = FaceEncoder()
        self.index = IdentityIndex()
        self.store = IdentityStore()

        # track_id -> person_id
        self.identity_memory = {}

        # enrollment buffer
        self.enrollment_buffer = {}

        # track_id -> temporal recognition state
        self.track_state = {}

        # person_id -> timestamp of last profile vector append
        self.profile_update_ts = {}

        # load stored embeddings
        existing = self.store.load_all()
        for person_id, embedding in existing:
            self.index.add_embedding(person_id, embedding)

        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )

        print(f"[IdentityStore] Loaded {len(existing)} persons")


    def _estimate_person_position(self, frame_shape, bbox):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) / 2.0
        box_h = max(1.0, y2 - y1)
        ratio = box_h / max(1.0, h)

        horizontal = "center"
        if cx < w * 0.33:
            horizontal = "left"
        elif cx > w * 0.67:
            horizontal = "right"

        depth = "middle"
        if ratio >= 0.55:
            depth = "close"
        elif ratio <= 0.28:
            depth = "far"

        return f"{depth}_{horizontal}"

    def _keypoints_dict(self, keypoints):
        mapped = {}

        if not keypoints:
            for _, name in COCO_KEYPOINTS.items():
                mapped[name] = None
            return mapped

        for idx, name in COCO_KEYPOINTS.items():
            if idx < len(keypoints):
                x, y = keypoints[idx][:2]
                if x > 0 and y > 0:
                    mapped[name] = (int(x), int(y))
                else:
                    mapped[name] = None
            else:
                mapped[name] = None

        return mapped

    def _classify_pose_and_asana(self, body_parts):
        left_hip = body_parts.get("left_hip")
        right_hip = body_parts.get("right_hip")
        left_knee = body_parts.get("left_knee")
        right_knee = body_parts.get("right_knee")
        left_ankle = body_parts.get("left_ankle")
        right_ankle = body_parts.get("right_ankle")
        left_wrist = body_parts.get("left_wrist")
        right_wrist = body_parts.get("right_wrist")
        left_shoulder = body_parts.get("left_shoulder")
        right_shoulder = body_parts.get("right_shoulder")

        pose_state = "not_visible"
        asana = "unknown"

        lower_points = [left_hip, right_hip, left_knee, right_knee]
        if any(p is not None for p in lower_points):
            pose_state = "standing"

        if all(p is not None for p in [left_hip, right_hip, left_knee, right_knee]):
            hip_y = (left_hip[1] + right_hip[1]) / 2.0
            knee_y = (left_knee[1] + right_knee[1]) / 2.0
            if abs(knee_y - hip_y) < 45:
                pose_state = "sitting"

        if all(p is not None for p in [left_ankle, right_ankle, left_hip, right_hip]):
            hip_y = (left_hip[1] + right_hip[1]) / 2.0
            ankle_y = (left_ankle[1] + right_ankle[1]) / 2.0
            if ankle_y - hip_y > 120:
                asana = "mountain"

        if all(p is not None for p in [left_wrist, right_wrist, left_shoulder, right_shoulder]):
            if left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]:
                asana = "raised_hands"
            elif abs(left_wrist[1] - left_shoulder[1]) < 45 and abs(right_wrist[1] - right_shoulder[1]) < 45:
                asana = "t_pose"

        if pose_state == "sitting" and asana == "unknown":
            asana = "seated"

        return pose_state, asana

    def _classify_eyes(self, frame, bbox, body_parts):
        left_eye = body_parts.get("left_eye")
        right_eye = body_parts.get("right_eye")

        if left_eye is None and right_eye is None:
            return "not_visible"

        x1, y1, x2, y2 = bbox
        head_h = max(1, int((y2 - y1) * 0.38))
        face_roi = frame[max(0, y1):max(0, y1) + head_h, max(0, x1):max(0, x2)]

        if face_roi.size == 0:
            return "not_visible"

        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(10, 10)
        )

        return "open" if len(eyes) >= 1 else "closed_or_not_detected"

    # -------------------------------------------------
    # Debug overlay
    # -------------------------------------------------
    def draw_overlay(self, frame, track_id, bbox, similarity, person_id, person_position, pose_state, eyes_state, asana):

        x1, y1, x2, y2 = bbox

        if person_id:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{track_id} | {person_id} | sim:{similarity:.2f}"
        debug_line = f"{person_position} | pose:{pose_state} | eyes:{eyes_state} | asana:{asana}"

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 26)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        cv2.putText(
            frame,
            debug_line,
            (x1, max(40, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        )

    # -------------------------------------------------
    # Main loop
    # -------------------------------------------------
    def process_frame(self):

        frame, tracks = self.detector.get_next()
        identity_results = []
        now = time.time()

        for track in tracks:

            track_id = track["track_id"]
            x1, y1, x2, y2 = track["bbox"]
            keypoints = track.get("keypoints")

            body_parts = self._keypoints_dict(keypoints)
            person_position = self._estimate_person_position(frame.shape, (x1, y1, x2, y2))
            pose_state, asana = self._classify_pose_and_asana(body_parts)
            eyes_state = self._classify_eyes(frame, (x1, y1, x2, y2), body_parts)

            embedding, quality = self.encoder.extract(
                frame,
                (x1, y1, x2, y2)
            )

            if embedding is None:
                identity_results.append(
                    (track_id, IdentityMatch(None, 0.0, person_position, pose_state, eyes_state, asana, body_parts))
                )
                continue

            matches = self.index.search(embedding)

            if matches:
                best_person, best_similarity = matches[0]
            else:
                best_person, best_similarity = None, 0.0

            state = self.track_state.get(track_id)
            if state is None:
                state = {
                    "ema_similarity": best_similarity,
                    "accumulance": 0.0,
                    "candidate_person": best_person,
                    "last_seen": now,
                }
            else:
                state["ema_similarity"] = (
                    EMA_ALPHA * best_similarity
                    + (1.0 - EMA_ALPHA) * state["ema_similarity"]
                )
                state["last_seen"] = now

            if best_person and best_similarity >= LOW_THRESHOLD:
                if state["candidate_person"] != best_person:
                    state["candidate_person"] = best_person
                    state["accumulance"] = 0.0
                state["accumulance"] += ACCUMULANCE_STEP_UP
            else:
                state["accumulance"] = max(
                    0.0,
                    state["accumulance"] - ACCUMULANCE_STEP_DOWN
                )

            self.track_state[track_id] = state

            person_id = None
            similarity = state["ema_similarity"]

            total_identities = self.index.index.ntotal

            # -------------------------------------------------
            # 1. SINGLE PERSON FALLBACK
            # -------------------------------------------------
            if (
                total_identities == 1
                and best_similarity >= SINGLE_PERSON_FALLBACK
            ):
                person_id = best_person
                self.identity_memory[track_id] = person_id

            # -------------------------------------------------
            # 2. NORMAL RECOGNITION
            # -------------------------------------------------
            elif best_similarity >= RECOGNITION_THRESHOLD:
                person_id = best_person
                self.identity_memory[track_id] = person_id

            # -------------------------------------------------
            # 2b. ACCUMULANCE GATE (stability before accept)
            # -------------------------------------------------
            elif (
                state["candidate_person"]
                and state["accumulance"] >= ACCUMULANCE_RECOGNIZE
                and similarity >= LOW_THRESHOLD
            ):
                person_id = state["candidate_person"]
                self.identity_memory[track_id] = person_id

            # -------------------------------------------------
            # 3. STICKINESS (temporal smoothing)
            # -------------------------------------------------
            elif (
                track_id in self.identity_memory
                and best_similarity >= STICKINESS_THRESHOLD
            ):
                person_id = self.identity_memory[track_id]

            # -------------------------------------------------
            # 4. UNKNOWN + FIRST ENROLL
            # -------------------------------------------------
            else:

                # Allow enrollment only if DB empty
                if total_identities == 0:

                    if track_id not in self.enrollment_buffer:
                        self.enrollment_buffer[track_id] = (
                            embedding,
                            now
                        )

                    first_seen = self.enrollment_buffer[track_id][1]

                    if now - first_seen > ENROLLMENT_TIME:

                        person_id = f"p_{uuid.uuid4().hex[:8]}"

                        self.index.add_embedding(person_id, embedding)
                        self.store.insert_person(person_id, embedding)

                        self.identity_memory[track_id] = person_id

                        print(f"[Auto-enroll] Created {person_id}")

                        del self.enrollment_buffer[track_id]

                        similarity = 1.0

            if (
                person_id
                and similarity >= PROFILE_UPDATE_THRESHOLD
                and quality is not None
                and quality >= 0.6
            ):
                last_update = self.profile_update_ts.get(person_id, 0.0)
                if now - last_update >= PROFILE_UPDATE_INTERVAL_SEC:
                    self.store.insert_person_embedding(person_id, embedding)
                    self.index.add_embedding(person_id, embedding)
                    self.profile_update_ts[person_id] = now

            identity_results.append(
                (track_id, IdentityMatch(person_id, similarity, person_position, pose_state, eyes_state, asana, body_parts))
            )

            self.draw_overlay(
                frame,
                track_id,
                (x1, y1, x2, y2),
                similarity,
                person_id,
                person_position,
                pose_state,
                eyes_state,
                asana
            )

        active_track_ids = {track["track_id"] for track in tracks}

        for track_id in list(self.identity_memory.keys()):
            if track_id not in active_track_ids:
                last_seen = self.track_state.get(track_id, {}).get("last_seen", now)
                if now - last_seen > TRACK_STALE_SEC:
                    del self.identity_memory[track_id]

        for track_id in list(self.track_state.keys()):
            if track_id not in active_track_ids:
                if now - self.track_state[track_id]["last_seen"] > TRACK_STALE_SEC:
                    del self.track_state[track_id]

        cv2.imshow("Johnny5 Vision", frame)
        cv2.waitKey(1)

        return identity_results
