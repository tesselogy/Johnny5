# l1/perception.py

import cv2
import uuid
import time
import numpy as np

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

        print(f"[IdentityStore] Loaded {len(existing)} persons")

    # -------------------------------------------------
    # Debug overlay
    # -------------------------------------------------
    def draw_overlay(self, frame, track_id, bbox, similarity, person_id):

        x1, y1, x2, y2 = bbox

        if person_id:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{track_id} | {person_id} | {similarity:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # -------------------------------------------------
    # Main loop
    # -------------------------------------------------
    def process_frame(self):

        frame, tracks = self.detector.get_next()
        identity_results = []
        now = time.time()

        for track_id, x1, y1, x2, y2, conf in tracks:

            embedding, quality = self.encoder.extract(
                frame,
                (x1, y1, x2, y2)
            )

            if embedding is None:
                identity_results.append(
                    (track_id, IdentityMatch(None, 0.0))
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
                (track_id, IdentityMatch(person_id, similarity))
            )

            self.draw_overlay(
                frame,
                track_id,
                (x1, y1, x2, y2),
                similarity,
                person_id
            )

        active_track_ids = {track_id for track_id, *_ in tracks}

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
