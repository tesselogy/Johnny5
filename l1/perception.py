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

            person_id = None
            similarity = best_similarity

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

        cv2.imshow("Johnny5 Vision", frame)
        cv2.waitKey(1)

        return identity_results