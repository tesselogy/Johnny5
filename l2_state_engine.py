
import time
from models import ParticipantState, SceneState
from config import RECOGNITION_THRESHOLD, LOW_THRESHOLD, GRACE_PERIOD_SEC


class SceneEngine:

    def __init__(self):
        self.scene = SceneState(timestamp=time.time())

    def update(self, identity_results):

        now = time.time()
        active_keys = set()

        for track_id, match in identity_results:

            if match.person_id:
                person_key = match.person_id
            else:
                person_key = f"unknown:{track_id}"

            active_keys.add(person_key)

            if person_key not in self.scene.participants:
                self.scene.participants[person_key] = ParticipantState(
                    person_key=person_key,
                    person_id=match.person_id,
                    current_track_id=track_id,
                    identity_status="RECOGNIZED" if match.similarity >= 0.75 else "UNRECOGNIZED",
                    last_seen=now,
                    person_position=match.person_position,
                    pose_state=match.pose_state,
                    eyes_state=match.eyes_state,
                    asana=match.asana,
                    body_parts=match.body_parts
                )
            else:
                p = self.scene.participants[person_key]
                p.current_track_id = track_id
                p.last_seen = now
                p.person_position = match.person_position
                p.pose_state = match.pose_state
                p.eyes_state = match.eyes_state
                p.asana = match.asana
                p.body_parts = match.body_parts

        for key, participant in list(self.scene.participants.items()):
            if key not in active_keys:
                if participant.last_seen + GRACE_PERIOD_SEC < now:
                    del self.scene.participants[key]

        self.scene.timestamp = now
