
import time
from models import ParticipantState, SceneState
from config import (
    GRACE_PERIOD_SEC,
    POSE_MIN_VISIBLE_KEYPOINTS,
    POSE_PROTOTYPES_PATH,
    POSE_DISTANCE_EMA_ALPHA,
    POSE_SWITCH_MARGIN,
)
from l2_pose_classifier import PoseClassifier


class SceneEngine:

    def __init__(self):
        self.scene = SceneState(timestamp=time.time())
        self.pose_classifier = PoseClassifier(
            prototypes_path=POSE_PROTOTYPES_PATH,
            min_visible_keypoints=POSE_MIN_VISIBLE_KEYPOINTS,
        )
        self.l2_pose_log_memory = {}
        self.pose_ema_memory = {}

    def update(self, identity_results):

        now = time.time()
        active_keys = set()

        for track_id, match in identity_results:

            if match.person_id:
                person_key = match.person_id
            else:
                person_key = f"unknown:{track_id}"

            active_keys.add(person_key)

            distances = self.pose_classifier.distance_map(match.body_parts)

            pose_label = "unknown"
            pose_confidence = 0.0
            pose_source = "prototype_angles_mirror_ema"

            if distances:
                state = self.pose_ema_memory.get(track_id, {"ema": {}, "label": None})
                ema = state["ema"]

                # update EMA for seen labels
                for label, dist in distances.items():
                    if label in ema:
                        ema[label] = (
                            POSE_DISTANCE_EMA_ALPHA * dist
                            + (1.0 - POSE_DISTANCE_EMA_ALPHA) * ema[label]
                        )
                    else:
                        ema[label] = dist

                # soft decay for labels not seen in current frame
                for label in list(ema.keys()):
                    if label not in distances:
                        ema[label] = min(1.0, ema[label] + 0.03)

                candidate_label, candidate_dist = min(ema.items(), key=lambda x: x[1])
                prev_label = state.get("label")

                if prev_label in ema:
                    prev_dist = ema[prev_label]
                    if candidate_label != prev_label and candidate_dist > prev_dist - POSE_SWITCH_MARGIN:
                        pose_label = prev_label
                    else:
                        pose_label = candidate_label
                else:
                    pose_label = candidate_label

                state["label"] = pose_label
                self.pose_ema_memory[track_id] = state

                effective_dist = ema.get(pose_label, candidate_dist)
                pose_confidence = max(0.0, min(1.0, 1.0 - effective_dist))
            else:
                pose_label, pose_confidence, pose_source = self.pose_classifier.classify(
                    match.body_parts,
                    match.asana,
                )

            prev_l2 = self.l2_pose_log_memory.get(track_id)
            curr_l2 = {
                "pose_label": pose_label,
                "pose_source": pose_source,
            }
            if prev_l2 is None:
                print(
                    f"[PoseDebugL2:init] track={track_id} person={match.person_id} "
                    f"pose_label={pose_label} confidence={pose_confidence:.2f} source={pose_source}"
                )
            elif (
                prev_l2.get("pose_label") != pose_label
                or prev_l2.get("pose_source") != pose_source
            ):
                print(
                    f"[PoseDebugL2:change] track={track_id} person={match.person_id} "
                    f"pose_label:{prev_l2.get('pose_label')}->{pose_label} "
                    f"source:{prev_l2.get('pose_source')}->{pose_source} "
                    f"confidence={pose_confidence:.2f}"
                )
            self.l2_pose_log_memory[track_id] = curr_l2

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
                    body_parts=match.body_parts,
                    pose_label=pose_label,
                    pose_confidence=pose_confidence,
                    pose_source=pose_source
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
                p.pose_label = pose_label
                p.pose_confidence = pose_confidence
                p.pose_source = pose_source

        for key, participant in list(self.scene.participants.items()):
            if key not in active_keys:
                if participant.last_seen + GRACE_PERIOD_SEC < now:
                    del self.scene.participants[key]

        active_track_ids = {track_id for track_id, _ in identity_results}
        for track_id in list(self.l2_pose_log_memory.keys()):
            if track_id not in active_track_ids:
                del self.l2_pose_log_memory[track_id]

        for track_id in list(self.pose_ema_memory.keys()):
            if track_id not in active_track_ids:
                del self.pose_ema_memory[track_id]

        self.scene.timestamp = now
