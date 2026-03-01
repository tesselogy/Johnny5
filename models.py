
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class IdentityMatch:
    person_id: Optional[str]
    similarity: float
    person_position: str = "unknown"
    pose_state: str = "not_visible"
    eyes_state: str = "not_visible"
    asana: str = "unknown"
    body_parts: Dict[str, Optional[Tuple[int, int]]] = field(default_factory=dict)
    pose_label: str = "unknown"
    pose_confidence: float = 0.0
    pose_source: str = "none"


@dataclass
class ParticipantState:
    person_key: str
    person_id: Optional[str]
    current_track_id: str
    identity_status: str
    last_seen: float
    person_position: str = "unknown"
    pose_state: str = "not_visible"
    eyes_state: str = "not_visible"
    asana: str = "unknown"
    body_parts: Dict[str, Optional[Tuple[int, int]]] = field(default_factory=dict)
    pose_label: str = "unknown"
    pose_confidence: float = 0.0
    pose_source: str = "none"
    revision: int = 0


@dataclass
class SceneState:
    timestamp: float
    participants: Dict[str, ParticipantState] = field(default_factory=dict)
