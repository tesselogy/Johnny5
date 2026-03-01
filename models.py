
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class IdentityMatch:
    person_id: Optional[str]
    similarity: float


@dataclass
class ParticipantState:
    person_key: str
    person_id: Optional[str]
    current_track_id: str
    identity_status: str
    last_seen: float
    revision: int = 0


@dataclass
class SceneState:
    timestamp: float
    participants: Dict[str, ParticipantState] = field(default_factory=dict)
