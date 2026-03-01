
import time
from config import TICK_INTERVAL


class ProjectionEngine:

    def __init__(self):
        self.last_tick = time.time()
        self.last_snapshot = None

    def maybe_project(self, scene):

        now = time.time()

        if now - self.last_tick < TICK_INTERVAL:
            return None

        self.last_tick = now

        snapshot = set(scene.participants.keys())

        if self.last_snapshot is None:
            self.last_snapshot = snapshot
            return None

        entered = snapshot - self.last_snapshot
        exited = self.last_snapshot - snapshot

        self.last_snapshot = snapshot

        if entered or exited:
            return {
                "entered": list(entered),
                "exited": list(exited)
            }

        return None
