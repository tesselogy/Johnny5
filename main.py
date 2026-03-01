
import time
from config import FRAME_INTERVAL
from l1.perception import Perception
from l2_state_engine import SceneEngine
from l3_projection import ProjectionEngine
from l4_persistence import Persistence


def main():

    perception = Perception()
    scene_engine = SceneEngine()
    projection = ProjectionEngine()
    persistence = Persistence()

    while True:

        identity_results = perception.process_frame()

        scene_engine.update(identity_results)

        delta = projection.maybe_project(scene_engine.scene)
        persistence.handle_delta(delta)

        time.sleep(FRAME_INTERVAL)


if __name__ == "__main__":
    main()
