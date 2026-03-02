
from ultralytics import YOLO


class PersonDetector:

    def __init__(self, model_path="yolov8s-pose.pt", tracker="botsort.yaml"):
        self.model = YOLO(model_path)

        self.stream = self.model.track(
            source=0,
            stream=True,
            persist=True,
            tracker=tracker,
            verbose=False
        )

    def get_next(self):
        result = next(self.stream)

        frame = result.orig_img
        tracks = []

        if result.boxes is None or result.boxes.id is None:
            return frame, tracks

        for box, track_id, cls, conf in zip(
            result.boxes.xyxy,
            result.boxes.id,
            result.boxes.cls,
            result.boxes.conf
        ):
            if int(cls) != 0:
                continue

            x1, y1, x2, y2 = box.tolist()

            tracks.append((
                f"t{int(track_id)}",
                int(x1), int(y1), int(x2), int(y2),
                float(conf)
            ))

        return frame, tracks
