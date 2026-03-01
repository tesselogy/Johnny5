
from ultralytics import YOLO


class PersonDetector:

    def __init__(self, model_path="yolov8n-pose.pt"):
        self.model = YOLO(model_path)

        self.stream = self.model.track(
            source=0,
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

    def get_next(self):
        result = next(self.stream)

        frame = result.orig_img
        tracks = []

        if result.boxes is None or result.boxes.id is None:
            return frame, tracks

        keypoints_xy = None
        if result.keypoints is not None:
            keypoints_xy = result.keypoints.xy

        for i, (box, track_id, cls, conf) in enumerate(zip(
            result.boxes.xyxy,
            result.boxes.id,
            result.boxes.cls,
            result.boxes.conf
        )):
            if int(cls) != 0:
                continue

            x1, y1, x2, y2 = box.tolist()
            person_keypoints = None
            if keypoints_xy is not None and i < len(keypoints_xy):
                person_keypoints = keypoints_xy[i].tolist()

            tracks.append({
                "track_id": f"t{int(track_id)}",
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "conf": float(conf),
                "keypoints": person_keypoints,
            })

        return frame, tracks
