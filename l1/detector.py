from ultralytics import YOLO


class PersonDetector:

    def __init__(self, model_path="l1/best.pt", confidence_threshold=0.6):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.stream = self.model(source=0, stream=True)

    def get_next(self):
        result = next(self.stream)
        frame = result.orig_img

        probs = result.probs
        if probs is None:
            return frame, []

        top1_idx = int(probs.top1)
        confidence = float(probs.top1conf.item())

        if confidence < self.confidence_threshold:
            return frame, []

        h, w = frame.shape[:2]
        label = self.model.names[top1_idx]

        tracks = [{
            "track_id": "t0",
            "bbox": (0, 0, int(w), int(h)),
            "conf": confidence,
            "keypoints": None,
            "label": label,
        }]

        return frame, tracks

    def run(self):
        while True:
            _, tracks = self.get_next()
            if not tracks:
                print("Pose: uncertain | Confidence: 0.00")
            else:
                track = tracks[0]
                print(f"Pose: {track['label']} | Confidence: {track['conf']:.2f}")
