from ultralytics import YOLO


class PersonDetector:

    def __init__(self, model_path="l1/best.pt", confidence_threshold=0.6):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def run(self):
        for result in self.model(source=0, stream=True):
            probs = result.probs

            if probs is None:
                label = "uncertain"
                confidence = 0.0
            else:
                top1_idx = int(probs.top1)
                confidence = float(probs.top1conf.item())

                if confidence < self.confidence_threshold:
                    label = "uncertain"
                else:
                    label = self.model.names[top1_idx]

            print(f"Pose: {label} | Confidence: {confidence:.2f}")
