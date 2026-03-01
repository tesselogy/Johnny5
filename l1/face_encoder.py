
import numpy as np
from insightface.app import FaceAnalysis


class FaceEncoder:

    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def extract(self, frame, bbox):

        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None, None

        faces = self.app.get(crop)
        if not faces:
            return None, None

        face = max(
            faces,
            key=lambda f: (f.bbox[2]-f.bbox[0]) *
                          (f.bbox[3]-f.bbox[1])
        )

        embedding = face.embedding.astype("float32")
        embedding /= np.linalg.norm(embedding)

        return embedding, float(face.det_score)
