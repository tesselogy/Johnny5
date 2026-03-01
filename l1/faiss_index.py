
import faiss
import numpy as np


class IdentityIndex:

    def __init__(self, dim=512):
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.index.hnsw.efSearch = 64
        self.person_ids = []

    def add_embedding(self, person_id, embedding):
        self.index.add(np.array([embedding]).astype("float32"))
        self.person_ids.append(person_id)

    def search(self, embedding, top_k=3):

        if self.index.ntotal == 0:
            return []

        D, I = self.index.search(
            np.array([embedding]).astype("float32"),
            top_k
        )

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(self.person_ids):
                results.append((self.person_ids[idx], float(dist)))

        return results
