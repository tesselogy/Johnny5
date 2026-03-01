import faiss
import numpy as np


class IdentityIndex:

    def __init__(self, dim=512):
        # Cosine similarity on normalized embeddings via inner product.
        self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = 64
        self.person_ids = []

    def add_embedding(self, person_id, embedding):
        self.index.add(np.array([embedding]).astype("float32"))
        self.person_ids.append(person_id)

    def search(self, embedding, top_k=3, candidate_k=20):

        if self.index.ntotal == 0:
            return []

        k = min(max(top_k, candidate_k), self.index.ntotal)
        D, I = self.index.search(
            np.array([embedding]).astype("float32"),
            k,
        )

        # Aggregate vector-level hits into person-level score.
        # This stabilizes matches when each person has many embeddings.
        grouped = {}
        for sim, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.person_ids):
                continue

            person_id = self.person_ids[idx]
            grouped.setdefault(person_id, []).append(float(sim))

        results = []
        for person_id, sims in grouped.items():
            sims.sort(reverse=True)
            top1 = sims[0]
            top2 = sims[1] if len(sims) > 1 else sims[0]
            support_bonus = min(0.03 * len(sims), 0.12)
            score = 0.75 * top1 + 0.25 * top2 + support_bonus
            results.append((person_id, float(min(score, 1.0))))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
