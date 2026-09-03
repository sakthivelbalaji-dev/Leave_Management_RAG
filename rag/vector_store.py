import faiss
import numpy as np


class FAISSVectorStore:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.items = []

    def add(self, embeddings, items):
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index.add(embeddings)
        self.items.extend(items)

    def search(self, query_embedding, top_k=3):
        if not self.items or self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        ).reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal),
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            item = dict(self.items[idx])
            item["score"] = float(score)
            results.append(item)

        return results
