import numpy as np

class EndeeVectorClient:
    """
    Logical abstraction of Endee Vector Database.
    Handles vector insertion and similarity search.
    """

    def __init__(self):
        self.vectors = []
        self.texts = []

    def insert(self, embeddings, texts):
        for emb, txt in zip(embeddings, texts):
            self.vectors.append(emb)
            self.texts.append(txt)

    def search(self, query_embedding, top_k=3):
        scores = []

        for idx, vec in enumerate(self.vectors):
            similarity = np.dot(query_embedding, vec) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(vec)
            )
            scores.append((similarity, self.texts[idx]))

        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[:top_k]
