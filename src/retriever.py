
import faiss
import numpy as np

from src.embeddings import EmbeddingModel


class Retriever:
    """Index document chunks and retrieve relevant passages."""

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = []

    def index_chunks(self, chunks: list[dict]) -> None:
        """Build a fresh FAISS index from the supplied chunks."""
        if not chunks:
            raise ValueError("Cannot index an empty chunk list.")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)

        embeddings = np.ascontiguousarray(
            embeddings, dtype=np.float32
        )

        # Normalized vectors + inner product = cosine similarity.
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks = list(chunks)

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        """Return the highest-scoring chunks for a question."""
        if self.index is None:
            raise RuntimeError("Build the index before searching.")

        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self.embedding_model.encode(
            [question.strip()]
        )
        query_embedding = np.ascontiguousarray(
            query_embedding, dtype=np.float32
        )

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            results.append({
                **self.chunks[int(idx)],
                "similarity_score": float(score),
            })

        return results
