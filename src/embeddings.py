
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Generate normalized semantic embeddings for text."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]):
        """Return normalized embeddings as a NumPy array."""
        if not texts:
            raise ValueError("texts must not be empty.")

        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
