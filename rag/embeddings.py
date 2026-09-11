from sentence_transformers import SentenceTransformer
import numpy as np
class EmbeddingModel:
    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        model_name=MODEL_NAME,
    ):
        self.model_name = model_name

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        self.dimension = self.model.get_sentence_embedding_dimension()

        print(
            "Embedding model loaded successfully. "
            f"Dimension: {self.dimension}"
        )

    def encode(self, texts):
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        return embeddings