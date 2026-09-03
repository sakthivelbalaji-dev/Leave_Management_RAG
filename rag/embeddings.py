from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(
        self,
        model_name="Qwen/Qwen3-Embedding-0.6B"
    ):
        print(f"Loading Qwen embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Qwen Embedding model loaded successfully.")

    def encode(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
