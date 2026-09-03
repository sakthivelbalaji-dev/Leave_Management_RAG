from .loader import KnowledgeBaseLoader
from .chunker import TextChunker
from .embeddings import EmbeddingModel
from .vector_store import FAISSVectorStore


class Retriever:

    def __init__(
        self,
        minimum_score: float = 0.25,
    ):
        self.minimum_score = minimum_score

        # --------------------------------------------------------
        # LOAD POLICY DOCUMENTS
        # --------------------------------------------------------

        loader = KnowledgeBaseLoader()

        documents = loader.load_documents()

        if not documents:
            raise RuntimeError(
                "No documents found in knowledge_base/. "
                "Add a .txt policy document."
            )

        # --------------------------------------------------------
        # CHUNK DOCUMENTS
        # --------------------------------------------------------

        chunker = TextChunker()

        chunks = chunker.chunk_documents(documents)

        if not chunks:
            raise RuntimeError(
                "Knowledge base contains no usable chunks."
            )

        # --------------------------------------------------------
        # EMBEDDINGS
        # --------------------------------------------------------

        self.embedding_model = EmbeddingModel()

        embeddings = self.embedding_model.encode(
            [
                item["content"]
                for item in chunks
            ]
        )

        # --------------------------------------------------------
        # VECTOR STORE
        # --------------------------------------------------------

        self.vector_store = FAISSVectorStore(
            embeddings.shape[1]
        )

        self.vector_store.add(
            embeddings,
            chunks
        )

        # --------------------------------------------------------
        # INFORMATION
        # --------------------------------------------------------

        print(
            f"RAG initialized with "
            f"{len(documents)} document(s) "
            f"and {len(chunks)} chunk(s)."
        )

        print(
            f"Minimum retrieval similarity: "
            f"{self.minimum_score}"
        )

    # ============================================================
    # RETRIEVE
    # ============================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ):

        if not query or not query.strip():
            return []

        # --------------------------------------------------------
        # EMBED USER QUERY
        # --------------------------------------------------------

        query_embedding = self.embedding_model.encode(
            [query.strip()]
        )[0]

        # --------------------------------------------------------
        # VECTOR SEARCH
        # --------------------------------------------------------

        results = self.vector_store.search(
            query_embedding,
            top_k,
        )

        # --------------------------------------------------------
        # FILTER LOW SIMILARITY
        # --------------------------------------------------------

        filtered = [
            item
            for item in results
            if float(item.get("score", 0.0))
            >= self.minimum_score
        ]

        # --------------------------------------------------------
        # DEBUG INFORMATION
        # --------------------------------------------------------

        print("\n" + "=" * 70)
        print("[RAG RETRIEVAL]")
        print(f"Query: {query}")
        print(f"Requested top_k: {top_k}")
        print(f"Results before filtering: {len(results)}")
        print(f"Results after filtering: {len(filtered)}")

        for index, item in enumerate(results, start=1):

            print(
                f"\nResult {index}"
                f"\nScore: {float(item.get('score', 0.0)):.4f}"
                f"\nSource: {item.get('source', 'unknown')}"
            )

            print(
                f"Content preview:\n"
                f"{item.get('content', '')[:500]}"
            )

        print("=" * 70 + "\n")

        return filtered