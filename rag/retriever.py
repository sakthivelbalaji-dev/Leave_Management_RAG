import os

from .loader import KnowledgeBaseLoader
from .chunker import TextChunker
from .embeddings import EmbeddingModel
from .vector_store import FAISSVectorStore


class Retriever:

    def __init__(
        self,
        minimum_score: float | None = None,
    ):
        configured_score = os.getenv("RAG_MINIMUM_SCORE", "0.15")
        self.minimum_score = (
            float(configured_score)
            if minimum_score is None
            else float(minimum_score)
        )

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
        self.documents = documents
        self.chunks = chunks

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
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print("Knowledge-base chunks:")
        for chunk in chunks:
            preview = chunk["content"][:120].replace("\n", " ")
            print(
                f"  {chunk['chunk_id']} | {chunk['source']} | {preview}"
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
            if float(
                item.get(
                    "score",
                    0.0
                )
            ) >= self.minimum_score
        ]

        # --------------------------------------------------------
        # DEBUG INFORMATION
        # --------------------------------------------------------

        print("\n" + "=" * 80)
        print("========== AI / RAG DEBUG ==========")

        print(
            "USER QUESTION:",
            query
        )

        print(
            "REQUESTED TOP_K:",
            top_k
        )

        print(
            "RESULTS BEFORE FILTERING:",
            len(results)
        )

        print(
            "QUERY EMBEDDING DIMENSION:",
            query_embedding.shape[0]
        )

        print(
            "MINIMUM SCORE:",
            self.minimum_score
        )

        print("\nRAW RESULTS:")
        for index, chunk in enumerate(results, start=1):
            print(
                f"RAW RESULT {index}: "
                f"id={chunk.get('chunk_id')} "
                f"score={float(chunk.get('score', 0.0)):.6f} "
                f"source={chunk.get('source', 'unknown')}"
            )
            print("CONTENT:")
            print(chunk.get("content", ""))

        print(
            "RESULTS AFTER FILTERING:",
            len(filtered)
        )

        print("\nRETRIEVED CHUNKS:")
        print("-" * 80)

        if filtered:

            for index, chunk in enumerate(
                filtered,
                start=1
            ):

                print(
                    f"\nCHUNK #{index}"
                )

                print(
                    "Chunk ID:",
                    chunk.get(
                        "chunk_id",
                        "N/A"
                    )
                )

                print(
                    "Score:",
                    f"{float(chunk.get('score', 0.0)):.4f}"
                )

                print(
                    "Source:",
                    chunk.get(
                        "source",
                        "unknown"
                    )
                )

                print("\nCONTENT:")

                print(
                    chunk.get(
                        "content",
                        ""
                    )
                )

                print("-" * 80)

        else:

            print(
                "NO RELEVANT CHUNKS FOUND."
            )

        print(
            "\nTOTAL RETRIEVED CHUNKS:",
            len(filtered)
        )

        print(
            "\n========== END AI / RAG DEBUG =========="
        )

        print("=" * 80 + "\n")

        return filtered