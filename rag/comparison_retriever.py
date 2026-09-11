'''import json
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from .chunker import TextChunker
from .loader import KnowledgeBaseLoader
from .vector_store import FAISSVectorStore


class EmbeddingComparisonRetriever:
    QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    BAAI_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self):
        documents = KnowledgeBaseLoader().load_documents()
        if not documents:
            raise RuntimeError("No documents found in knowledge_base/.")

        chunks = TextChunker().chunk_documents(documents)
        if not chunks:
            raise RuntimeError("Knowledge base contains no usable chunks.")

        for index, chunk in enumerate(chunks):
            chunk["chunk_id"] = f"chunk_{index}"

        self.chunks = chunks
        self.evaluation_cases = self._load_evaluation_cases()
        texts = [chunk.get("content", "") for chunk in chunks]

        print("Loading Qwen/Qwen3-Embedding-0.6B...")
        self.qwen_model = SentenceTransformer(self.QWEN_MODEL)
        print("Loading BAAI/bge-small-en-v1.5...")
        self.bge_model = SentenceTransformer(self.BAAI_MODEL)

        qwen_embeddings = self.qwen_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")
        bge_embeddings = self.bge_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        self.qwen_store = FAISSVectorStore(qwen_embeddings.shape[1])
        self.bge_store = FAISSVectorStore(bge_embeddings.shape[1])
        self.qwen_store.add(qwen_embeddings, chunks)
        self.bge_store.add(bge_embeddings, chunks)

    @staticmethod
    def _load_evaluation_cases() -> Dict[str, List[str]]:
        dataset_path = Path(__file__).resolve().parent.parent / "evaluation_dataset.json"
        try:
            with dataset_path.open(encoding="utf-8") as dataset_file:
                cases = json.load(dataset_file)
        except (OSError, json.JSONDecodeError):
            return {}

        return {
            " ".join(case.get("question", "").casefold().split()): [
                term.casefold()
                for term in case.get("relevance_terms", [])
                if term
            ]
            for case in cases
            if case.get("question")
        }

    def _retrieve(
        self,
        model: SentenceTransformer,
        store: FAISSVectorStore,
        query: str,
        top_k: int,
    ) -> Dict[str, Any]:
        total_start = time.perf_counter()

        embedding_start = time.perf_counter()
        query_embedding = model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype("float32")
        embedding_latency_ms = (time.perf_counter() - embedding_start) * 1000

        retrieval_start = time.perf_counter()
        results = store.search(query_embedding, top_k)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        return {
            "results": results,
            "embedding_latency_ms": round(embedding_latency_ms, 3),
            "retrieval_latency_ms": round(retrieval_latency_ms, 3),
            "total_latency_ms": round((time.perf_counter() - total_start) * 1000, 3),
        }

    def retrieve_qwen(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        return self._retrieve(self.qwen_model, self.qwen_store, query, top_k)

    def retrieve_BAAI(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        return self._retrieve(self.bge_model, self.bge_store, query, top_k)

    @staticmethod
    def _result_ids(results: List[Dict[str, Any]]) -> List[str]:
        return [item["chunk_id"] for item in results if item.get("chunk_id")]

    @staticmethod
    def _score(results: List[Dict[str, Any]]) -> float:
        return float(results[0].get("score", 0.0)) if results else 0.0

    @staticmethod
    def _average_score(results: List[Dict[str, Any]]) -> float:
        if not results:
            return 0.0
        return sum(float(item.get("score", 0.0)) for item in results) / len(results)

    @staticmethod
    def _top_source(results: List[Dict[str, Any]]) -> Optional[str]:
        return results[0].get("source") if results else None

    @staticmethod
    def _winner(first: float, second: float) -> str:
        if first > second:
            return "QWEN"
        if second > first:
            return "BAAI"
        return "Tie"

    @staticmethod
    def _evaluation_metrics(
        results: List[Dict[str, Any]],
        relevant_terms: List[str],
        total_relevant_chunks: int,
    ) -> Dict[str, Optional[float]]:
        if not relevant_terms or total_relevant_chunks == 0:
            return {
                "precision": None,
                "recall": None,
                "f1": None,
                "mrr": None,
            }

        relevant_results = [
            any(
                EmbeddingComparisonRetriever._contains_relevance_term(
                    item.get("content", ""),
                    term,
                )
                for term in relevant_terms
            )
            for item in results
        ]
        relevant_retrieved = sum(relevant_results)
        precision = relevant_retrieved / len(results) if results else 0.0
        recall = relevant_retrieved / total_relevant_chunks
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        first_relevant_rank = next(
            (rank for rank, is_relevant in enumerate(relevant_results, start=1) if is_relevant),
            None,
        )

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mrr": round(1 / first_relevant_rank, 4) if first_relevant_rank else 0.0,
        }

    @staticmethod
    def _contains_relevance_term(content: str, term: str) -> bool:
        normalized_content = " ".join(content.casefold().split())
        normalized_term = " ".join(term.casefold().split())
        if normalized_term in normalized_content:
            return True

        term_tokens = {
            token.strip(".,:;!?()[]")
            for token in normalized_term.split()
            if len(token.strip(".,:;!?()[]")) > 2
        }
        content_tokens = {
            token.strip(".,:;!?()[]")
            for token in normalized_content.split()
        }
        return bool(term_tokens) and term_tokens.issubset(content_tokens)

    def _evaluation_terms_for_query(self, query: str) -> List[str]:
        normalized_query = " ".join(query.casefold().split())
        exact_terms = self.evaluation_cases.get(normalized_query)
        if exact_terms is not None:
            return exact_terms

        query_tokens = set(normalized_query.split())
        best_score = 0.0
        best_terms: List[str] = []
        for evaluation_question, terms in self.evaluation_cases.items():
            question_tokens = set(evaluation_question.split())
            shared_tokens = len(query_tokens & question_tokens)
            token_score = shared_tokens / max(len(query_tokens), len(question_tokens))
            text_score = SequenceMatcher(
                None,
                normalized_query,
                evaluation_question,
            ).ratio()
            score = max(token_score, text_score)
            if score > best_score:
                best_score = score
                best_terms = terms

        return best_terms if best_score >= 0.45 else []

    def compare(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        query = query.strip()
        QWEN = self.retrieve_qwen(query, top_k)
        BAAI = self.retrieve_BAAI(query, top_k)
        qwen_results = QWEN["results"]
        BAAI_results = BAAI["results"]
        qwen_ids = self._result_ids(qwen_results)
        BAAI_ids = self._result_ids(BAAI_results)
        common_results = len(set(qwen_ids) & set(BAAI_ids))
        union_results = len(set(qwen_ids) | set(BAAI_ids))
        evaluation_terms = self._evaluation_terms_for_query(query)
        total_relevant_chunks = sum(
            any(
                self._contains_relevance_term(
                    chunk.get("content", ""),
                    term,
                )
                for term in evaluation_terms
            )
            for chunk in self.chunks
        )
        qwen_evaluation = self._evaluation_metrics(
            qwen_results,
            evaluation_terms,
            total_relevant_chunks,
        )
        BAAI_evaluation = self._evaluation_metrics(
            BAAI_results,
            evaluation_terms,
            total_relevant_chunks,
        )

        return {
            "query": query,
            "qwen": {
                "model": self.QWEN_MODEL,
                "top_score": round(self._score(qwen_results), 4),
                "average_score": round(self._average_score(qwen_results), 4),
                "top_source": self._top_source(qwen_results),
                "embedding_latency_ms": QWEN["embedding_latency_ms"],
                "retrieval_latency_ms": QWEN["retrieval_latency_ms"],
                "total_latency_ms": QWEN["total_latency_ms"],
                **qwen_evaluation,
                "results": qwen_results,
            },
            "BAAI": {
                "model": self.BAAI_MODEL,
                "top_score": round(self._score(BAAI_results), 4),
                "average_score": round(self._average_score(BAAI_results), 4),
                "top_source": self._top_source(BAAI_results),
                "embedding_latency_ms": BAAI["embedding_latency_ms"],
                "retrieval_latency_ms": BAAI["retrieval_latency_ms"],
                "total_latency_ms": BAAI["total_latency_ms"],
                **BAAI_evaluation,
                "results": BAAI_results,
            },
            "comparison": {
                "similarity_winner": self._winner(
                    self._score(qwen_results), self._score(BAAI_results)
                ),
                "speed_winner": self._winner(
                    -QWEN["total_latency_ms"], -BAAI["total_latency_ms"]
                ),
                "top_result_agreement": (
                    qwen_ids[0] if qwen_ids else None
                ) == (BAAI_ids[0] if BAAI_ids else None),
                "common_results": common_results,
                "result_overlap_percentage": round(
                    common_results / union_results * 100 if union_results else 0.0,
                    2,
                ),
                "evaluation_available": bool(evaluation_terms),
            },
        }
'''