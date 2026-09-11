"""Standalone single-model RAG evaluator."""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from dotenv import load_dotenv
from groq import Groq

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_GROQ_MODEL = "qwen/qwen3.6-27b"
DIMENSION = 384
NOT_AVAILABLE = "not_available"
STRICT_PROMPT = """You are a strict policy-document question-answering assistant.

Answer only from the supplied policy context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts, dates, benefits, leave balances, or durations.
3. If the answer is not present in the context, clearly say:
   "The provided policy context does not specify this information."
4. Distinguish fictional/demo policy values from real legal or HR entitlements.
5. Answer clearly and directly.
6. Do not claim that information exists unless it is supported by the context.
7. Use the retrieved policy context as the only source of truth.

POLICY CONTEXT:
{context}

USER QUESTION:
{question}
"""
JUDGE_PROMPT = """Evaluate whether the answer is supported by the exact policy context.
Return JSON only: {\"grounded\": true, \"score\": 0.0, \"unsupported_claims\": [], \"explanation\": \"\"}
The score must be a number from 0 to 1. Treat fictional/demo values as supported only
when they appear in context, and flag outside legal or HR claims.

POLICY CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}
"""


def _json_cell(value: Any) -> str:
    return value if value == NOT_AVAILABLE else json.dumps(value, ensure_ascii=True, default=str)


def _terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item) for item in value if str(item).strip()]


def _contains(text: str, term: str) -> bool:
    return term.casefold().strip() in text.casefold()


def _word_overlap(left: str, right: str) -> float:
    left_words = set(re.findall(r"[a-z0-9']+", left.casefold()))
    right_words = set(re.findall(r"[a-z0-9']+", right.casefold()))
    return len(left_words & right_words) / len(right_words) if right_words else 0.0


@dataclass
class PolicyChunk:
    chunk_id: int
    text: str
    source: str


class SingleRAGRetriever:
    """Independent normalized cosine-similarity retriever."""

    def __init__(self, model: Any, policy_path: Path, chunk_size: int = 900, chunk_overlap: int = 120):
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and overlap must be smaller than size")
        self.model = model
        self.policy_path = policy_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = self._chunk(policy_path.read_text(encoding="utf-8"))
        self.embeddings = self._encode([chunk.text for chunk in self.chunks])

    def _chunk(self, text: str) -> list[PolicyChunk]:
        words = text.split()
        chunks: list[PolicyChunk] = []
        start = 0
        chunk_id = 0
        while start < len(words):
            end = min(len(words), start + self.chunk_size)
            chunks.append(PolicyChunk(chunk_id, " ".join(words[start:end]), self.policy_path.name))
            if end == len(words):
                break
            start = end - self.chunk_overlap
            chunk_id += 1
        return chunks

    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        vectors = self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != DIMENSION:
            raise ValueError(f"Expected {DIMENSION}-dimensional embeddings, got {vectors.shape[1]}")
        return vectors

    def retrieve(self, question: str, top_k: int) -> tuple[list[dict[str, Any]], float, float]:
        started = time.perf_counter()
        query_embedding = self._encode([question])[0]
        embedding_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        scores = self.embeddings @ query_embedding
        indexes = np.argsort(scores)[::-1][:max(0, top_k)]
        results = [{"chunk_id": int(index), "text": self.chunks[index].text, "score": round(float(scores[index]), 6), "source": self.chunks[index].source} for index in indexes]
        return results, embedding_ms, (time.perf_counter() - started) * 1000


class SingleRAGEvaluator:
    def __init__(self, root: str | Path | None = None, top_k: int = 3, chunk_size: int = 900, chunk_overlap: int = 120, temperature: float = 0.0, max_output_tokens: int = 512, enable_grounding_judge: bool = True):
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.top_k = top_k
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.enable_grounding_judge = enable_grounding_judge
        load_dotenv(self.root / ".env")
        self.model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self.dataset_path = self.root / "evaluation_dataset.json"
        self.policy_path = self.root / "knowledge_base" / "leave_policy.txt"
        if not self.dataset_path.is_file():
            raise FileNotFoundError(self.dataset_path)
        if not self.policy_path.is_file():
            raise FileNotFoundError(self.policy_path)
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(MODEL_NAME, device="cpu")
        self.retriever = SingleRAGRetriever(self.embedding_model, self.policy_path, chunk_size, chunk_overlap)
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key) if api_key else None
        LOGGER.info("Loaded %s chunks with %s on CPU", len(self.retriever.chunks), MODEL_NAME)

    def _context(self, retrieved: list[dict[str, Any]]) -> str:
        return "\n\n".join(f"[{item['source']} | chunk {item['chunk_id']}]\n{item['text']}" for item in retrieved)

    @staticmethod
    def _usage(response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"input_tokens": NOT_AVAILABLE, "output_tokens": NOT_AVAILABLE, "total_tokens": NOT_AVAILABLE}
        return {"input_tokens": getattr(usage, "prompt_tokens", NOT_AVAILABLE), "output_tokens": getattr(usage, "completion_tokens", NOT_AVAILABLE), "total_tokens": getattr(usage, "total_tokens", NOT_AVAILABLE)}

    def _chat(self, prompt: str, temperature: float | None = None, max_tokens: int | None = None) -> tuple[str, dict[str, Any], float]:
        if not self.client:
            raise RuntimeError("GROQ_API_KEY is missing")
        started = time.perf_counter()
        response = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": prompt}], temperature=self.temperature if temperature is None else temperature, max_tokens=self.max_output_tokens if max_tokens is None else max_tokens, timeout=60)
        content = response.choices[0].message.content if response.choices else ""
        return (content or "").strip(), self._usage(response), (time.perf_counter() - started) * 1000

    def _judge(self, question: str, context: str, answer: str) -> dict[str, Any]:
        if not self.enable_grounding_judge:
            return {"grounding_score": NOT_AVAILABLE, "grounding_status": "disabled", "unsupported_claims": [], "latency": 0.0}
        try:
            raw, _, latency = self._chat(JUDGE_PROMPT.format(context=context, question=question, answer=answer), temperature=0.0, max_tokens=400)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("grounding judge returned no JSON")
            parsed = json.loads(match.group(0))
            score = float(parsed["score"])
            if not 0 <= score <= 1:
                raise ValueError("grounding score outside 0..1")
            claims = parsed.get("unsupported_claims", [])
            return {"grounding_score": score, "grounding_status": "judge_success", "unsupported_claims": claims if isinstance(claims, list) else [str(claims)], "latency": latency}
        except Exception as exc:
            LOGGER.warning("Grounding judge failed: %s", exc)
            return {"grounding_score": None, "grounding_status": "judge_fallback", "unsupported_claims": [], "latency": 0.0, "judge_error": str(exc)}

    def evaluate_question(self, question: str, relevance_terms: Any = None, expected_answer: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        terms = _terms(relevance_terms)
        row: dict[str, Any] = {"question": question, "relevance_terms": terms, "model": self.model_name, "expected_answer": expected_answer or NOT_AVAILABLE}
        try:
            retrieved, embedding_ms, retrieval_ms = self.retriever.retrieve(question, self.top_k)
            context = self._context(retrieved)
            matched = [term for term in terms if any(_contains(item["text"], term) for item in retrieved)]
            relevant_chunk_count = sum(
                any(_contains(item["text"], term) for term in terms)
                for item in retrieved
            )
            answer, usage, generation_ms = self._chat(STRICT_PROMPT.format(context=context, question=question))
            answer_matched = [term for term in terms if _contains(answer, term)]
            judge = self._judge(question, context, answer)
            term_count = len(terms)
            retrieved_count = len(retrieved)
            precision_at_k = relevant_chunk_count / retrieved_count if retrieved_count and term_count else NOT_AVAILABLE
            recall_at_k = len(matched) / term_count if term_count else NOT_AVAILABLE
            f1_score = (
                2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
                if isinstance(precision_at_k, float) and isinstance(recall_at_k, float) and precision_at_k + recall_at_k
                else (0.0 if isinstance(precision_at_k, float) and isinstance(recall_at_k, float) else NOT_AVAILABLE)
            )
            correctness = _word_overlap(answer, expected_answer) if expected_answer else NOT_AVAILABLE
            top_score = retrieved[0]["score"] if retrieved else NOT_AVAILABLE
            average_score = sum(item["score"] for item in retrieved) / retrieved_count if retrieved else NOT_AVAILABLE
            row.update({"retrieved_chunks": retrieved, "retrieval_scores": [item["score"] for item in retrieved], "retrieval_hit": bool(matched), "accuracy_at_k": 1.0 if matched else (0.0 if terms else NOT_AVAILABLE), "precision_at_k": precision_at_k, "recall_at_k": recall_at_k, "f1_score": f1_score, "precision_like_score": len(matched) / term_count if term_count else NOT_AVAILABLE, "mrr": next((1 / (index + 1) for index, item in enumerate(retrieved) if any(_contains(item["text"], term) for term in terms)), NOT_AVAILABLE), "context": context, "answer": answer, "answer_term_coverage": len(answer_matched) / term_count if term_count else NOT_AVAILABLE, "answer_correctness": correctness, "grounding_score": judge["grounding_score"], "grounding_status": judge["grounding_status"], "unsupported_claims": judge["unsupported_claims"], "hallucination_risk": 1 - judge["grounding_score"] if isinstance(judge["grounding_score"], (int, float)) else NOT_AVAILABLE, "answer_length_characters": len(answer), "answer_length_words": len(answer.split()), "embedding_latency_ms": embedding_ms, "retrieval_latency_ms": retrieval_ms, "latency_ms": retrieval_ms + generation_ms + judge["latency"], "generation_latency_ms": generation_ms, "grounding_judge_latency_ms": judge["latency"], **usage, "retrieved_chunk_count": retrieved_count, "context_characters": len(context), "context_words": len(context.split()), "similarity_score": top_score, "average_similarity_score": average_score, "top_retrieval_score": top_score, "average_retrieval_score": average_score, "error": ""})
        except Exception as exc:
            LOGGER.exception("Question evaluation failed")
            row.update({"retrieved_chunks": [], "retrieval_scores": [], "retrieval_hit": False, "accuracy_at_k": NOT_AVAILABLE, "precision_at_k": NOT_AVAILABLE, "recall_at_k": NOT_AVAILABLE, "f1_score": NOT_AVAILABLE, "precision_like_score": NOT_AVAILABLE, "mrr": NOT_AVAILABLE, "context": "", "answer": "", "answer_term_coverage": NOT_AVAILABLE, "answer_correctness": NOT_AVAILABLE, "grounding_score": None, "grounding_status": "judge_error", "unsupported_claims": [], "hallucination_risk": NOT_AVAILABLE, "answer_length_characters": 0, "answer_length_words": 0, "embedding_latency_ms": NOT_AVAILABLE, "retrieval_latency_ms": NOT_AVAILABLE, "latency_ms": NOT_AVAILABLE, "generation_latency_ms": NOT_AVAILABLE, "grounding_judge_latency_ms": NOT_AVAILABLE, "input_tokens": NOT_AVAILABLE, "output_tokens": NOT_AVAILABLE, "total_tokens": NOT_AVAILABLE, "retrieved_chunk_count": 0, "context_characters": 0, "context_words": 0, "similarity_score": NOT_AVAILABLE, "average_similarity_score": NOT_AVAILABLE, "top_retrieval_score": NOT_AVAILABLE, "average_retrieval_score": NOT_AVAILABLE, "error": str(exc)})
        row["total_latency_ms"] = (time.perf_counter() - started) * 1000
        return row

    def load_dataset(self) -> list[dict[str, Any]]:
        data = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("evaluation dataset must be a JSON array")
        return data

    def evaluate_dataset(self) -> list[dict[str, Any]]:
        rows = []
        for index, item in enumerate(self.load_dataset(), 1):
            LOGGER.info("Evaluating question %d", index)
            if not isinstance(item, dict) or not str(item.get("question", "")).strip():
                row = self.evaluate_question("", [], item.get("expected_answer") if isinstance(item, dict) else None)
                row["error"] = "dataset item missing question"
            else:
                row = self.evaluate_question(item["question"], item.get("relevance_terms", []), item.get("expected_answer"))
            rows.append(row)
        return rows

    @staticmethod
    def write_csv(rows: list[dict[str, Any]], output_dir: str | Path) -> tuple[Path, Path]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        fields = ["question", "relevance_terms", "retrieved_chunks", "retrieval_scores", "retrieval_hit", "accuracy_at_k", "precision_at_k", "recall_at_k", "f1_score", "precision_like_score", "mrr", "similarity_score", "average_similarity_score", "context", "model", "answer", "answer_term_coverage", "expected_answer", "answer_correctness", "grounding_score", "grounding_status", "unsupported_claims", "hallucination_risk", "embedding_latency_ms", "retrieval_latency_ms", "latency_ms", "generation_latency_ms", "grounding_judge_latency_ms", "total_latency_ms", "input_tokens", "output_tokens", "total_tokens", "retrieved_chunk_count", "context_characters", "context_words", "top_retrieval_score", "average_retrieval_score", "error"]
        timestamped = directory / f"single_rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        latest = directory / "single_rag_evaluation_latest.csv"
        for path in (timestamped, latest):
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows({field: _json_cell(row.get(field, NOT_AVAILABLE)) for field in fields} for row in rows)
        return latest, timestamped


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluator = SingleRAGEvaluator()
    print(SingleRAGEvaluator.write_csv(evaluator.evaluate_dataset(), evaluator.root / "evaluation_results"))