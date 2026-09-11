"""Standalone BGE-to-LLM comparison evaluation logic."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

SAFE_RESPONSE = "I don't have enough information in the provided policy context to answer that."
BGE_MODEL = "BAAI/bge-small-en-v1.5"
USD_TO_INR = float(os.getenv("USD_TO_INR", "83.5"))

# Pricing configuration (USD per 1M tokens)
GROQ_PRICING = {
    "openai/gpt-oss-20b": {
        "input": float(os.getenv("GROQ_INPUT_COST_PER_MILLION", "0.075")),
        "output": float(os.getenv("GROQ_OUTPUT_COST_PER_MILLION", "0.30")),
    }
}

QWEN_PRICING = {
    "input": float(os.getenv("QWEN_INPUT_COST_PER_MILLION", "0.29")),
    "output": float(os.getenv("QWEN_OUTPUT_COST_PER_MILLION", "0.59")),
}


class DualLLMEvaluator:
    """Evaluate two LLMs (Groq and Qwen) against context returned by a BGE retriever."""

    SYSTEM_PROMPT = """You are a leave-policy assistant for a fictional demonstration/test policy.

Answer from the entire POLICY CONTEXT supplied in the user message. Answer
every part of a multi-topic question when the context supports it.

Preserve exact rules, values, conditions, limitations, and terminology.
Do not refuse an answer just because one retrieved chunk is incomplete.

The document may contain an earlier general rule and a later expanded
demonstration rule that appear to conflict. Do not silently remove either
one. Explain both and identify the later value as DEMONSTRATION / TEST POLICY
DATA, not as a real legal entitlement.

Do not invent facts or legal/company rules not present in the context. If a
fact is absent, say so and direct the user to HR where appropriate.
"""

    def __init__(self, retriever: Any, groq_model: str = "openai/gpt-oss-20b", qwen_model: str = "qwen/qwen3.6-27b") -> None:
        self.retriever = retriever
        self.groq_model = groq_model
        self.qwen_model = qwen_model
        self._groq_client = None

    def _get_groq_client(self):
        if self._groq_client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY was not found in the environment.")
            try:
                from groq import Groq
            except ImportError as exc:
                raise RuntimeError("The Groq Python package is not installed.") from exc
            self._groq_client = Groq(api_key=api_key)
        return self._groq_client

    @staticmethod
    def _normalize_results(results: Any) -> List[Dict[str, Any]]:
        if isinstance(results, dict):
            results = results.get("results", results.get("chunks", [results]))
        if results is None:
            return []
        if not isinstance(results, (list, tuple)):
            results = [results]

        normalized = []
        for index, item in enumerate(results):
            if isinstance(item, str):
                text, chunk_id, score, metadata = item, index, None, {}
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("page_content") or ""
                chunk_id = item.get("chunk_id", item.get("id", index))
                score = item.get("score", item.get("similarity", item.get("similarity_score")))
                metadata = item.get("metadata") or {}
            else:
                text = getattr(item, "text", getattr(item, "page_content", str(item)))
                chunk_id = getattr(item, "chunk_id", getattr(item, "id", index))
                score = getattr(item, "score", getattr(item, "similarity", None))
                metadata = getattr(item, "metadata", {}) or {}
            normalized.append({
                "rank": index + 1,
                "chunk_id": str(chunk_id),
                "text": str(text).strip(),
                "score": float(score) if score is not None else None,
                "source": str(metadata.get("source", metadata.get("file", "leave_policy.txt"))),
                "metadata": metadata,
            })
        return [item for item in normalized if item["text"]]

    def retrieve_bge(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        method = getattr(self.retriever, "retrieve_bge", None)
        if not callable(method):
            method = getattr(self.retriever, "retrieve", None)
        if not callable(method):
            raise TypeError("The supplied retriever must expose retrieve_bge(query, top_k).")
        try:
            results = method(query, top_k=top_k)
        except TypeError:
            results = method(query, top_k)
        return self._normalize_results(results)

    @staticmethod
    def build_context(chunks: List[Dict[str, Any]], context_window_tokens: int) -> Tuple[str, List[Dict[str, Any]]]:
        max_chars = max(1000, int(context_window_tokens) * 4)
        parts, used, current = [], [], 0
        for chunk in chunks:
            block = f"\n--- POLICY CHUNK {chunk['chunk_id']} ---\n{chunk['text']}\n"
            if current + len(block) > max_chars:
                remaining = max_chars - current
                if remaining > 300:
                    parts.append(block[:remaining])
                    used.append(chunk)
                break
            parts.append(block)
            used.append(chunk)
            current += len(block)
        return "".join(parts), used

    @staticmethod
    def _usage(response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        total = int(getattr(usage, "total_tokens", prompt + completion) or 0) if usage else 0
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}

    def generate_groq(self, query: str, context: str, temperature: float, max_output_tokens: int, max_input_tokens: int, model_name: Optional[str] = None) -> Dict[str, Any]:
        if not context.strip():
            return {"answer": SAFE_RESPONSE, "latency_ms": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        prompt = f"POLICY CONTEXT:\n{context}\n\nUSER QUESTION:\n{query}\n\nAnswer using only the policy context."
        started = time.perf_counter()
        response = self._get_groq_client().chat.completions.create(
            model=model_name or self.groq_model,
            messages=[{"role": "system", "content": self.SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            temperature=float(temperature),
            max_tokens=int(max_output_tokens),
        )
        answer = (response.choices[0].message.content or "").strip() if response.choices else SAFE_RESPONSE
        return {"answer": answer, "latency_ms": round((time.perf_counter() - started) * 1000, 2), **self._usage(response)}

    def generate_qwen(self, query: str, context: str, temperature: float, max_output_tokens: int, max_input_tokens: int) -> Dict[str, Any]:
        return self.generate_groq(
            query,
            context,
            temperature,
            max_output_tokens,
            max_input_tokens,
            model_name=self.qwen_model,
        )

    @staticmethod
    def _tokens(text: str) -> set:
        words = re.findall(r"[a-z0-9]+", text.lower())
        stop_words = {"this", "that", "with", "from", "have", "will", "would", "could", "should", "what", "your", "only", "provided", "policy", "context"}
        return {word for word in words if len(word) >= 4 and word not in stop_words}

    def lexical_grounding(self, answer: str, context: str) -> Dict[str, Any]:
        answer_tokens, context_tokens = self._tokens(answer), self._tokens(context)
        supported = answer_tokens & context_tokens
        unsupported = sorted(answer_tokens - context_tokens)
        score = len(supported) / len(answer_tokens) if answer_tokens else 0.0
        return {"grounding_score": round(score, 4), "supported": score >= 0.45, "unsupported_tokens": unsupported}

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE).strip()
        try:
            value = json.loads(cleaned)
            return value if isinstance(value, dict) else {}
        except (json.JSONDecodeError, TypeError):
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    value = json.loads(match.group(0))
                    return value if isinstance(value, dict) else {}
                except json.JSONDecodeError:
                    pass
        return {}

    def grounding_judge(self, query: str, context: str, answer: str, model_name: str = "groq") -> Dict[str, Any]:
        prompt = f"""Judge whether every factual claim in the answer is supported by the policy context.
QUESTION: {query}
POLICY CONTEXT: {context}
ANSWER: {answer}
Return JSON only with grounded (boolean), score (0 to 1), factual_claims (array),
supported_claims (array), unsupported_claims (array), contradictions (array),
and explanation (string)."""
        try:
            request = {
                "model": self.groq_model,
                "messages": [{"role": "system", "content": "You are a strict grounding judge. Return JSON only."}, {"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 1000,
                "response_format": {"type": "json_object"},
            }
            try:
                response = self._get_groq_client().chat.completions.create(**request)
            except Exception:
                request.pop("response_format")
                response = self._get_groq_client().chat.completions.create(**request)
            raw = response.choices[0].message.content or "" if response.choices else ""
            parsed = self._parse_json(raw)
            raw_score = parsed.get("score")
            score = max(0.0, min(1.0, float(raw_score))) if raw_score is not None else None
            factual_claims = parsed.get("factual_claims", [])
            supported_claims = parsed.get("supported_claims", [])
            unsupported = parsed.get("unsupported_claims", [])
            contradictions = parsed.get("contradictions", [])
            factual_claims = factual_claims if isinstance(factual_claims, list) else []
            supported_claims = supported_claims if isinstance(supported_claims, list) else []
            unsupported = unsupported if isinstance(unsupported, list) else [str(unsupported)]
            contradictions = contradictions if isinstance(contradictions, list) else [str(contradictions)]
            true_positive = len(supported_claims)
            false_positive = len(unsupported) + len(contradictions)
            false_negative = max(0, len(factual_claims) - true_positive - false_positive)
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else score
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else score
            f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
            grounded_value = parsed.get("grounded")
            grounded = bool(grounded_value) if grounded_value is not None else score == 1.0
            return {"grounded": grounded, "score": score, "hallucination_score": round(1.0 - score, 4) if score is not None else None, "precision": round(precision, 4) if precision is not None else None, "recall": round(recall, 4) if recall is not None else None, "f1": round(f1, 4) if precision is not None and recall is not None else None, "factual_claims": factual_claims, "supported_claims": supported_claims, "unsupported_claims": unsupported, "contradictions": contradictions, "reason": str(parsed.get("explanation", parsed.get("reason", "")))}
        except Exception as exc:
            return {"grounded": None, "score": None, "hallucination_score": None, "precision": None, "recall": None, "f1": None, "factual_claims": [], "supported_claims": [], "unsupported_claims": [], "contradictions": [], "reason": f"The grounding response could not be parsed or completed: {exc}"}

    def hallucination_guard(self, query: str, context: str, draft_answer: str) -> Dict[str, Any]:
        prompt = f"""Rewrite the draft using only facts explicitly present in the policy context.
QUESTION: {query}
POLICY CONTEXT: {context}
DRAFT: {draft_answer}
Return JSON only: {{"hallucination": false, "final_answer": "", "removed_claims": []}}.
Use this exact answer if the context is insufficient: {SAFE_RESPONSE}"""
        try:
            response = self._get_groq_client().chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "system", "content": "You are a strict policy safety filter. Return JSON only."}, {"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=700,
            )
            parsed = self._parse_json(response.choices[0].message.content or "") if response.choices else {}
            final_answer = str(parsed.get("final_answer", "")).strip()
            removed = parsed.get("removed_claims", [])
            return {"hallucination": bool(parsed.get("hallucination", True)), "final_answer": final_answer or SAFE_RESPONSE, "removed_claims": removed if isinstance(removed, list) else [str(removed)]}
        except Exception as exc:
            return {"hallucination": True, "final_answer": SAFE_RESPONSE, "removed_claims": [f"Guard unavailable: {exc}"]}

    @staticmethod
    def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        """Calculate cost based on actual token usage and pricing configuration."""
        if input_tokens is None or output_tokens is None:
            return {
                "input_cost": None,
                "output_cost": None,
                "total_cost": None,
                "input_cost_inr": None,
                "output_cost_inr": None,
                "total_cost_inr": None,
                "usd_to_inr": USD_TO_INR,
                "pricing_configured": False,
                "pricing_source": "unavailable",
            }
        
        if model_name.startswith("openai/"):
            pricing = GROQ_PRICING.get(model_name, {"input": 0.0, "output": 0.0})
        else:
            pricing = QWEN_PRICING
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "input_cost_inr": round(input_cost * USD_TO_INR, 4),
            "output_cost_inr": round(output_cost * USD_TO_INR, 4),
            "total_cost_inr": round(total_cost * USD_TO_INR, 4),
            "usd_to_inr": USD_TO_INR,
            "pricing_configured": pricing["input"] > 0 or pricing["output"] > 0,
            "pricing_source": "groq_api"
        }

    def evaluate_single_model(self, query: str, context: str, used_chunks: List[Dict[str, Any]], model_name: str, temperature: float, max_output_tokens: int, max_input_tokens: int, use_grounding_judge: bool, use_hallucination_guard: bool) -> Dict[str, Any]:
        """Evaluate a single model with the given context."""
        if model_name.startswith("openai/"):
            generation = self.generate_groq(query, context, temperature, max_output_tokens, max_input_tokens)
        else:
            generation = self.generate_qwen(query, context, temperature, max_output_tokens, max_input_tokens)
        
        lexical = self.lexical_grounding(generation["answer"], context)
        judge = self.grounding_judge(query, context, generation["answer"], model_name) if use_grounding_judge else None
        
        judge_fallback_used = False
        if use_grounding_judge and (judge is None or judge.get("score") is None):
            fallback_score = lexical["grounding_score"]
            judge = {
                "grounded": lexical["supported"],
                "score": fallback_score,
                "hallucination_score": round(1.0 - fallback_score, 4),
                "precision": fallback_score,
                "recall": fallback_score,
                "f1": fallback_score,
                "factual_claims": [],
                "supported_claims": [],
                "unsupported_claims": [],
                "contradictions": [],
                "reason": "Grounding judge unavailable; lexical grounding fallback used.",
            }
            judge_fallback_used = True
        
        unsupported = judge.get("unsupported_claims", []) if judge else []
        contradictions = judge.get("contradictions", []) if judge else []
        hallucination = (not judge.get("grounded", False) or bool(unsupported) or bool(contradictions)) if judge else not lexical["supported"]
        
        guard = self.hallucination_guard(query, context, generation["answer"]) if hallucination and use_hallucination_guard else None
        final_answer = guard["final_answer"] if guard else generation["answer"]
        
        cost = self.calculate_cost(model_name, generation.get("prompt_tokens"), generation.get("completion_tokens"))
        
        evaluation_metrics = {
            "hallucination_score": judge.get("hallucination_score") if judge else None,
            "precision": judge.get("precision") if judge else None,
            "recall": judge.get("recall") if judge else None,
            "f1": judge.get("f1") if judge else None,
        }
        
        evaluation_status = "success"
        if generation.get("error"):
            evaluation_status = "generation_failed"
        elif judge_fallback_used:
            evaluation_status = "grounding_evaluation_fallback"
        elif not use_grounding_judge:
            evaluation_status = "grounding_judge_disabled"
        
        return {
            "model": model_name,
            "draft_answer": generation["answer"],
            "final_answer": final_answer,
            "generation_latency_ms": generation["latency_ms"],
            "prompt_tokens": generation.get("prompt_tokens"),
            "completion_tokens": generation.get("completion_tokens"),
            "total_tokens": generation.get("total_tokens"),
            "lexical_grounding": lexical,
            "grounding_judge": judge,
            "hallucination": hallucination,
            "hallucination_score": evaluation_metrics["hallucination_score"],
            "precision": evaluation_metrics["precision"],
            "recall": evaluation_metrics["recall"],
            "f1": evaluation_metrics["f1"],
            "unsupported_claims": unsupported,
            "contradictions": contradictions,
            "hallucination_guard": guard,
            "evaluation_status": evaluation_status,
            "cost": cost,
            "error": generation.get("error")
        }

    def evaluate_dual_models(self, query: str, top_k: int = 5, temperature: float = 0.2, context_window_tokens: int = 6000, max_output_tokens: int = 700, use_grounding_judge: bool = True, use_hallucination_guard: bool = True, max_input_tokens: int = 4000) -> Dict[str, Any]:
        """Evaluate both Groq and Qwen models with the SAME BGE-retrieved context."""
        started = time.perf_counter()
        
        # BGE retrieval happens once per question and decomposes multi-intent queries.
        retrieval_started = time.perf_counter()
        retrieved = self.retrieve_bge(query, top_k)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        
        # Build context ONCE
        context, used = self.build_context(
            retrieved,
            int(context_window_tokens),
        )
        
        # Evaluate both models with the SAME context
        groq_result = self.evaluate_single_model(
            query, context, used, self.groq_model, temperature, max_output_tokens, max_input_tokens, use_grounding_judge, use_hallucination_guard
        )
        
        qwen_result = self.evaluate_single_model(
            query, context, used, self.qwen_model, temperature, max_output_tokens, max_input_tokens, use_grounding_judge, use_hallucination_guard
        )
        
        total_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        
        return {
            "query": query,
            "embedding_model": BGE_MODEL,
            "temperature": temperature,
            "top_k": top_k,
            "context_window_tokens": context_window_tokens,
            "max_input_tokens": max_input_tokens,
            "max_output_tokens": max_output_tokens,
            "retrieved_chunks": retrieved,
            "used_chunks": used,
            "context": context,
            "retrieval_latency_ms": round(retrieval_ms, 2),
            "total_latency_ms": total_latency_ms,
            "context_chars": len(context),
            "context_chunks": len(used),
            "context_tokens": None,
            "model_a": groq_result,
            "model_b": qwen_result,
            "retrieval_metrics": {"ground_truth_available": False}
        }
