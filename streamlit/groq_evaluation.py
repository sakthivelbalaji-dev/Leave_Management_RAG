"""Streamlit page for LLM comparison evaluation (Groq vs Qwen) with standalone BGE retrieval."""

import os
import sys
import importlib.util
import csv
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BGE_MODEL = "BAAI/bge-small-en-v1.5"
# Keep the comparison roles separate: GROQ_MODEL may be left over from an
# older configuration and must not silently turn Model A into the Qwen model.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen/qwen3.6-27b")
KNOWLEDGE_BASE_PATH = os.path.join(PROJECT_ROOT, "knowledge_base")
EVALUATION_RESULTS_PATH = os.path.join(PROJECT_ROOT, "evaluation_results")
EVALUATION_CSV_PATH = os.getenv(
    "EVALUATION_CSV_PATH",
    os.path.join(EVALUATION_RESULTS_PATH, "llm_comparison_results.csv"),
)
CSV_FIELDS = [
    "timestamp", "evaluation_id", "question", "model", "embedding_model", "temperature", "top_k",
    "context_window_tokens", "max_input_tokens", "max_output_tokens", "retrieved_chunks", "used_context_chunks",
    "context_tokens", "input_tokens", "output_tokens", "total_tokens", "retrieval_latency_ms",
    "generation_latency_ms", "total_latency_ms", "quality_score", "grounding_score", "hallucination_score",
    "hallucination", "factual_claims_count", "supported_claims_count", "unsupported_claims_count",
    "contradiction_count", "unsupported_claims", "contradictions", "generated_answer", "evaluation_status",
    "input_cost", "output_cost", "total_cost", "input_cost_inr", "output_cost_inr", "total_cost_inr",
    "usd_to_inr", "pricing_source",
]

EVALUATOR_PATH = os.path.join(PROJECT_ROOT, "rag", "groq_evaluator.py")
evaluator_spec = importlib.util.spec_from_file_location("dual_llm_evaluator", EVALUATOR_PATH)
if evaluator_spec is None or evaluator_spec.loader is None:
    raise ImportError(f"Could not load dual LLM evaluator from {EVALUATOR_PATH}")
evaluator_module = importlib.util.module_from_spec(evaluator_spec)
evaluator_spec.loader.exec_module(evaluator_module)
DualLLMEvaluator = evaluator_module.DualLLMEvaluator

st.set_page_config(page_title="LLM RAG Comparison Evaluation", page_icon="⚖️", layout="wide")


class BGERetriever:
    """Load the policy and retrieve chunks using normalized BGE vectors."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 140):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = self._load_chunks()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required to load BGE.") from exc
        self.model = SentenceTransformer(BGE_MODEL, device="cpu")
        self.embeddings = np.asarray(self.model.encode([chunk["text"] for chunk in self.chunks], normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    def _load_chunks(self):
        if not os.path.isdir(KNOWLEDGE_BASE_PATH):
            raise FileNotFoundError(f"Knowledge base directory not found: {KNOWLEDGE_BASE_PATH}")
        chunks, chunk_id = [], 0
        for filename in sorted(os.listdir(KNOWLEDGE_BASE_PATH)):
            if not filename.lower().endswith(".txt"):
                continue
            path = os.path.join(KNOWLEDGE_BASE_PATH, filename)
            with open(path, "r", encoding="utf-8") as file:
                text = file.read()
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append({"chunk_id": str(chunk_id), "text": chunk_text, "source": filename, "metadata": {"source": filename}})
                    chunk_id += 1
                if end >= len(text):
                    break
                start = end - self.chunk_overlap
        if not chunks:
            raise FileNotFoundError("No policy text files were found in the knowledge base.")
        return chunks

    def retrieve_bge(self, query: str, top_k: int = 5):
        """Retrieve with BGE for the original query and its detected intents."""
        queries = [query]
        lowered = query.lower()
        if "paternity" in lowered:
            queries.append("What is paternity leave and its demonstration entitlement?")
        if "maternity" in lowered:
            queries.append("What is maternity leave and its demonstration entitlement?")
        if any(term in lowered for term in ("working hours", "work time", "office hours", "working days")):
            queries.append("What are the company standard working days, start time, end time, lunch, and working hours?")
        if "leave" in lowered and len(queries) > 1:
            queries.append("What does the company leave policy say about leave types and eligibility?")

        query_vectors = np.asarray(
            self.model.encode(queries, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )
        scores = self.embeddings @ query_vectors.T
        per_query_limit = min(max(1, int(top_k)), len(self.chunks))
        merged = {}
        for query_index in range(len(queries)):
            indices = np.argsort(scores[:, query_index])[::-1][:per_query_limit]
            for index in indices:
                chunk = self.chunks[int(index)]
                chunk_id = chunk["chunk_id"]
                score = float(scores[int(index), query_index])
                existing = merged.get(chunk_id)
                if existing is None or score > existing["score"]:
                    merged[chunk_id] = {
                        "chunk_id": chunk_id,
                        "text": chunk["text"],
                        "score": score,
                        "source": chunk["source"],
                        "metadata": {**chunk["metadata"], "matched_query": queries[query_index]},
                    }

        ranked = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
        max_results = min(len(ranked), max(int(top_k), len(queries) * int(top_k)))
        return [
            {**item, "rank": rank}
            for rank, item in enumerate(ranked[:max_results], 1)
        ]


@st.cache_resource(show_spinner=False)
def load_bge():
    return BGERetriever()


@st.cache_resource(show_spinner=False)
def load_evaluator(_retriever, groq_model, qwen_model):
    return DualLLMEvaluator(_retriever, groq_model, qwen_model)


def _json_cell(value):
    return json.dumps(value, ensure_ascii=True, default=str) if value is not None else ""


def _result_to_csv_row(question, result, model_result, evaluation_id):
    judge = model_result.get("grounding_judge") or {}
    factual_claims = judge.get("factual_claims", [])
    supported_claims = judge.get("supported_claims", [])
    unsupported_claims = model_result.get("unsupported_claims", [])
    contradictions = model_result.get("contradictions", [])
    cost = model_result.get("cost") or {}
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "evaluation_id": evaluation_id,
        "question": question,
        "model": model_result.get("model"),
        "embedding_model": result.get("embedding_model"),
        "temperature": result.get("temperature"),
        "top_k": result.get("top_k"),
        "context_window_tokens": result.get("context_window_tokens"),
        "max_input_tokens": result.get("max_input_tokens"),
        "max_output_tokens": result.get("max_output_tokens"),
        "retrieved_chunks": len(result.get("retrieved_chunks") or []),
        "used_context_chunks": len(result.get("used_chunks") or []),
        "context_tokens": result.get("context_tokens"),
        "input_tokens": model_result.get("prompt_tokens"),
        "output_tokens": model_result.get("completion_tokens"),
        "total_tokens": model_result.get("total_tokens"),
        "retrieval_latency_ms": result.get("retrieval_latency_ms"),
        "generation_latency_ms": model_result.get("generation_latency_ms"),
        "total_latency_ms": result.get("total_latency_ms"),
        "quality_score": judge.get("score"),  # Using grounding score as quality proxy
        "grounding_score": judge.get("score"),
        "hallucination_score": model_result.get("hallucination_score"),
        "hallucination": model_result.get("hallucination"),
        "factual_claims_count": len(factual_claims),
        "supported_claims_count": len(supported_claims),
        "unsupported_claims_count": len(unsupported_claims),
        "contradiction_count": len(contradictions),
        "unsupported_claims": _json_cell(unsupported_claims),
        "contradictions": _json_cell(contradictions),
        "generated_answer": model_result.get("draft_answer"),
        "evaluation_status": model_result.get("evaluation_status", "success"),
        "input_cost": cost.get("input_cost"),
        "output_cost": cost.get("output_cost"),
        "total_cost": cost.get("total_cost"),
        "input_cost_inr": cost.get("input_cost_inr"),
        "output_cost_inr": cost.get("output_cost_inr"),
        "total_cost_inr": cost.get("total_cost_inr"),
        "usd_to_inr": cost.get("usd_to_inr"),
        "pricing_source": cost.get("pricing_source"),
    }


def _append_evaluation_csv(question, result, evaluation_id):
    os.makedirs(os.path.dirname(EVALUATION_CSV_PATH) or PROJECT_ROOT, exist_ok=True)
    file_exists = os.path.exists(EVALUATION_CSV_PATH)
    
    rows_written = 0
    with open(EVALUATION_CSV_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if not file_exists or os.path.getsize(EVALUATION_CSV_PATH) == 0:
            writer.writeheader()
        
        # Write one row for Model A (Groq)
        model_a_result = result.get("model_a", {})
        if model_a_result and model_a_result.get("evaluation_status") != "generation_failed":
            writer.writerow(_result_to_csv_row(question, result, model_a_result, evaluation_id))
            rows_written += 1
        
        # Write one row for Model B (Qwen)
        model_b_result = result.get("model_b", {})
        if model_b_result and model_b_result.get("evaluation_status") != "generation_failed":
            writer.writerow(_result_to_csv_row(question, result, model_b_result, evaluation_id))
            rows_written += 1
    
    with open(EVALUATION_CSV_PATH, "r", newline="", encoding="utf-8") as file:
        return sum(1 for _ in csv.DictReader(file))


def _ensure_evaluation_csv():
    os.makedirs(os.path.dirname(EVALUATION_CSV_PATH) or PROJECT_ROOT, exist_ok=True)
    if not os.path.exists(EVALUATION_CSV_PATH) or os.path.getsize(EVALUATION_CSV_PATH) == 0:
        with open(EVALUATION_CSV_PATH, "w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=CSV_FIELDS).writeheader()


_ensure_evaluation_csv()


def _display_metric(value):
    return f"{value:.2f}" if isinstance(value, (int, float)) else "Unavailable"


def _display_cost(cost, field, currency):
    if not cost or not cost.get("pricing_configured", False):
        return "Not configured"
    value = cost.get(field)
    return f"{currency}{value:.6f}" if currency == "$" else f"{currency}{value:.4f}"


st.title("LLM RAG Comparison Evaluation")
st.caption("User Query  ->  BGE  ->  Policy Context  ->  Model A (Groq) & Model B (Qwen)  ->  Comparison")
st.caption(f"Evaluation results are saved to: {EVALUATION_CSV_PATH}")

with st.sidebar:
    st.header("Generation Settings")
    simple_mode = os.getenv("SIMPLE_EVALUATION_MODE") == "1"
    if simple_mode:
        groq_model = DEFAULT_GROQ_MODEL
        qwen_model = DEFAULT_QWEN_MODEL
    else:
        groq_model = st.text_input("Model A: Groq Model", DEFAULT_GROQ_MODEL)
        qwen_model = st.text_input("Model B: Qwen Model", DEFAULT_QWEN_MODEL)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    top_k = st.slider("BGE Top-K", 1, 12, 5)
    if simple_mode:
        context_window_tokens = 6000
        max_input_tokens = context_window_tokens
        max_output_tokens = 1200
        use_grounding_judge = True
        use_hallucination_guard = True
    else:
        context_window_tokens = st.selectbox("Context Window", [2000, 4000, 6000, 8000, 12000], index=2)
        max_input_tokens = st.slider(
            "Max Input Tokens",
            min_value=1000,
            max_value=12000,
            value=4000,
            step=500,
            help="Maximum prompt tokens sent to each model. Qwen enforces this exactly; Groq usage is reported from the API.",
        )
        max_output_tokens = st.slider(
            "Max Output Tokens",
            min_value=100,
            max_value=2000,
            value=1200,
            step=100,
            help="Maximum tokens for both models to generate.",
        )
        use_grounding_judge = st.checkbox("Grounding Judge", True)
        use_hallucination_guard = st.checkbox("Hallucination Guard", True)


try:
    with st.spinner("Loading BAAI/bge-small-en-v1.5 on CPU..."):
        bge = load_bge()
    evaluator = load_evaluator(bge, groq_model, qwen_model)
except Exception as exc:
    st.error("Failed to initialize the dual LLM evaluator.")
    st.exception(exc)
    st.stop()

query = st.text_area("Enter your leave-policy question", height=120, placeholder="Example: How many annual leave days are employees entitled to?")
if st.button("Run Evaluation", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            import uuid
            evaluation_id = str(uuid.uuid4())[:8]
            with st.spinner("Retrieving BGE context and evaluating both models..."):
                st.session_state["dual_result"] = evaluator.evaluate_dual_models(
                    query.strip(), top_k, temperature, context_window_tokens, max_output_tokens,
                    use_grounding_judge, use_hallucination_guard, max_input_tokens,
                )
            saved_count = _append_evaluation_csv(query.strip(), st.session_state["dual_result"], evaluation_id)
            st.session_state["evaluation_saved_message"] = f"Evaluation saved. Total records: {saved_count} to {EVALUATION_CSV_PATH}"
        except Exception as exc:
            st.error("Dual model evaluation failed.")
            st.exception(exc)

saved_message = st.session_state.get("evaluation_saved_message")
if saved_message:
    st.success(saved_message)

result = st.session_state.get("dual_result")
if result:
    # Shared BGE Retrieved Context
    st.subheader("Shared BGE Retrieved Context")
    st.dataframe(pd.DataFrame([{"Rank": item["rank"], "Chunk Index": item["chunk_id"], "Similarity": item["score"], "Source": item["source"], "Text": item["text"]} for item in result["retrieved_chunks"]]), width="stretch", hide_index=True)
    st.subheader("FINAL CONTEXT SENT TO BOTH LLMs")
    with st.expander("Show exact final context"):
        st.code(result["context"], language="text")
    
    # Model A and Model B Results
    model_a = result.get("model_a", {})
    model_b = result.get("model_b", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Model A: {model_a.get('model', 'Groq')}")
        if model_a.get("error"):
            st.error(f"Generation failed: {model_a['error']}")
        else:
            st.write(model_a.get("final_answer", "No answer generated"))
            st.caption(f"Status: {model_a.get('evaluation_status', 'unknown')}")
        st.caption(
            f"Input tokens: {model_a.get('prompt_tokens', 'N/A') if model_a.get('prompt_tokens') is not None else 'N/A'} | "
            f"Output tokens: {model_a.get('completion_tokens', 'N/A') if model_a.get('completion_tokens') is not None else 'N/A'}"
        )
        st.caption(
            f"Hallucination risk: {_display_metric(model_a.get('hallucination_score'))} "
            "(0.00 = no detected hallucination)"
        )
    
    with col2:
        st.subheader(f"Model B: {model_b.get('model', 'Qwen')}")
        if model_b.get("error"):
            st.error(f"Generation failed: {model_b['error']}")
        else:
            st.write(model_b.get("final_answer", "No answer generated"))
            st.caption(f"Status: {model_b.get('evaluation_status', 'unknown')}")
        st.caption(
            f"Input tokens: {model_b.get('prompt_tokens', 'N/A') if model_b.get('prompt_tokens') is not None else 'N/A'} | "
            f"Output tokens: {model_b.get('completion_tokens', 'N/A') if model_b.get('completion_tokens') is not None else 'N/A'}"
        )
        st.caption(
            f"Hallucination risk: {_display_metric(model_b.get('hallucination_score'))} "
            "(0.00 = no detected hallucination)"
        )
    
    # Comparison Table
    st.subheader("Side-by-Side Comparison")
    comparison_data = {
        "Metric": [
            "Grounding Score",
            "Hallucination Risk (0 = none)",
            "Generation Latency (ms)",
            "Input Tokens",
            "Output Tokens",
            "Total Tokens",
            "Input Cost (USD)",
            "Output Cost (USD)",
            "Cost (USD)",
            "Total Cost (INR)",
            "Unsupported Claims",
            "Contradictions",
        ],
        f"Model A: {model_a.get('model', 'Groq')}": [
            _display_metric(model_a.get("grounding_judge", {}).get("score") if model_a.get("grounding_judge") else model_a.get("lexical_grounding", {}).get("grounding_score")),
            _display_metric(model_a.get("hallucination_score")),
            f"{model_a.get('generation_latency_ms', 0):.0f}" if model_a.get("generation_latency_ms") else "N/A",
            model_a.get("prompt_tokens") if model_a.get("prompt_tokens") is not None else "N/A",
            model_a.get("completion_tokens") if model_a.get("completion_tokens") is not None else "N/A",
            model_a.get("total_tokens") if model_a.get("total_tokens") is not None else "N/A",
            _display_cost(model_a.get("cost"), "input_cost", "$"),
            _display_cost(model_a.get("cost"), "output_cost", "$"),
            _display_cost(model_a.get("cost"), "total_cost", "$"),
            _display_cost(model_a.get("cost"), "total_cost_inr", "Rs. "),
            len(model_a.get("unsupported_claims", [])),
            len(model_a.get("contradictions", [])),
        ],
        f"Model B: {model_b.get('model', 'Qwen')}": [
            _display_metric(model_b.get("grounding_judge", {}).get("score") if model_b.get("grounding_judge") else model_b.get("lexical_grounding", {}).get("grounding_score")),
            _display_metric(model_b.get("hallucination_score")),
            f"{model_b.get('generation_latency_ms', 0):.0f}" if model_b.get("generation_latency_ms") else "N/A",
            model_b.get("prompt_tokens") if model_b.get("prompt_tokens") is not None else "N/A",
            model_b.get("completion_tokens") if model_b.get("completion_tokens") is not None else "N/A",
            model_b.get("total_tokens") if model_b.get("total_tokens") is not None else "N/A",
            _display_cost(model_b.get("cost"), "input_cost", "$"),
            _display_cost(model_b.get("cost"), "output_cost", "$"),
            _display_cost(model_b.get("cost"), "total_cost", "$"),
            _display_cost(model_b.get("cost"), "total_cost_inr", "Rs. "),
            len(model_b.get("unsupported_claims", [])),
            len(model_b.get("contradictions", [])),
        ],
    }
    comparison_frame = pd.DataFrame(comparison_data).astype(str)
    st.dataframe(comparison_frame, width="stretch", hide_index=True)

    st.subheader("Token Usage Comparison")
    model_a_input_tokens = model_a.get("prompt_tokens")
    model_b_input_tokens = model_b.get("prompt_tokens")
    token_comparison = pd.DataFrame({
        "Metric": ["Input Tokens", "Output Tokens", "Total Tokens"],
        f"Model A: {model_a.get('model', 'Groq')}": [
            model_a_input_tokens if model_a_input_tokens is not None else "N/A",
            model_a.get("completion_tokens") if model_a.get("completion_tokens") is not None else "N/A",
            model_a.get("total_tokens") if model_a.get("total_tokens") is not None else "N/A",
        ],
        f"Model B: {model_b.get('model', 'Qwen')}": [
            model_b_input_tokens if model_b_input_tokens is not None else "N/A",
            model_b.get("completion_tokens") if model_b.get("completion_tokens") is not None else "N/A",
            model_b.get("total_tokens") if model_b.get("total_tokens") is not None else "N/A",
        ],
    }).astype(str)
    st.dataframe(token_comparison, width="stretch", hide_index=True)
    if model_a_input_tokens is not None and model_b_input_tokens is not None:
        if model_a_input_tokens < model_b_input_tokens:
            st.info(f"Model A used fewer input tokens ({model_a_input_tokens} vs {model_b_input_tokens}).")
        elif model_b_input_tokens < model_a_input_tokens:
            st.info(f"Model B used fewer input tokens ({model_b_input_tokens} vs {model_a_input_tokens}).")
        else:
            st.info(f"Both models used the same number of input tokens ({model_a_input_tokens}).")
    
    # Cost vs Quality Analysis
    st.subheader("Cost vs Quality Analysis")
    a_score = model_a.get("grounding_judge", {}).get("score") if model_a.get("grounding_judge") else model_a.get("lexical_grounding", {}).get("grounding_score")
    b_score = model_b.get("grounding_judge", {}).get("score") if model_b.get("grounding_judge") else model_b.get("lexical_grounding", {}).get("grounding_score")
    a_cost = model_a.get("cost", {}).get("total_cost") or 0
    b_cost = model_b.get("cost", {}).get("total_cost") or 0
    
    if a_score is not None and b_score is not None and a_cost is not None and b_cost is not None:
        if a_score > b_score:
            st.info(f"Model A achieved higher grounding score ({a_score:.4f} vs {b_score:.4f}).")
        elif b_score > a_score:
            st.info(f"Model B achieved higher grounding score ({b_score:.4f} vs {a_score:.4f}).")
        else:
            st.info(f"Both models achieved equal grounding score ({a_score:.4f}).")
        
        if a_cost < b_cost:
            st.info(f"Model A had lower cost (${a_cost:.6f} vs ${b_cost:.6f}).")
        elif b_cost < a_cost:
            st.info(f"Model B had lower cost (${b_cost:.6f} vs ${a_cost:.6f}).")
        else:
            st.info(f"Both models had equal cost (${a_cost:.6f}).")
        
        if a_score > 0 and b_score > 0:
            a_cost_per_quality = a_cost / a_score if a_score > 0 else float('inf')
            b_cost_per_quality = b_cost / b_score if b_score > 0 else float('inf')
            if a_cost_per_quality < b_cost_per_quality:
                st.info(f"Model A has better cost-per-quality ratio (${a_cost_per_quality:.6f} per quality point vs ${b_cost_per_quality:.6f}).")
            elif b_cost_per_quality < a_cost_per_quality:
                st.info(f"Model B has better cost-per-quality ratio (${b_cost_per_quality:.6f} per quality point vs ${a_cost_per_quality:.6f}).")
    
    # Detailed Grounding & Hallucination for each model
    st.subheader("Detailed Grounding & Hallucination")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Model A: {model_a.get('model', 'Groq')}**")
        if model_a.get("grounding_judge"):
            st.write(f"Reason: {model_a['grounding_judge'].get('reason', 'No explanation')}")
        if model_a.get("unsupported_claims"):
            st.warning(f"Unsupported claims: {'; '.join(map(str, model_a['unsupported_claims']))}")
        if model_a.get("contradictions"):
            st.error(f"Contradictions: {'; '.join(map(str, model_a['contradictions']))}")
    
    with col2:
        st.write(f"**Model B: {model_b.get('model', 'Qwen')}**")
        if model_b.get("grounding_judge"):
            st.write(f"Reason: {model_b['grounding_judge'].get('reason', 'No explanation')}")
        if model_b.get("unsupported_claims"):
            st.warning(f"Unsupported claims: {'; '.join(map(str, model_b['unsupported_claims']))}")
        if model_b.get("contradictions"):
            st.error(f"Contradictions: {'; '.join(map(str, model_b['contradictions']))}")
    
    # Retrieval Metrics
    st.subheader("Retrieval Metrics")
    st.metric("BGE Retrieval Latency", f"{result.get('retrieval_latency_ms', 0):.0f} ms")
    st.metric("Total Evaluation Latency", f"{result.get('total_latency_ms', 0):.0f} ms")
    st.metric("Context Chunks Used", result.get("context_chunks", 0))
