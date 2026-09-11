"""Streamlit UI for the independent single RAG evaluation system."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rag.single_rag_evaluator import DIMENSION, MODEL_NAME, SingleRAGEvaluator

st.set_page_config(page_title="Single RAG Analysis", layout="wide")


@st.cache_resource(show_spinner="Loading BGE-small and policy embeddings...")
def get_evaluator(top_k, chunk_size, chunk_overlap, temperature, max_output_tokens, judge):
    return SingleRAGEvaluator(ROOT, top_k, chunk_size, chunk_overlap, temperature, max_output_tokens, judge)


def average(rows, key):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return sum(values) / len(values) if values else "not_available"


def show_row(row):
    st.subheader(row.get("question") or "Invalid question")
    st.write(row.get("answer") or "No answer")
    with st.expander("Retrieved chunks, context, and metrics"):
        st.code(row.get("context", ""), language="text")
        st.json({key: row.get(key) for key in ("retrieved_chunks", "retrieval_hit", "accuracy_at_k", "precision_at_k", "recall_at_k", "f1_score", "precision_like_score", "mrr", "similarity_score", "average_similarity_score", "latency_ms", "total_latency_ms", "answer_term_coverage", "grounding_score", "grounding_status", "unsupported_claims", "hallucination_risk", "error")})


st.title("Single RAG Analysis")
st.caption("CPU-only evaluation using BAAI/bge-small-en-v1.5 and Qwen 27b.")
with st.sidebar:
    st.header("Configuration")
    top_k = st.slider("Top-K", 1, 10, 3)
    chunk_size = st.number_input("Chunk size (words)", 100, 3000, 900, 50)
    chunk_overlap = st.number_input("Chunk overlap (words)", 0, 1000, 120, 10)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.1)
    max_output_tokens = st.number_input("Max output tokens", 32, 4096, 512, 32)
    judge = st.checkbox("Enable grounding judge", True)
    st.text(f"Embedding: {MODEL_NAME}\nLLM: qwen/qwen3.6-27b\nDataset: {ROOT / 'evaluation_dataset.json'}\nKnowledge base: {ROOT / 'knowledge_base' / 'leave_policy.txt'}")

st.header("System configuration")
st.json({"Embedding": MODEL_NAME, "Dimension": DIMENSION, "LLM": "qwen/qwen3.6-27b", "Knowledge base": "leave_policy.txt", "Dataset": "evaluation_dataset.json", "Device": "CPU"})
try:
    evaluator = get_evaluator(top_k, int(chunk_size), int(chunk_overlap), temperature, int(max_output_tokens), judge)
except Exception as exc:
    st.error(f"Unable to initialize evaluator: {exc}")
    st.stop()

st.header("Single-question test")
question = st.text_area("Question", placeholder="Ask a question about the policy")
if st.button("Run question", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            show_row(evaluator.evaluate_question(question))

st.header("Dataset evaluation")
if st.button("Run Full Dataset Evaluation"):
    progress = st.progress(0.0)
    rows = []
    try:
        dataset = evaluator.load_dataset()
        for index, item in enumerate(dataset, 1):
            if isinstance(item, dict) and item.get("question"):
                rows.append(evaluator.evaluate_question(item["question"], item.get("relevance_terms", []), item.get("expected_answer")))
            else:
                rows.append(evaluator.evaluate_question("", [], None))
                rows[-1]["error"] = "dataset item missing question"
            progress.progress(index / len(dataset))
        latest, timestamped = evaluator.write_csv(rows, ROOT / "evaluation_results")
        st.session_state["rows"] = rows
        st.success(f"Saved {latest.name} and {timestamped.name}")
    except Exception as exc:
        st.error(f"Dataset evaluation failed: {exc}")

rows = st.session_state.get("rows", [])
if rows:
    st.header("Summary metrics")
    successful = sum(not row.get("error") for row in rows)
    labels = ["Total", "Successful", "Failed", "Accuracy@K", "Precision@K", "Recall@K", "F1 score", "MRR", "Avg similarity", "Avg answer coverage", "Avg grounding", "Avg hallucination risk", "Avg latency (ms)", "Avg total latency (ms)", "Avg input tokens", "Avg output tokens"]
    values = [len(rows), successful, len(rows) - successful, average(rows, "accuracy_at_k"), average(rows, "precision_at_k"), average(rows, "recall_at_k"), average(rows, "f1_score"), average(rows, "mrr"), average(rows, "average_similarity_score"), average(rows, "answer_term_coverage"), average(rows, "grounding_score"), average(rows, "hallucination_risk"), average(rows, "latency_ms"), average(rows, "total_latency_ms"), average(rows, "input_tokens"), average(rows, "output_tokens")]
    for start in range(0, len(labels), 4):
        columns = st.columns(4)
        for column, label, value in zip(columns, labels[start:start + 4], values[start:start + 4]):
            column.metric(label, value if isinstance(value, str) else round(value, 4))
    st.dataframe([{key: row.get(key) for key in ("question", "answer", "retrieval_hit", "accuracy_at_k", "precision_at_k", "recall_at_k", "f1_score", "mrr", "similarity_score", "average_similarity_score", "latency_ms", "answer_term_coverage", "grounding_status", "hallucination_risk", "error")} for row in rows], use_container_width=True)
    st.download_button("Download latest CSV", (ROOT / "evaluation_results" / "single_rag_evaluation_latest.csv").read_bytes(), "single_rag_evaluation_latest.csv", "text/csv")
    st.header("Per-question analysis")
    for row in rows:
        show_row(row)
    failures = [row for row in rows if row.get("error") or not row.get("retrieval_hit") or (isinstance(row.get("recall_at_k"), (int, float)) and row["recall_at_k"] < 0.5) or (isinstance(row.get("answer_term_coverage"), (int, float)) and row["answer_term_coverage"] < 0.5) or row.get("grounding_status") != "judge_success" or (isinstance(row.get("hallucination_risk"), (int, float)) and row["hallucination_risk"] > 0.5) or not row.get("answer")]
    st.header("Failure analysis")
    st.dataframe([{key: row.get(key) for key in ("question", "retrieval_hit", "accuracy_at_k", "precision_at_k", "recall_at_k", "f1_score", "mrr", "similarity_score", "average_similarity_score", "latency_ms", "answer_term_coverage", "grounding_status", "hallucination_risk", "error")} for row in failures], use_container_width=True)