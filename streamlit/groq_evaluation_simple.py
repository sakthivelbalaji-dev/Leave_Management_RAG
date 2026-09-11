"""Simple LLM comparison page with only Temperature and BGE Top-K controls."""

import os
import runpy


os.environ["SIMPLE_EVALUATION_MODE"] = "1"
runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "groq_evaluation.py"),
    run_name="__main__",
)