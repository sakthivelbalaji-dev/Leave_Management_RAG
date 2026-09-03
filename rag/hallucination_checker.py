import re

from .embeddings import EmbeddingModel
from .prompt_builder import REFUSAL


_embedding_model = None


def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:
        _embedding_model = EmbeddingModel()

    return _embedding_model


def split_sentences(text: str):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def check_hallucination(
    answer: str,
    context: str,
    sentence_threshold: float = 0.45,
):

    if not answer or not answer.strip():

        return {
            "hallucination_score": 1.0,
            "grounded": False,
        }

    if answer.strip() == REFUSAL:

        return {
            "hallucination_score": 0.0,
            "grounded": True,
        }

    if not context or not context.strip():

        return {
            "hallucination_score": 1.0,
            "grounded": False,
        }

    sentences = split_sentences(answer)

    if not sentences:

        return {
            "hallucination_score": 1.0,
            "grounded": False,
        }

    model = get_embedding_model()

    answer_embeddings = model.encode(
        sentences
    )

    context_sentences = split_sentences(
        context
    )

    if not context_sentences:

        context_sentences = [context]

    context_embeddings = model.encode(
        context_sentences
    )

    grounded_scores = []

    for answer_embedding in answer_embeddings:

        scores = context_embeddings @ answer_embedding

        best_score = float(scores.max())

        grounded_scores.append(
            best_score
        )

    average_score = sum(
        grounded_scores
    ) / len(grounded_scores)

    weakest_score = min(
        grounded_scores
    )

    # A response is grounded only when:
    #
    # 1. Average semantic support is sufficient
    # 2. No individual sentence is extremely unsupported

    grounded = (
        average_score >= sentence_threshold
        and weakest_score >= sentence_threshold - 0.08
    )

    hallucination_score = round(
        max(
            0.0,
            1.0 - average_score
        ),
        4,
    )

    return {
        "hallucination_score": hallucination_score,
        "grounded": grounded,
    }