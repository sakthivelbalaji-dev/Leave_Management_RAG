"""RAG package exports with lazy loading.

The standalone evaluator must be importable without initializing the legacy
pipeline. Legacy symbols remain available when explicitly requested.
"""

_EXPORTS = {
    "KnowledgeBaseLoader": ("loader", "KnowledgeBaseLoader"),
    "TextChunker": ("chunker", "TextChunker"),
    "EmbeddingModel": ("embeddings", "EmbeddingModel"),
    "FAISSVectorStore": ("vector_store", "FAISSVectorStore"),
    "Retriever": ("retriever", "Retriever"),
    "ContextBuilder": ("context_builder", "ContextBuilder"),
    "build_system_prompt": ("prompt_builder", "build_system_prompt"),
    "build_user_prompt": ("prompt_builder", "build_user_prompt"),
    "check_hallucination": ("hallucination_checker", "check_hallucination"),
    "SingleRAGRetriever": ("single_rag_retriever", "SingleRAGRetriever"),
    "RAGPipeline": ("initializer", "RAGPipeline"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    import importlib
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value
