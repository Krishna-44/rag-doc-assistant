"""ragdoc — retrieval-augmented question answering over your own documents."""

from ragdoc.documents import Document, load_directory, load_document
from ragdoc.generation import Answer, Citation, ClaudeGenerator, ExtractiveGenerator
from ragdoc.knowledge_base import KnowledgeBase
from ragdoc.pipeline import RAGPipeline
from ragdoc.retrieval import SearchHit

__all__ = [
    "Answer",
    "Citation",
    "ClaudeGenerator",
    "Document",
    "ExtractiveGenerator",
    "KnowledgeBase",
    "RAGPipeline",
    "SearchHit",
    "load_directory",
    "load_document",
]

__version__ = "0.1.0"
