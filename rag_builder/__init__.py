"""rag-builder: chunk documents and assemble retrieval-augmented prompts."""

__version__ = "0.2.0"

from .core import DocumentChunker, RAGBuilder, chunk_text, create_rag, main

__all__ = ["DocumentChunker", "RAGBuilder", "chunk_text", "create_rag", "main"]
