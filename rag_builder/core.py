"""
RAG builder - Simple retrieval-augmented generation.

Document chunking, embedding search, context injection.
"""

import re
from typing import List, Dict, Tuple


class DocumentChunker:
    """Split documents into chunks."""

    @staticmethod
    def chunk_by_size(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        step = chunk_size - overlap

        for i in range(0, len(text), step):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    @staticmethod
    def chunk_by_sentences(text: str, sentences_per_chunk: int = 5) -> List[str]:
        """Split text by sentences."""
        sentences = re.split(r"[.!?]+", text)
        chunks = []

        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ". ".join(sentences[i : i + sentences_per_chunk])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    @staticmethod
    def chunk_by_paragraphs(text: str) -> List[str]:
        """Split text by paragraphs."""
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]


class RAGBuilder:
    """Build RAG systems."""

    def __init__(self):
        """Initialize RAG builder."""
        self.documents: List[Dict] = []
        self.chunks: List[Dict] = []

    def add_document(self, doc_id: str, text: str, metadata: Dict = None) -> None:
        """Add document."""
        self.documents.append({
            "id": doc_id,
            "text": text,
            "metadata": metadata or {},
        })

    def chunk_documents(self, chunk_method: str = "size", **kwargs) -> List[Dict]:
        """Chunk all documents."""
        self.chunks = []

        for doc in self.documents:
            if chunk_method == "size":
                chunks_list = DocumentChunker.chunk_by_size(
                    doc["text"],
                    chunk_size=kwargs.get("chunk_size", 500),
                    overlap=kwargs.get("overlap", 100),
                )
            elif chunk_method == "sentences":
                chunks_list = DocumentChunker.chunk_by_sentences(
                    doc["text"],
                    sentences_per_chunk=kwargs.get("sentences_per_chunk", 5),
                )
            elif chunk_method == "paragraphs":
                chunks_list = DocumentChunker.chunk_by_paragraphs(doc["text"])
            else:
                chunks_list = [doc["text"]]

            for i, chunk in enumerate(chunks_list):
                self.chunks.append({
                    "doc_id": doc["id"],
                    "chunk_id": f"{doc['id']}_chunk_{i}",
                    "text": chunk,
                    "metadata": doc["metadata"],
                })

        return self.chunks

    def build_context(self, query: str, top_k: int = 3) -> str:
        """Build context from most relevant chunks."""
        # Simple keyword matching (use embedding search for production)
        query_words = set(query.lower().split())

        scored = []
        for chunk in self.chunks:
            chunk_words = set(chunk["text"].lower().split())
            score = len(query_words & chunk_words)

            if score > 0:
                scored.append((chunk, score))

        # Sort by relevance and take top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        context_chunks = [chunk["text"] for chunk, _ in scored[:top_k]]

        return "\n\n---\n\n".join(context_chunks)

    def generate_rag_prompt(self, query: str, system_prompt: str = "", top_k: int = 3) -> str:
        """Generate complete RAG prompt."""
        context = self.build_context(query, top_k)

        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")

        prompt_parts.append("Context:")
        prompt_parts.append(context)
        prompt_parts.append(f"\nQuery: {query}")

        return "\n".join(prompt_parts)

    def get_stats(self) -> Dict:
        """Get RAG statistics."""
        total_tokens = sum(len(d["text"].split()) for d in self.documents)
        chunk_tokens = sum(len(c["text"].split()) for c in self.chunks)

        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "total_tokens": total_tokens,
            "avg_chunk_tokens": chunk_tokens / len(self.chunks) if self.chunks else 0,
            "avg_document_tokens": total_tokens / len(self.documents) if self.documents else 0,
        }


def create_rag() -> RAGBuilder:
    """Create RAG builder."""
    return RAGBuilder()


def chunk_text(text: str, method: str = "size") -> List[str]:
    """Quick chunking."""
    if method == "sentences":
        return DocumentChunker.chunk_by_sentences(text)
    elif method == "paragraphs":
        return DocumentChunker.chunk_by_paragraphs(text)
    else:
        return DocumentChunker.chunk_by_size(text)


def process(data: str, **kwargs) -> str:
    """Process function."""
    return "RAG builder ready"


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Build RAG systems with document chunking")
    parser.add_argument("method", nargs="?", default="size", choices=["size", "sentences", "paragraphs"])
    parser.add_argument("--chunk-size", type=int, default=500)

    args = parser.parse_args()

    print(f"RAG builder ready (method: {args.method})")


if __name__ == "__main__":
    main()
