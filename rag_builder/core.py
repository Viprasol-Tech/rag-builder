"""
RAG builder - Simple retrieval-augmented generation.

Document chunking, embedding search, context injection.
"""

import re


class DocumentChunker:
    """Split documents into chunks."""

    @staticmethod
    def chunk_by_size(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
        """Split text into overlapping chunks.

        Raises ValueError if chunk_size is not positive or overlap is not
        in the range 0 <= overlap < chunk_size.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= overlap < chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size")

        chunks = []
        step = chunk_size - overlap

        for i in range(0, len(text), step):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    @staticmethod
    def chunk_by_sentences(text: str, sentences_per_chunk: int = 5) -> list[str]:
        """Split text by sentences."""
        sentences = re.split(r"[.!?]+", text)
        chunks = []

        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ". ".join(sentences[i : i + sentences_per_chunk])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    @staticmethod
    def chunk_by_paragraphs(text: str) -> list[str]:
        """Split text by paragraphs."""
        paragraphs = text.split("\n\n")
        return [p.strip() for p in paragraphs if p.strip()]


class RAGBuilder:
    """Build RAG systems."""

    def __init__(self):
        """Initialize RAG builder."""
        self.documents: list[dict] = []
        self.chunks: list[dict] = []

    def add_document(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        """Add document."""
        self.documents.append(
            {
                "id": doc_id,
                "text": text,
                "metadata": metadata or {},
            }
        )

    def chunk_documents(self, chunk_method: str = "size", **kwargs) -> list[dict]:
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
                self.chunks.append(
                    {
                        "doc_id": doc["id"],
                        "chunk_id": f"{doc['id']}_chunk_{i}",
                        "text": chunk,
                        "metadata": doc["metadata"],
                    }
                )

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

    def get_stats(self) -> dict:
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


def chunk_text(text: str, method: str = "size") -> list[str]:
    """Quick chunking."""
    if method == "sentences":
        return DocumentChunker.chunk_by_sentences(text)
    elif method == "paragraphs":
        return DocumentChunker.chunk_by_paragraphs(text)
    else:
        return DocumentChunker.chunk_by_size(text)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: chunk a file (or stdin) and optionally rank chunks for a query."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        prog="rag-builder",
        description="Chunk a document and build a retrieval-augmented prompt.",
    )
    parser.add_argument("file", nargs="?", help="Text file to chunk (reads stdin if omitted)")
    parser.add_argument(
        "-m", "--method", default="size", choices=["size", "sentences", "paragraphs"]
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="Characters per chunk")
    parser.add_argument("--overlap", type=int, default=100, help="Characters shared by chunks")
    parser.add_argument("-q", "--query", help="Print a RAG prompt for this query instead")
    parser.add_argument("-k", "--top-k", type=int, default=3, help="Chunks to include")
    args = parser.parse_args(argv)

    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()

    rag = RAGBuilder()
    rag.add_document(args.file or "stdin", text)
    try:
        rag.chunk_documents(args.method, chunk_size=args.chunk_size, overlap=args.overlap)
    except ValueError as exc:
        parser.error(str(exc))

    if args.query:
        print(rag.generate_rag_prompt(args.query, top_k=args.top_k))
    else:
        print(json.dumps({"stats": rag.get_stats(), "chunks": rag.chunks}, indent=2))


if __name__ == "__main__":
    main()
