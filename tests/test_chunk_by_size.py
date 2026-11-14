import pytest

from rag_builder import DocumentChunker, RAGBuilder


def test_overlapping_chunks():
    chunks = DocumentChunker.chunk_by_size("abcdefghij", chunk_size=4, overlap=2)
    assert chunks == ["abcd", "cdef", "efgh", "ghij", "ij"]


@pytest.mark.parametrize("overlap", [50, 100, 200, -1])
def test_invalid_overlap_raises(overlap):
    with pytest.raises(ValueError):
        DocumentChunker.chunk_by_size("a" * 300, chunk_size=50, overlap=overlap)


def test_non_positive_chunk_size_raises():
    with pytest.raises(ValueError):
        DocumentChunker.chunk_by_size("text", chunk_size=0, overlap=0)


def test_small_chunk_size_with_default_overlap_is_not_silently_empty():
    rag = RAGBuilder()
    rag.add_document("d", "a" * 300)
    with pytest.raises(ValueError):
        rag.chunk_documents("size", chunk_size=50)
    assert len(rag.chunk_documents("size", chunk_size=50, overlap=10)) > 0
