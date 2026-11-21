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


DOC = (
    "Viprasol builds trading software.\n\n"
    "The risk engine checks every order before it reaches the broker.\n\n"
    "Reports are emailed each morning."
)


def test_build_context_ranks_matching_chunks_first():
    rag = RAGBuilder()
    rag.add_document("doc", DOC, {"source": "test"})
    chunks = rag.chunk_documents("paragraphs")
    assert [c["chunk_id"] for c in chunks] == ["doc_chunk_0", "doc_chunk_1", "doc_chunk_2"]
    assert chunks[0]["metadata"] == {"source": "test"}
    context = rag.build_context("which engine checks the order risk", top_k=1)
    assert context.startswith("The risk engine")


def test_generate_rag_prompt_includes_system_context_and_query():
    rag = RAGBuilder()
    rag.add_document("doc", DOC)
    rag.chunk_documents("paragraphs")
    prompt = rag.generate_rag_prompt("When are reports sent?", system_prompt="Be brief.")
    assert prompt.startswith("System: Be brief.")
    assert "Reports are emailed each morning." in prompt
    assert prompt.endswith("Query: When are reports sent?")


def test_cli_outputs_chunks_and_prompt(tmp_path, capsys):
    import json

    from rag_builder import main

    f = tmp_path / "doc.txt"
    f.write_text(DOC, encoding="utf-8")
    main([str(f), "-m", "paragraphs"])
    out = json.loads(capsys.readouterr().out)
    assert out["stats"]["chunks"] == 3

    main([str(f), "-m", "paragraphs", "-q", "risk engine", "-k", "1"])
    assert "The risk engine checks every order" in capsys.readouterr().out


def test_cli_rejects_bad_overlap(tmp_path):
    from rag_builder import main

    f = tmp_path / "doc.txt"
    f.write_text("abc", encoding="utf-8")
    with pytest.raises(SystemExit):
        main([str(f), "--chunk-size", "10", "--overlap", "10"])
