<div align="center">

<img src="docs/assets/logo.png" alt="Viprasol Tech logo" width="120" />

# rag-builder

**Chunk your documents and build retrieval-augmented prompts in pure Python.**

_Built and maintained by [Viprasol Tech](https://viprasol.com) — Fintech Experts. Full-Stack Builders._

[![CI](https://img.shields.io/github/actions/workflow/status/Viprasol-Tech/rag-builder/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/Viprasol-Tech/rag-builder/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-black?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen?style=flat-square)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230?style=flat-square)](https://github.com/astral-sh/ruff)

</div>

---

`rag-builder` covers the plumbing of a RAG pipeline: split documents into chunks, keep each chunk's source and metadata, pick the chunks that match a question, and assemble the final prompt. It uses only the standard library, so it runs anywhere and is easy to read when you want to see how RAG works.

> **Retrieval is keyword overlap.** `build_context` scores chunks by the words they share with the query. That is enough for small document sets, demos and tests. For production search, put an embedding model or vector database behind the same chunk records.

## ✨ Features

- ✂️ **Three chunkers:** fixed size with overlap, by sentences, or by paragraphs.
- 🏷️ **Traceable chunks:** every chunk keeps its `doc_id`, a stable `chunk_id` and the document's metadata.
- 🎯 **Top-k context** ranked by query-word overlap.
- 🧩 **Prompt assembly** with an optional system prompt, the context and the query.
- 📊 **Stats:** document, chunk and average word counts.
- 🖥️ **CLI** (`rag-builder`) that reads a file or stdin; zero runtime dependencies.

## 🚀 Quickstart

```bash
git clone https://github.com/Viprasol-Tech/rag-builder.git
cd rag-builder
pip install -e .

rag-builder notes.txt -m paragraphs                          # JSON: stats + chunks
rag-builder notes.txt -m paragraphs -q "risk engine" -k 2    # print a RAG prompt
cat notes.txt | rag-builder --chunk-size 800 --overlap 150   # read from stdin
```

## 🐍 Python API

```python
from rag_builder import DocumentChunker, RAGBuilder

rag = RAGBuilder()
rag.add_document("handbook", open("handbook.txt", encoding="utf-8").read(), {"source": "hr"})
rag.chunk_documents("size", chunk_size=500, overlap=100)  # or "sentences" / "paragraphs"

prompt = rag.generate_rag_prompt(
    "How many leave days do new staff get?",
    system_prompt="Answer only from the context.",
    top_k=3,
)
print(prompt)
print(rag.get_stats())

DocumentChunker.chunk_by_size("abcdefghij", chunk_size=4, overlap=2)
# ['abcd', 'cdef', 'efgh', 'ghij', 'ij']
```

`chunk_by_size` raises `ValueError` unless `chunk_size > 0` and `0 <= overlap < chunk_size`.

## ⚙️ CLI options

| Option | Meaning |
|---|---|
| `file` | Text file to read (stdin if omitted) |
| `-m, --method` | `size` (default), `sentences` or `paragraphs` |
| `--chunk-size`, `--overlap` | Characters per chunk and shared between chunks (`size` only) |
| `-q, --query` | Print a RAG prompt for this question instead of the chunks |
| `-k, --top-k` | Number of chunks to put in the prompt (default 3) |

## 🧪 Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest
```

## 🤝 Contributing

PRs are welcome. Run `ruff check . && ruff format --check . && pytest` before you open one.

## 📬 Contact — Viprasol Tech Private Limited

- 🌐 Website: [viprasol.com](https://viprasol.com)
- ✉️ Email: [support@viprasol.com](mailto:support@viprasol.com)
- 💬 Telegram: [t.me/viprasol_help](https://t.me/viprasol_help) · 📱 WhatsApp: +91 96336 52112
- 🐙 GitHub: [@Viprasol-Tech](https://github.com/Viprasol-Tech) · 💼 [LinkedIn](https://www.linkedin.com/in/viprasol/) · 𝕏 [@viprasol](https://twitter.com/viprasol)

> *Viprasol Tech builds fintech software, algorithmic trading systems, MT4/MT5 bots, AI voice agents, and B2B SaaS. Need a custom build? [Get in touch](mailto:support@viprasol.com).*

## License

[MIT](LICENSE) © 2026 Viprasol Tech Private Limited
