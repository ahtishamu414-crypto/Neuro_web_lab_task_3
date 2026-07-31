"""
Chunk ByteHive's policy markdown files and build a FAISS similarity index over them.

Run this once (or whenever policies/*.md changes) before using rag/retriever.py or
the Streamlit app:

    python rag/build_index.py

Produces:
    rag/index/policy_index.faiss   -- FAISS vector index
    rag/index/policy_chunks.json   -- chunk text + metadata, aligned by row id with
                                       the FAISS index
"""
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

POLICY_DIR = Path(__file__).parent.parent / "data" / "policies"
INDEX_DIR = Path(__file__).parent / "index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small, free, CPU-friendly


def chunk_markdown(text: str, source: str) -> list[dict]:
    """Split a policy markdown file into chunks along '## ' headers.

    Each chunk keeps its header as context, which materially helps retrieval
    precision since headers ('## Refunds After the 14-Day Window') are close
    paraphrases of the kinds of questions agents/customers ask.
    """
    # Split on level-2 headers, keep the header text attached to its section.
    sections = re.split(r"(?m)^## ", text)
    chunks = []
    # sections[0] is the H1 title / preamble before the first "## "
    preamble = sections[0].strip()
    body_sections = sections[1:]
    for sec in body_sections:
        lines = sec.strip().split("\n", 1)
        header = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        chunk_text = f"## {header}\n{body}".strip()
        chunks.append({
            "source": source,
            "header": header,
            "text": chunk_text,
        })
    if not chunks and preamble:
        chunks.append({"source": source, "header": "(full document)", "text": preamble})
    return chunks


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    for md_path in sorted(POLICY_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_markdown(text, source=md_path.name))

    print(f"Built {len(all_chunks)} policy chunks from {POLICY_DIR}")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    # Inner product on normalized vectors == cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "policy_index.faiss"))
    with (INDEX_DIR / "policy_chunks.json").open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Saved index + {len(all_chunks)} chunks to {INDEX_DIR}")


if __name__ == "__main__":
    main()
