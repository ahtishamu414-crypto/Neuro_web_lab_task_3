"""
Loads the FAISS policy index and exposes a simple `retrieve(query, k)` API,
plus an uncertainty check based on similarity score.

Both the fine-tuning data prep script and the Streamlit app import this module so
that training-time and inference-time retrieval behavior are identical -- the model
is trained on exactly the kind of retrieved context it will see in production.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path(__file__).parent / "index"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Cosine similarity threshold below which we tell the model (and the UI) that
# retrieval did not find a confident policy match. Tuned empirically on the
# labelled "uncertain_edge_case" examples in data/tickets_dataset.jsonl -- see
# eval/evaluate.py for the threshold sweep.
UNCERTAINTY_THRESHOLD = 0.42


@dataclass
class RetrievedChunk:
    source: str
    header: str
    text: str
    score: float


class PolicyRetriever:
    def __init__(self, index_dir: Path = INDEX_DIR):
        index_path = index_dir / "policy_index.faiss"
        chunks_path = index_dir / "policy_chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Index not found at {index_dir}. Run `python rag/build_index.py` first."
            )
        self.index = faiss.read_index(str(index_path))
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        scores, idxs = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            c = self.chunks[idx]
            results.append(RetrievedChunk(
                source=c["source"], header=c["header"], text=c["text"], score=float(score)
            ))
        return results

    def retrieve_with_uncertainty(self, query: str, k: int = 3):
        """Returns (chunks, is_uncertain). is_uncertain=True means the top match's
        similarity is below UNCERTAINTY_THRESHOLD, i.e. retrieval did not find a
        confident policy match for this query and the model/UI should say so
        rather than answer as if it did."""
        chunks = self.retrieve(query, k=k)
        is_uncertain = (len(chunks) == 0) or (chunks[0].score < UNCERTAINTY_THRESHOLD)
        return chunks, is_uncertain

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "[No matching policy content found.]"
        parts = []
        for c in chunks:
            parts.append(f"[{c.source} \u2014 {c.header}]\n{c.text}")
        return "\n\n".join(parts)


if __name__ == "__main__":
    retriever = PolicyRetriever()
    demo_queries = [
        "customer wants a refund after 3 weeks",
        "does downgrading give me money back",
        "nonprofit discount",
    ]
    for q in demo_queries:
        chunks, uncertain = retriever.retrieve_with_uncertainty(q)
        print(f"\nQuery: {q}\nUncertain: {uncertain}")
        for c in chunks:
            print(f"  score={c.score:.3f} [{c.source} - {c.header}]")
