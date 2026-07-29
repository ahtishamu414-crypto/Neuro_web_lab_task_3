"""
Memory store used to answer: "do we already know this?" before the assistant
is allowed to ask the patient a clarifying question.

Two layers, checked in order (cheap first):
  1. Structured field lookup — exact match against already-logged IntakeFields.
  2. Semantic search over the raw transcript + document chunks (TF-IDF here;
     swap in a real embedding model / vector DB for production use).

Keeping a running structured summary means the orchestrator never has to
replay the full 100+ turn conversation to the LLM — only the compact
structured profile plus whatever the semantic search retrieves.
"""
from __future__ import annotations

from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models import IntakeField, IntakeSession


@dataclass
class MemoryHit:
    text: str
    source_kind: str    # "message" | "document" | "prior_visit"
    ref_id: str
    score: float


class MemoryStore:
    def __init__(self):
        # chunk_id -> (text, source_kind, ref_id)
        self._chunks: dict[str, tuple[str, str, str]] = {}
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._dirty = True

    # ---- ingestion -------------------------------------------------
    def add_chunk(self, chunk_id: str, text: str, source_kind: str, ref_id: str) -> None:
        self._chunks[chunk_id] = (text, source_kind, ref_id)
        self._dirty = True

    def add_message(self, session: IntakeSession, turn_id: str, text: str) -> None:
        self.add_chunk(f"msg:{turn_id}", text, "message", turn_id)

    def add_document_chunks(self, document_id: str, chunks: list[str]) -> None:
        for i, chunk in enumerate(chunks):
            self.add_chunk(f"doc:{document_id}:{i}", chunk, "document", document_id)

    def add_prior_visit(self, visit_id: str, field_name: str, value: str) -> None:
        self.add_chunk(f"visit:{visit_id}:{field_name}", f"{field_name}: {value}", "prior_visit", visit_id)

    # ---- structured lookup (cheap, exact) ---------------------------
    @staticmethod
    def lookup_field(session: IntakeSession, field_name: str) -> IntakeField | None:
        return session.fields.get(field_name)

    # ---- semantic lookup (fallback, broader) ------------------------
    def _rebuild_index(self) -> None:
        if not self._chunks:
            self._vectorizer, self._matrix = None, None
            self._dirty = False
            return
        texts = [t for t, _, _ in self._chunks.values()]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(texts)
        self._dirty = False

    def search(self, query: str, top_k: int = 3, min_score: float = 0.15) -> list[MemoryHit]:
        """Return the best-matching known chunks for a query, or [] if nothing
        clears the relevance bar — an empty result is the signal to ask."""
        if self._dirty:
            self._rebuild_index()
        if self._vectorizer is None:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ids = list(self._chunks.keys())

        ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
        hits = []
        for chunk_id, score in ranked:
            if score < min_score:
                continue
            text, source_kind, ref_id = self._chunks[chunk_id]
            hits.append(MemoryHit(text=text, source_kind=source_kind, ref_id=ref_id, score=float(score)))
        return hits
