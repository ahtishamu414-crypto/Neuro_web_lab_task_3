"""
Stage 1: Ingest and classify.

Each document is handled independently here -- this module never sees
more than one document's raw text at a time. That isolation is
deliberate: it is the first line of defense against "monolithic
confusion" (mixing figures between similarly-formatted documents),
because there is no shared context for figures to leak across until
the *structured* extractions (not raw text) reach the verification
stage in module 4.

In a production system this stage would run real OCR (e.g. for scanned
or photographed documents) and assign ocr_quality from OCR engine
confidence scores. This demo takes clean text input and assigns a
quality tag, since the sample bundle has no scanned/low-quality files.
"""

from __future__ import annotations
from models import DocumentType, Confidence


def classify_document(document_id: str, file_name: str, doc_type: str) -> tuple[DocumentType, Confidence]:
    try:
        dtype = DocumentType(doc_type)
    except ValueError:
        dtype = DocumentType.OTHER
    # Stand-in for real OCR confidence; a scanned/blurry page would
    # score lower here and propagate that lower confidence onto every
    # field extracted from it in the next stage.
    quality = Confidence.HIGH
    return dtype, quality
