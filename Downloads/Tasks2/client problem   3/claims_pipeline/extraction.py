"""
Stage 2: Per-document extraction (parallel, isolated).

One model call per document. Each call's context contains exactly one
document's text -- never a bundle of several. This is what prevents
figures from one invoice being attributed to another when a bundle
has several similarly-formatted documents, which is the exact
failure the client's first attempt suffered from.

Cost note: this is the stage that scales with document count (n calls
for n documents), and it runs on the FAST_MODEL tier. Reasoning-tier
calls happen only in stages 4 and 5, over the small structured output
of this stage rather than over raw document text -- this is what keeps
total cost roughly linear (not superlinear) in bundle size.
"""

from __future__ import annotations
from models import DocumentRecord, ExtractedField, SourceLocator, Confidence, DocumentType
from ingestion import classify_document
from llm_client import LLMClient, FAST_MODEL

EXTRACTION_SYSTEM_PROMPT = (
    "You are a document field-extraction agent. You will be given the raw "
    "text of exactly ONE document from an insurance claim bundle. Extract "
    "only fields that are literally present in THIS document's text -- "
    "never infer, assume, or borrow a value from outside this text. For "
    "each field return its page and a specific location description "
    "(section, line number, or table row/column) so the value can be "
    "traced back to its exact source. If you are not confident in a "
    "value (unclear scan, ambiguous phrasing), mark confidence as 'low' "
    "rather than guessing. Return JSON: "
    '{"fields": [{"field_name": str, "value": str, "confidence": '
    '"high"|"medium"|"low", "page": int, "location": str}], '
    '"notes": [str]}'
)


def extract_document(document_id: str, file_name: str, doc_type: str,
                      text: str, client: LLMClient) -> DocumentRecord:
    dtype, ocr_quality = classify_document(document_id, file_name, doc_type)

    user_prompt = (
        f"DOCUMENT_ID: {document_id}\n"
        f"DOCUMENT_NAME: {file_name}\n"
        f"DOCUMENT_TEXT:\n{text}\n"
    )
    result = client.complete_json(FAST_MODEL, EXTRACTION_SYSTEM_PROMPT, user_prompt)

    record = DocumentRecord(
        document_id=document_id,
        file_name=file_name,
        doc_type=dtype,
        ocr_quality=ocr_quality,
    )
    for f in result.get("fields", []):
        source = SourceLocator(
            document_id=document_id,
            document_name=file_name,
            page=f["page"],
            line_or_region=f["location"],
        )
        record.fields.append(ExtractedField(
            field_name=f["field_name"],
            value=f["value"],
            confidence=Confidence(f["confidence"]),
            source=source,
        ))
    record.extraction_notes = result.get("notes", [])
    return record
