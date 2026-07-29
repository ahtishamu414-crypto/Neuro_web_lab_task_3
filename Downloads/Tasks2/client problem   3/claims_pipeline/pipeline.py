"""
End-to-end orchestrator. Wires the six stages together in a fixed
order. Note the confidence gate (stage 3) runs before verification,
coverage, and fraud scoring all see the extracted fields -- so every
downstream stage already knows which fields are trustworthy before it
reasons over them.
"""

from __future__ import annotations
from models import ClaimBundle
from extraction import extract_document
from confidence_gate import run_confidence_gate
from verification import run_verification
from coverage import run_coverage_determination
from fraud_scoring import run_fraud_scoring
from routing import route
from llm_client import get_client


def adjudicate_claim(claim_id: str, raw_documents: dict) -> ClaimBundle:
    client = get_client()
    bundle = ClaimBundle(claim_id=claim_id)

    # Stage 1 + 2: ingest and extract each document independently.
    for document_id, doc in raw_documents.items():
        record = extract_document(
            document_id=document_id,
            file_name=doc["file_name"],
            doc_type=doc["doc_type"],
            text=doc["text"],
            client=client,
        )
        bundle.documents.append(record)

    # Stage 3: confidence gate -- tag anything downstream must not trust blindly.
    run_confidence_gate(bundle)

    # Stage 4: cross-document verification, over structured fields only.
    run_verification(bundle, client)
    run_coverage_determination(bundle, client)

    # Stage 5: fraud scoring, over discrepancies/findings only.
    run_fraud_scoring(bundle, client)

    # Stage 6: deterministic routing.
    route(bundle)

    return bundle
