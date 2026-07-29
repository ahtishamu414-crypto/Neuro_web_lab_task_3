"""
Stage 3: Confidence gate.

This is the pipeline's answer to "uncontained cascading errors". No
stage after this one is allowed to treat a field as settled fact
purely because an earlier stage produced it. Concretely:

  - Every field below HIGH confidence, or coming from a LOW ocr_quality
    document, is tagged into `unresolved_flags` on the bundle.
  - Any financial field (amount-like field names) below HIGH confidence
    is hard-blocked from contributing to an auto-approve decision --
    it can only route to manual review or fraud investigation, never
    silently pass through as a fact.
  - The gate runs again structurally at stage 4 (verification) and
    stage 6 (routing): each stage re-checks the confidence tags it
    receives rather than assuming a clean bill of health from upstream.

The intent is containment, not just detection: a single OCR misread on
one invoice can only ever demote that invoice's fields to
"needs manual verification" -- it cannot silently affect the coverage
determination or fraud score for the rest of the claim.
"""

from __future__ import annotations
from models import ClaimBundle, Confidence

FINANCIAL_FIELD_MARKERS = ("amount", "total", "cost", "line_", "estimate")


def run_confidence_gate(bundle: ClaimBundle) -> None:
    for doc in bundle.documents:
        for f in doc.fields:
            effective_low = (
                f.confidence == Confidence.LOW
                or doc.ocr_quality == Confidence.LOW
            )
            is_financial = any(m in f.field_name for m in FINANCIAL_FIELD_MARKERS)

            if effective_low:
                bundle.unresolved_flags.append(
                    f"LOW CONFIDENCE: {f.field_name}={f.value!r} in "
                    f"{doc.file_name} ({f.source.cite()}) -- excluded from "
                    f"auto-approval eligibility."
                )
            if effective_low and is_financial:
                bundle.unresolved_flags.append(
                    f"BLOCKED: financial field {f.field_name} in "
                    f"{doc.file_name} is below confidence threshold -- "
                    f"claim cannot auto-approve until a human confirms it."
                )
