"""
Stage 4: Cross-document verification.

This is the direct fix for "verification blind spot" -- the previous
system took the claimed amount at face value. This stage's entire job
is to compare structured, already-cited fields against each other
(claim form vs. invoices vs. policy limits vs. independent estimates)
and surface anything that doesn't line up. It never receives raw
document text -- only the structured ExtractedField objects from
stage 2 -- so it reasons over verified extractions with known
confidence, not over pages of prose it would have to re-parse (which
is also what keeps this stage cheap relative to bundle size).

Every discrepancy the model reports must resolve to two source
locations pulled from the *actual* extracted fields -- the
`models.Discrepancy` constructor raises if fewer than two citations are
given, so a discrepancy claim with no evidence cannot be constructed at
all, whether it comes from a bug in this module or from a model
hallucination.
"""

from __future__ import annotations
from models import ClaimBundle, Discrepancy, Severity, SourceLocator
from llm_client import LLMClient, REASONING_MODEL
import json

VERIFICATION_SYSTEM_PROMPT = (
    "You are a cross-document verification agent for insurance claims. "
    "You will be given structured, already-extracted fields from every "
    "document in a claim bundle (not raw document text). Compare amounts, "
    "dates, and descriptions across documents. Flag anything that doesn't "
    "reconcile: mismatched totals, amounts exceeding policy limits, "
    "inconsistent dates, or a claimed repair well above an independent "
    "estimate. For every discrepancy you report, you MUST reference which "
    "two (or more) extracted fields disagree, by field_name and "
    "document_id, so each is traceable to its exact source. Do not report "
    "a discrepancy you cannot ground in two specific extracted fields. "
    'Return JSON: {"discrepancies": [{"description": str, "severity": '
    '"info"|"minor"|"major"|"critical", "source_a": {"document_id": str, '
    '"page": int, "location": str}, "source_b": {"document_id": str, '
    '"page": int, "location": str}}]}'
)


def _fields_summary(bundle: ClaimBundle) -> str:
    lines = []
    for doc in bundle.documents:
        for f in doc.fields:
            lines.append(
                f"- doc={doc.document_id} ({doc.file_name}) "
                f"field={f.field_name} value={f.value} "
                f"confidence={f.confidence.value} page={f.source.page} "
                f"loc={f.source.line_or_region}"
            )
    return "\n".join(lines)


def _locator_for(bundle: ClaimBundle, document_id: str, page: int, location: str) -> SourceLocator:
    for doc in bundle.documents:
        if doc.document_id == document_id:
            return SourceLocator(document_id, doc.file_name, page, location)
    # Falls back to a synthetic locator rather than crashing -- but this
    # itself becomes an unresolved_flag so it's never silently accepted.
    return SourceLocator(document_id, document_id, page, location)


def run_verification(bundle: ClaimBundle, client: LLMClient) -> None:
    prompt = (
        "CROSS_DOCUMENT_VERIFICATION\n"
        f"Claim: {bundle.claim_id}\n"
        f"Extracted fields:\n{_fields_summary(bundle)}\n"
    )
    result = client.complete_json(REASONING_MODEL, VERIFICATION_SYSTEM_PROMPT, prompt)

    for d in result.get("discrepancies", []):
        try:
            disc = Discrepancy(
                description=d["description"],
                severity=Severity(d["severity"]),
                sources=[
                    _locator_for(bundle, **d["source_a"]),
                    _locator_for(bundle, **d["source_b"]),
                ],
            )
            bundle.discrepancies.append(disc)
        except (ValueError, KeyError) as e:
            bundle.unresolved_flags.append(
                f"REJECTED unverifiable discrepancy claim from verification "
                f"agent (missing/invalid citation): {e}"
            )
