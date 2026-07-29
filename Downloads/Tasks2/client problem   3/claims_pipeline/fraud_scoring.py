"""
Stage 5: Fraud risk scoring.

Takes the discrepancies and coverage findings already produced by
stages 3-4 (not raw documents) and produces a weighted, itemized score.
Every contributing signal must cite the source field/document it's
based on -- this keeps the score explainable to a regulator ("why did
this claim score 68?") rather than an opaque single number.
"""

from __future__ import annotations
from models import ClaimBundle, FraudAssessment, FraudSignal, SourceLocator
from llm_client import LLMClient, REASONING_MODEL

FRAUD_SYSTEM_PROMPT = (
    "You are a fraud risk scoring agent for insurance claims. You will be "
    "given the discrepancies and coverage findings already identified for "
    "a claim (not raw documents). Produce a 0-100 fraud risk score and an "
    "itemized breakdown of contributing signals. Every signal must cite "
    "the specific document/field it is based on and include a weight "
    "(points contributed to the total). Do not include a signal you "
    "cannot ground in the given discrepancies/findings. "
    'Return JSON: {"score": number, "signals": [{"name": str, "weight": '
    'number, "evidence": str, "source_document_id": str, "page": int, '
    '"location": str}]}'
)


def run_fraud_scoring(bundle: ClaimBundle, client: LLMClient) -> None:
    disc_summary = "\n".join(
        f"- [{d.severity.value}] {d.description} "
        f"(sources: {'; '.join(s.cite() for s in d.sources)})"
        for d in bundle.discrepancies
    ) or "none"
    coverage_summary = "\n".join(
        f"- {c.claim_item}: covered={c.covered} -- {c.rationale}"
        for c in bundle.coverage_findings
    ) or "none"

    prompt = (
        "FRAUD_SCORING\n"
        f"Discrepancies:\n{disc_summary}\n\n"
        f"Coverage findings:\n{coverage_summary}\n"
    )
    result = client.complete_json(REASONING_MODEL, FRAUD_SYSTEM_PROMPT, prompt)

    signals = []
    for s in result.get("signals", []):
        doc_name = next(
            (d.file_name for d in bundle.documents if d.document_id == s["source_document_id"]),
            s["source_document_id"],
        )
        signals.append(FraudSignal(
            name=s["name"],
            weight=s["weight"],
            evidence=s["evidence"],
            sources=[SourceLocator(
                document_id=s["source_document_id"],
                document_name=doc_name,
                page=s["page"],
                line_or_region=s["location"],
            )],
        ))
    bundle.fraud_assessment = FraudAssessment(score=result.get("score", 0), signals=signals)
