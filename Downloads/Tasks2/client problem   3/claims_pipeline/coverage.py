"""
Coverage determination.

Direct fix for "silent fabrication" -- the previous system filled gaps
in ambiguous or missing policy terms with a plausible-sounding
assumption. Here, the model is explicitly instructed to output
covered=null plus a rationale whenever a clause is ambiguous or a
required document is missing, and `models.CoverageDetermination`
enforces this at the type level: constructing a determination with
covered=True/False without a policy_citation raises an error. There is
no code path that lets a coverage finding reach the routing stage
without either a citation or an explicit "cannot determine".
"""

from __future__ import annotations
from models import ClaimBundle, CoverageDetermination, Confidence, SourceLocator
from llm_client import LLMClient, REASONING_MODEL

COVERAGE_SYSTEM_PROMPT = (
    "You are a policy coverage determination agent. You will be given the "
    "extracted policy terms and the extracted claim items for one claim. "
    "For each claim item, determine whether it is covered, citing the "
    "exact policy field/section that supports your determination. If the "
    "policy is ambiguous, silent on this item, or a required document is "
    "missing, you MUST set covered to null and explain what is missing in "
    "the rationale -- do NOT guess or assume typical/standard coverage. "
    'Return JSON: {"findings": [{"claim_item": str, "covered": true|false|'
    'null, "policy_document_id": str|null, "policy_page": int|null, '
    '"policy_location": str|null, "rationale": str, "confidence": '
    '"high"|"medium"|"low"}]}'
)


def run_coverage_determination(bundle: ClaimBundle, client: LLMClient) -> None:
    policy_fields, claim_fields = [], []
    for doc in bundle.documents:
        target = policy_fields if doc.doc_type.value == "policy" else claim_fields
        for f in doc.fields:
            target.append(f"{doc.document_id}: {f.field_name}={f.value} "
                           f"(p.{f.source.page}, {f.source.line_or_region})")

    prompt = (
        "COVERAGE_DETERMINATION\n"
        f"Policy fields:\n" + "\n".join(policy_fields) +
        f"\n\nClaim fields:\n" + "\n".join(claim_fields)
    )
    result = client.complete_json(REASONING_MODEL, COVERAGE_SYSTEM_PROMPT, prompt)

    for f in result.get("findings", []):
        citation = None
        if f.get("policy_document_id"):
            doc_name = next(
                (d.file_name for d in bundle.documents if d.document_id == f["policy_document_id"]),
                f["policy_document_id"],
            )
            citation = SourceLocator(
                document_id=f["policy_document_id"],
                document_name=doc_name,
                page=f["policy_page"],
                line_or_region=f["policy_location"],
            )
        try:
            finding = CoverageDetermination(
                claim_item=f["claim_item"],
                covered=f["covered"],
                policy_citation=citation,
                rationale=f["rationale"],
                confidence=Confidence(f["confidence"]),
            )
            bundle.coverage_findings.append(finding)
            if finding.covered is None:
                bundle.unresolved_flags.append(
                    f"COVERAGE UNDETERMINED: {finding.claim_item} -- "
                    f"{finding.rationale}"
                )
        except ValueError as e:
            bundle.unresolved_flags.append(
                f"REJECTED coverage claim with no citation "
                f"({f.get('claim_item')}): {e}"
            )
