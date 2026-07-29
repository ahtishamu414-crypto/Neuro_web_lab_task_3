"""
Audit trail generator.

Renders a single human-readable decision record per claim: every
extracted fact with its source, every discrepancy, every coverage
finding, the fraud score breakdown, and the final routing decision
with rationale. This is what satisfies "fully auditable... every
approval/denial must be traceable to source evidence" -- the report is
built directly from the same citation-bearing objects the pipeline
used to make the decision, not a separate summary written after the
fact, so it cannot drift from what actually happened.
"""

from __future__ import annotations
from models import ClaimBundle


def render_audit_trail(bundle: ClaimBundle) -> str:
    lines = [f"=== Decision trail: {bundle.claim_id} ===\n"]

    lines.append("-- Documents ingested --")
    for doc in bundle.documents:
        lines.append(f"  {doc.file_name} [{doc.doc_type.value}] "
                      f"(ocr_quality={doc.ocr_quality.value})")
        for f in doc.fields:
            lines.append(f"    - {f.field_name} = {f.value} "
                          f"[{f.confidence.value}] <- {f.source.cite()}")
    lines.append("")

    lines.append("-- Cross-document discrepancies --")
    if bundle.discrepancies:
        for d in bundle.discrepancies:
            cites = "; ".join(s.cite() for s in d.sources)
            lines.append(f"  [{d.severity.value.upper()}] {d.description}")
            lines.append(f"    sources: {cites}")
    else:
        lines.append("  none")
    lines.append("")

    lines.append("-- Coverage findings --")
    for c in bundle.coverage_findings:
        cite = c.policy_citation.cite() if c.policy_citation else "no citation (undetermined)"
        lines.append(f"  {c.claim_item}: covered={c.covered} [{c.confidence.value}]")
        lines.append(f"    rationale: {c.rationale}")
        lines.append(f"    citation: {cite}")
    lines.append("")

    lines.append("-- Fraud assessment --")
    if bundle.fraud_assessment:
        lines.append("  " + bundle.fraud_assessment.explanation().replace("\n", "\n  "))
    else:
        lines.append("  not run")
    lines.append("")

    lines.append("-- Unresolved flags (confidence gate) --")
    if bundle.unresolved_flags:
        for flag in bundle.unresolved_flags:
            lines.append(f"  - {flag}")
    else:
        lines.append("  none")
    lines.append("")

    lines.append("-- Routing decision --")
    lines.append(f"  {bundle.routing.value if bundle.routing else 'NOT ROUTED'}")
    lines.append(f"  Rationale: {bundle.routing_rationale}")

    return "\n".join(lines)
