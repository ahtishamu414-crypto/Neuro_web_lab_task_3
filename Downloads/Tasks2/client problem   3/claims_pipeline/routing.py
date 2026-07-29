"""
Stage 6: Routing.

Deliberately NOT a model call. The approve/review/investigate boundary
is the single highest-stakes decision in the pipeline and the one a
regulator will ask about most, so it is plain, deterministic Python --
auditable by reading the code, not by trusting a model's judgement.
Every upstream stage already did the judgement work (verification,
coverage, fraud scoring); this stage only applies fixed thresholds to
their outputs.

Auto-approve requires ALL of:
  - no unresolved_flags (nothing blocked by the confidence gate)
  - no MAJOR or CRITICAL discrepancies
  - every coverage finding resolved (none is covered=None)
  - fraud score under FRAUD_INVESTIGATION_THRESHOLD

Anything short of that falls through to manual review, and a high
fraud score routes straight to investigation regardless of how clean
everything else looks.
"""

from __future__ import annotations
from models import ClaimBundle, RoutingDecision, Severity

FRAUD_INVESTIGATION_THRESHOLD = 60
FRAUD_REVIEW_THRESHOLD = 30


def route(bundle: ClaimBundle) -> None:
    fraud_score = bundle.fraud_assessment.score if bundle.fraud_assessment else 0
    major_discrepancies = [d for d in bundle.discrepancies
                            if d.severity in (Severity.MAJOR, Severity.CRITICAL)]
    undetermined_coverage = [c for c in bundle.coverage_findings if c.covered is None]

    if fraud_score >= FRAUD_INVESTIGATION_THRESHOLD:
        bundle.routing = RoutingDecision.FRAUD_INVESTIGATION
        bundle.routing_rationale = (
            f"Fraud score {fraud_score:.0f} >= threshold "
            f"{FRAUD_INVESTIGATION_THRESHOLD}. Routed to fraud investigation "
            f"regardless of other findings."
        )
        return

    if (bundle.unresolved_flags or major_discrepancies or undetermined_coverage
            or fraud_score >= FRAUD_REVIEW_THRESHOLD):
        reasons = []
        if bundle.unresolved_flags:
            reasons.append(f"{len(bundle.unresolved_flags)} unresolved confidence/citation flag(s)")
        if major_discrepancies:
            reasons.append(f"{len(major_discrepancies)} major/critical discrepancy(ies)")
        if undetermined_coverage:
            reasons.append(f"{len(undetermined_coverage)} undetermined coverage finding(s)")
        if fraud_score >= FRAUD_REVIEW_THRESHOLD:
            reasons.append(f"fraud score {fraud_score:.0f} at/above review threshold {FRAUD_REVIEW_THRESHOLD}")
        bundle.routing = RoutingDecision.MANUAL_REVIEW
        bundle.routing_rationale = "Routed to manual review: " + "; ".join(reasons)
        return

    bundle.routing = RoutingDecision.AUTO_APPROVE
    bundle.routing_rationale = (
        "All checks passed with full confidence and full citation "
        "coverage: no unresolved flags, no major/critical discrepancies, "
        f"all coverage determined, fraud score {fraud_score:.0f} below "
        f"review threshold."
    )
