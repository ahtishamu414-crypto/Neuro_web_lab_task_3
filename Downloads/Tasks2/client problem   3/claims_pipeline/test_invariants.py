"""
These tests aren't testing business logic -- they're proving the
structural guarantees the write-up claims: that a discrepancy or a
coverage approval literally cannot be constructed without citations,
regardless of what any model returns. Run with: pytest test_invariants.py -v
"""

import pytest
from models import Discrepancy, Severity, CoverageDetermination, Confidence, SourceLocator

LOC_A = SourceLocator("d1", "invoice.pdf", 1, "line 3")
LOC_B = SourceLocator("d2", "policy.pdf", 2, "section 4")


def test_discrepancy_requires_two_citations():
    with pytest.raises(ValueError):
        Discrepancy(description="mismatch", severity=Severity.MAJOR, sources=[LOC_A])


def test_discrepancy_with_two_citations_succeeds():
    d = Discrepancy(description="mismatch", severity=Severity.MAJOR, sources=[LOC_A, LOC_B])
    assert len(d.sources) == 2


def test_coverage_true_without_citation_rejected():
    with pytest.raises(ValueError):
        CoverageDetermination(
            claim_item="x", covered=True, policy_citation=None,
            rationale="looks fine", confidence=Confidence.HIGH,
        )


def test_coverage_none_without_citation_allowed():
    # Explicitly "cannot determine" never requires a citation --
    # this is the escape hatch that replaces silent fabrication.
    c = CoverageDetermination(
        claim_item="x", covered=None, policy_citation=None,
        rationale="policy silent on this item, no rider on file",
        confidence=Confidence.MEDIUM,
    )
    assert c.covered is None


def test_coverage_true_with_citation_succeeds():
    c = CoverageDetermination(
        claim_item="x", covered=True, policy_citation=LOC_B,
        rationale="within limit", confidence=Confidence.HIGH,
    )
    assert c.covered is True
