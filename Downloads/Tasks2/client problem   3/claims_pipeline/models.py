"""
Core data models for the claims adjudication pipeline.

Design principle: nothing downstream is allowed to consume a bare value.
Every extracted fact is wrapped in a Citation-bearing object, so a
verification/fraud/routing stage physically cannot "know" a number
without also knowing where it came from and how confident the
extractor was in it. This is what makes the audit trail possible and
what stops silent fabrication -- there is no field for "the model's
opinion of coverage" that lacks a source pointer.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid


class Confidence(str, Enum):
    HIGH = "high"       # >= 0.9, corroborated or clean text extraction
    MEDIUM = "medium"   # 0.6 - 0.9, single-source, clean OCR
    LOW = "low"         # < 0.6, poor OCR / ambiguous / conflicting


class DocumentType(str, Enum):
    POLICY = "policy"
    CLAIM_FORM = "claim_form"
    INVOICE = "invoice"
    MEDICAL_BILL = "medical_bill"
    DAMAGE_REPORT = "damage_report"
    OTHER = "other"


@dataclass(frozen=True)
class SourceLocator:
    """Exact pointer into a source document -- required, never optional."""
    document_id: str
    document_name: str
    page: int
    line_or_region: str  # e.g. "line 14" or "table row 3, col 'Amount'"

    def cite(self) -> str:
        return f"{self.document_name} (p.{self.page}, {self.line_or_region})"


@dataclass
class ExtractedField:
    """A single fact pulled from a document. Confidence is mandatory."""
    field_name: str
    value: str
    confidence: Confidence
    source: SourceLocator
    raw_ocr_confidence: Optional[float] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class DocumentRecord:
    document_id: str
    file_name: str
    doc_type: DocumentType
    ocr_quality: Confidence  # overall scan/photo quality signal
    fields: list[ExtractedField] = field(default_factory=list)
    extraction_notes: list[str] = field(default_factory=list)


class Severity(str, Enum):
    INFO = "info"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


@dataclass
class Discrepancy:
    """A cross-document mismatch. Must cite at least two locations."""
    description: str
    severity: Severity
    sources: list[SourceLocator]

    def __post_init__(self):
        if len(self.sources) < 2:
            raise ValueError(
                "A discrepancy must cite at least two source locations "
                "(the two things that disagree) -- unverifiable claims "
                "are rejected at construction time, not caught later."
            )


@dataclass
class CoverageDetermination:
    """A policy coverage finding. Refuses to exist without a citation."""
    claim_item: str
    covered: Optional[bool]  # None = "cannot determine" (never guessed)
    policy_citation: Optional[SourceLocator]
    rationale: str
    confidence: Confidence

    def __post_init__(self):
        if self.covered is not None and self.policy_citation is None:
            raise ValueError(
                "A coverage determination of True/False requires a "
                "policy_citation. Use covered=None + rationale to flag "
                "an ambiguous or missing clause for human review instead "
                "of fabricating a determination."
            )


@dataclass
class FraudSignal:
    name: str
    weight: float
    evidence: str
    sources: list[SourceLocator]


@dataclass
class FraudAssessment:
    score: float  # 0-100
    signals: list[FraudSignal]

    def explanation(self) -> str:
        lines = [f"Fraud risk score: {self.score:.0f}/100"]
        for s in sorted(self.signals, key=lambda x: -x.weight):
            cites = "; ".join(c.cite() for c in s.sources)
            lines.append(f"  [+{s.weight:.0f}] {s.name}: {s.evidence} ({cites})")
        return "\n".join(lines)


class RoutingDecision(str, Enum):
    AUTO_APPROVE = "auto_approve"
    MANUAL_REVIEW = "manual_review"
    FRAUD_INVESTIGATION = "fraud_investigation"


@dataclass
class ClaimBundle:
    claim_id: str
    documents: list[DocumentRecord] = field(default_factory=list)
    discrepancies: list[Discrepancy] = field(default_factory=list)
    coverage_findings: list[CoverageDetermination] = field(default_factory=list)
    fraud_assessment: Optional[FraudAssessment] = None
    routing: Optional[RoutingDecision] = None
    routing_rationale: str = ""
    unresolved_flags: list[str] = field(default_factory=list)
