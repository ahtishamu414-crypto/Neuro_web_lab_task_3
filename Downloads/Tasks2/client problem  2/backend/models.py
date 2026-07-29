"""
Core data models for the intake assistant.

Every extracted clinical field carries a `status` (patient_stated / inferred /
unresolved) and a `source` back to the exact turn or document that produced
it. This is what makes the system auditable and stops silent guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import uuid


class FieldStatus(str, Enum):
    PATIENT_STATED = "patient_stated"   # verbatim / directly confirmed by patient
    INFERRED = "inferred"               # model inference, not yet confirmed
    UNRESOLVED = "unresolved"           # genuinely unknown, flagged for follow-up


class Urgency(str, Enum):
    ROUTINE = "routine"
    SOON = "soon"                # see within a few days
    URGENT = "urgent"            # same-day physician review
    EMERGENCY = "emergency"      # red-flag, escalate immediately


@dataclass
class SourceRef:
    """Where a piece of information came from — required for every field."""
    kind: str                     # "message" | "document" | "prior_visit"
    ref_id: str                   # turn id, document id, or prior visit id
    snippet: str                  # short verbatim excerpt supporting the value


@dataclass
class IntakeField:
    field_name: str               # e.g. "chief_complaint", "pain_location"
    value: Optional[str]
    status: FieldStatus
    sources: list[SourceRef] = field(default_factory=list)
    contradiction_with: Optional[SourceRef] = None
    confidence_note: str = ""     # short human-readable reasoning
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ConversationTurn:
    turn_id: str
    speaker: str                  # "patient" | "assistant"
    text: str
    document_ids: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AuditLogEntry:
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action: str = ""              # "log_field" | "ask_clarifying_question" | "flag_contradiction"
    detail: dict = field(default_factory=dict)
    reasoning: str = ""            # why the assistant took this action
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class IntakeSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str = ""
    turns: list[ConversationTurn] = field(default_factory=list)
    fields: dict[str, IntakeField] = field(default_factory=dict)
    audit_log: list[AuditLogEntry] = field(default_factory=list)
    urgency: Urgency = Urgency.ROUTINE
