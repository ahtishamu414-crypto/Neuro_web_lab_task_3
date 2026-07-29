"""
Tool implementations + schemas. The orchestrator forces a specific order via
the system prompt: the model must call search_known_context (and, if that
comes back empty, check_contradiction) before it is allowed to call
ask_clarifying_question. log_field is the only way information ever reaches
the patient record, and it always requires a status + source.
"""
from __future__ import annotations

from models import AuditLogEntry, FieldStatus, IntakeField, IntakeSession, SourceRef
from memory import MemoryStore

TOOL_SCHEMAS = [
    {
        "name": "search_known_context",
        "description": (
            "Search the patient's conversation history, uploaded documents, and "
            "prior visit records for information relevant to a query. ALWAYS call "
            "this before asking the patient a clarifying question — never ask for "
            "something that might already be known."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What you're trying to find out, in plain language."},
                "field_name": {"type": "string", "description": "The structured field this relates to, e.g. 'pain_location'."},
            },
            "required": ["query", "field_name"],
        },
    },
    {
        "name": "check_contradiction",
        "description": (
            "Compare a new candidate value for a field against what is already "
            "logged (this session or a prior visit). Call this before logging any "
            "field that search_known_context returned prior information for."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "field_name": {"type": "string"},
                "new_value": {"type": "string"},
            },
            "required": ["field_name", "new_value"],
        },
    },
    {
        "name": "ask_clarifying_question",
        "description": (
            "Ask the patient a clarifying question. Only call this after "
            "search_known_context returned nothing usable for this field — never "
            "re-ask something already known."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "field_name": {"type": "string"},
                "reason": {"type": "string", "description": "Why this couldn't be resolved from existing context."},
            },
            "required": ["question", "field_name", "reason"],
        },
    },
    {
        "name": "log_field",
        "description": (
            "Record a structured intake field. status must be 'patient_stated' "
            "only if the patient said it directly or confirmed it; use 'inferred' "
            "for a reasonable but unconfirmed reading, and 'unresolved' if it "
            "genuinely isn't known yet. Never fabricate a value."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "field_name": {"type": "string"},
                "value": {"type": "string"},
                "status": {"type": "string", "enum": ["patient_stated", "inferred", "unresolved"]},
                "source_kind": {"type": "string", "enum": ["message", "document", "prior_visit"]},
                "source_ref_id": {"type": "string"},
                "source_snippet": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["field_name", "value", "status", "source_kind", "source_ref_id", "reasoning"],
        },
    },
    {
        "name": "flag_contradiction",
        "description": (
            "Surface a contradiction between the patient's current statement and "
            "an existing record. Never silently overwrite — this keeps both "
            "values visible to the physician until the patient confirms which is current."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "field_name": {"type": "string"},
                "new_value": {"type": "string"},
                "existing_value": {"type": "string"},
                "existing_source_kind": {"type": "string"},
                "existing_source_ref_id": {"type": "string"},
            },
            "required": ["field_name", "new_value", "existing_value", "existing_source_kind", "existing_source_ref_id"],
        },
    },
]


def dispatch_tool(name: str, tool_input: dict, session: IntakeSession, memory: MemoryStore) -> dict:
    """Executes a tool call and returns the result to feed back to the model."""
    if name == "search_known_context":
        hits = memory.search(tool_input["query"])
        structured = memory.lookup_field(session, tool_input["field_name"])
        result = {
            "structured_field": _field_to_dict(structured) if structured else None,
            "semantic_hits": [h.__dict__ for h in hits],
            "found": bool(structured or hits),
        }
        return result

    if name == "check_contradiction":
        existing = session.fields.get(tool_input["field_name"])
        if existing and existing.value and existing.value.strip().lower() != tool_input["new_value"].strip().lower():
            return {"contradiction": True, "existing_value": existing.value, "existing_status": existing.status}
        return {"contradiction": False}

    if name == "ask_clarifying_question":
        session.audit_log.append(AuditLogEntry(
            action="ask_clarifying_question",
            detail={"field_name": tool_input["field_name"], "question": tool_input["question"]},
            reasoning=tool_input["reason"],
        ))
        return {"status": "question_queued_for_patient"}

    if name == "log_field":
        source = SourceRef(
            kind=tool_input["source_kind"],
            ref_id=tool_input["source_ref_id"],
            snippet=tool_input.get("source_snippet", ""),
        )
        session.fields[tool_input["field_name"]] = IntakeField(
            field_name=tool_input["field_name"],
            value=tool_input["value"],
            status=FieldStatus(tool_input["status"]),
            sources=[source],
            confidence_note=tool_input["reasoning"],
        )
        memory.add_chunk(f"field:{tool_input['field_name']}", f"{tool_input['field_name']}: {tool_input['value']}", "message", source.ref_id)
        session.audit_log.append(AuditLogEntry(
            action="log_field",
            detail={"field_name": tool_input["field_name"], "value": tool_input["value"], "status": tool_input["status"]},
            reasoning=tool_input["reasoning"],
        ))
        return {"status": "logged"}

    if name == "flag_contradiction":
        existing_source = SourceRef(
            kind=tool_input["existing_source_kind"],
            ref_id=tool_input["existing_source_ref_id"],
            snippet="",
        )
        field_name = tool_input["field_name"]
        if field_name in session.fields:
            session.fields[field_name].contradiction_with = existing_source
        session.audit_log.append(AuditLogEntry(
            action="flag_contradiction",
            detail={
                "field_name": field_name,
                "new_value": tool_input["new_value"],
                "existing_value": tool_input["existing_value"],
            },
            reasoning="Contradiction with prior record — surfaced instead of overwritten.",
        ))
        return {"status": "flagged_for_review"}

    raise ValueError(f"Unknown tool: {name}")


def _field_to_dict(f: IntakeField) -> dict:
    return {"value": f.value, "status": f.status, "confidence_note": f.confidence_note}
