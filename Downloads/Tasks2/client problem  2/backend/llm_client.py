"""
Wraps the Anthropic Messages API tool-use loop.

Context/cost control: we never resend the full turn-by-turn transcript.
Each call sends only (a) the system prompt, (b) the running structured
profile (compact), and (c) the last few raw turns. Older detail is only
pulled back in via search_known_context, on demand.
"""
from __future__ import annotations

import json
import os

import anthropic

from models import IntakeSession
from tools import TOOL_SCHEMAS, dispatch_tool
from memory import MemoryStore

# Swap this for whichever current model string your Anthropic account has
# access to — check docs.claude.com for the latest.
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You are a clinical intake assistant for MedLink Health Partners.

Rules you must follow on every turn:
1. Never fabricate or assume a medical detail the patient has not given you.
2. Before asking the patient anything, call search_known_context first. Only
   ask a clarifying question if that search genuinely comes back empty or
   insufficient.
3. If search_known_context returns an existing value for a field the patient
   is now describing differently, call check_contradiction before logging
   anything. If there's a real contradiction, call flag_contradiction instead
   of silently logging the new value.
4. Every fact you log must go through log_field with an honest status:
   patient_stated only if the patient said it directly, inferred if you're
   reading between the lines (and in that case you should usually confirm
   with the patient rather than logging it outright), unresolved if it's
   genuinely unknown.
5. Assign or update triage urgency (routine / soon / urgent / emergency) as
   soon as you have enough information, and clearly call out any red-flag
   symptom (chest pain, difficulty breathing, stroke symptoms, severe
   bleeding, suicidal ideation, etc.) as urgent or emergency.
6. Keep responses to the patient short, warm, and focused on one question
   at a time.
"""


def run_turn(session: IntakeSession, memory: MemoryStore, patient_message: str, client: anthropic.Anthropic | None = None) -> str:
    """Runs one patient turn through the tool-use loop and returns the
    assistant's reply text. Mutates `session` in place (fields, audit log)."""
    client = client or anthropic.Anthropic()

    messages = _build_messages(session, patient_message)

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if not tool_calls:
            # plain text reply — conversation turn is done
            text_blocks = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_blocks)

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for call in tool_calls:
            result = dispatch_tool(call.name, call.input, session, memory)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": json.dumps(result),
            })
        messages.append({"role": "user", "content": tool_results})


def _build_messages(session: IntakeSession, patient_message: str) -> list[dict]:
    """Compact context: structured profile summary + last few raw turns +
    the new message. Not the full transcript."""
    profile_summary = {
        name: {"value": f.value, "status": f.status}
        for name, f in session.fields.items()
    }
    recent_turns = session.turns[-6:]
    recent_text = "\n".join(f"{t.speaker}: {t.text}" for t in recent_turns)

    context_block = (
        f"Known structured profile so far:\n{json.dumps(profile_summary, indent=2)}\n\n"
        f"Recent conversation:\n{recent_text}\n\n"
        f"New patient message: {patient_message}"
    )
    return [{"role": "user", "content": context_block}]
