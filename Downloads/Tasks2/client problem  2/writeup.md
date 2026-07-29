# MedLink Autonomous Patient Intake & Clinical Triage Assistant
### Technical Write-Up — Internship Task 2

---

## 1. Problem summary

MedLink's previous intake bot was pulled from production after two near-miss
incidents caused by **silent guessing** on ambiguous symptoms. A root-cause
review surfaced three further failure modes: **context amnesia** (re-asking
things already known), **contradiction blindness** (overwriting conflicting
records instead of surfacing them), and **no confidence signal** (every
extracted field looked equally certain to the reviewing physician).

This system is designed so that each of those four failure modes is closed
off structurally — not just prompted against — by making "check memory
first," "log with a status," and "never overwrite silently" the only paths
the assistant is allowed to take.

## 2. System architecture

```
Frontend (chat + upload)
        │
        ▼
Orchestrator (conversation controller, tool dispatch)
   │         │              │
   ▼         ▼              ▼
LLM+tools  Memory store   Document ingestion
(extract,  (structured    (PDF text extraction
 verify)   profile +      + OCR fallback)
           embeddings)
   │         │              │
   └─────────┴──────────────┘
              ▼
   Structured intake store + audit log
   (confidence-tagged, traceable fields)
```

The orchestrator never lets the model write directly to the patient record.
Every fact the model wants to log has to go through a tool call
(`log_field`) that requires a `status` and a `source` — there is no code
path where a value reaches the intake summary without both.

## 3. Turn-by-turn decision logic

For every new patient statement, the assistant follows a fixed order of
operations, enforced by the system prompt and the tool schemas themselves
(the model is only given `ask_clarifying_question` and `flag_contradiction`
as options *after* the earlier checks have run):

1. **Search known context** (`search_known_context`) — check the structured
   profile (exact match, cheap) and a semantic search over the transcript,
   uploaded documents, and prior-visit records (broader, catches paraphrases).
2. **If found** → log the field directly as `patient_stated` or leave it as
   already-logged; no question asked.
3. **If not found** → check for contradiction (`check_contradiction`)
   against any existing value for that field.
   - **Contradiction found** → `flag_contradiction`: both values are kept,
     the conflict is surfaced to the patient for confirmation and to the
     physician in the summary. The record is never silently overwritten.
   - **No contradiction, just incomplete** → `ask_clarifying_question`,
     but only for the specific missing piece — never a broad re-ask.
4. **Log the field** with an honest status: `patient_stated` if the patient
   said it directly, `inferred` if the model is reading between the lines
   (which should usually still trigger a confirming question rather than
   being logged outright), or `unresolved` if it's genuinely unknown.
5. **Audit entry** is written for every action above — question asked,
   contradiction flagged, or field logged — with the reasoning attached.

This directly maps onto the failure modes:

| Failure mode (original bot) | How this design closes it |
|---|---|
| Silent guessing | `log_field` requires an honest `status`; `inferred` values are visually flagged, not presented as fact |
| Context amnesia | `search_known_context` is mandatory before any question |
| Contradiction blindness | `check_contradiction` + `flag_contradiction` keep both values visible instead of overwriting |
| No confidence signal | Every field in the summary carries `status` + source, color-coded in the UI |

## 4. Data model & audit trail

Each `IntakeField` records:

```python
field_name: str
value: str | None
status: "patient_stated" | "inferred" | "unresolved"
sources: list[SourceRef]        # kind, ref_id, verbatim snippet
contradiction_with: SourceRef | None
confidence_note: str
```

Every `AuditLogEntry` records the action taken (`log_field`,
`ask_clarifying_question`, `flag_contradiction`), the detail, the reasoning,
and a timestamp. Because `sources` always points back to a specific turn ID,
document ID, or prior-visit ID with a verbatim snippet, any field in the
final summary can be traced back to the exact patient statement or document
line that produced it — satisfying the auditability requirement without
needing a separate reconciliation step.

## 5. Cost control on long conversations

The model is never sent the full 100+ turn transcript. Each API call
receives only:
- the compact structured profile (already-logged fields, a few lines of
  JSON), and
- the last ~6 raw turns, and
- the new patient message.

Anything older is retrieved on demand through `search_known_context`'s
semantic search rather than replayed by default. This keeps token usage
roughly constant per turn regardless of how long the conversation has been
running, rather than growing linearly (or worse) with conversation length.

## 6. Document ingestion & graceful degradation

Uploaded PDFs are processed page-by-page:
1. Attempt direct text extraction (`pdfplumber`) — fast, cheap, works for
   any text-based referral letter or lab report.
2. If a page yields near-empty text (a strong signal it's a scanned image),
   fall back to OCR (`pytesseract` + `pdf2image`).
3. If OCR also fails or the page is unreadable, that page is marked
   `low_quality`/`unreadable` rather than aborting the whole document. The
   conversation continues, and the assistant only asks the patient about the
   specific fields that couldn't be recovered from the document — it never
   stalls waiting on a perfect parse.

## 7. Error recovery

Conversation state lives server-side, keyed by session ID, not in the
browser. If a document misreads or a turn's extraction fails, the
orchestrator can re-run just that step against the stored raw input (the
original message text or document text) rather than needing to replay or
restart the whole conversation — the existing structured profile and audit
log are untouched by a single failed step.

## 8. Triage & escalation

Urgency (`routine` / `soon` / `urgent` / `emergency`) is assigned and
updated by the model as soon as sufficient information is available, per
the system prompt's explicit instruction to flag red-flag symptoms (chest
pain, difficulty breathing, stroke symptoms, severe bleeding, suicidal
ideation) as `urgent` or `emergency`. This is surfaced prominently in the UI
header so a coordinator scanning many conversations can triage at a glance.

## 9. Known limitations & future work

- **Semantic search** currently uses TF-IDF for simplicity and to avoid
  external embedding-API dependencies in the demo; a production deployment
  should swap in a proper embedding model + vector database (e.g. pgvector)
  for better recall on paraphrased patient language.
- **Session storage** is in-memory for the demo; production would use
  Postgres/Redis so state survives a server restart.
- **Urgency scoring** is currently model-judged rather than backed by a
  validated clinical triage protocol (e.g. Manchester Triage Scale) — a real
  deployment should cross-check the model's urgency assignment against a
  rules-based red-flag checklist before it reaches a physician.
- **Multi-session continuity** (a patient returning after 100+ turns across
  multiple sessions) is supported by the memory store's `prior_visit`
  ingestion path, but the demo doesn't yet include a scheduled job to pull
  new prior-visit records from MedLink's EHR automatically.

## 10. Deliverable mapping

| Deliverable | Location |
|---|---|
| Working demo | `frontend/index.html` (standalone or live against `backend/`) |
| Source code | `backend/` — `models.py`, `memory.py`, `tools.py`, `llm_client.py`, `document_ingest.py`, `app.py` |
| Architecture diagram | Section 2 above / rendered diagram provided separately |
| Technical write-up | This document |
| Sample transcripts | See `transcripts/` — three conversations demonstrating ambiguity, contradiction, and memory-resolved (no re-ask) scenarios |
