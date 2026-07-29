"""
FastAPI entrypoint. Sessions are kept server-side (in-memory dict here —
swap for Redis/Postgres in production) so state survives page reloads and
lets the assistant recover from a dropped connection without restarting
the conversation.
"""
from __future__ import annotations

import shutil
import tempfile
import uuid

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import traceback
from pydantic import BaseModel

from document_ingest import ingest_pdf
from llm_client import run_turn
from memory import MemoryStore
from models import ConversationTurn, IntakeSession

app = FastAPI(title="MedLink Intake Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "null"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, IntakeSession] = {}
_memories: dict[str, MemoryStore] = {}


class MessageIn(BaseModel):
    session_id: str
    text: str


@app.post("/session")
def create_session(patient_id: str = "demo-patient"):
    session = IntakeSession(patient_id=patient_id)
    _sessions[session.session_id] = session
    _memories[session.session_id] = MemoryStore()
    return {"session_id": session.session_id}


@app.post("/message")
def send_message(payload: MessageIn):
    try:
        if payload.session_id not in _sessions:
            raise HTTPException(status_code=404, detail=f"session {payload.session_id} not found")
        session = _sessions[payload.session_id]
        memory = _memories[payload.session_id]

        turn = ConversationTurn(turn_id=str(uuid.uuid4()), speaker="patient", text=payload.text)
        session.turns.append(turn)
        memory.add_message(session, turn.turn_id, payload.text)

        reply_text = run_turn(session, memory, payload.text)

        reply_turn = ConversationTurn(turn_id=str(uuid.uuid4()), speaker="assistant", text=reply_text)
        session.turns.append(reply_turn)

        return {"reply": reply_text, "urgency": session.urgency}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/{session_id}")
async def upload_document(session_id: str, file: UploadFile):
    if session_id not in _memories:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    memory = _memories[session_id]
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    doc = ingest_pdf(tmp_path, file.filename)
    memory.add_document_chunks(doc.document_id, doc.chunks)
    return {"document_id": doc.document_id, "quality": doc.quality, "chunks_indexed": len(doc.chunks)}


@app.get("/summary/{session_id}")
def get_summary(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")
    session = _sessions[session_id]
    return {
        "fields": {k: v.__dict__ for k, v in session.fields.items()},
        "urgency": session.urgency,
        "audit_log": [e.__dict__ for e in session.audit_log],
    }
