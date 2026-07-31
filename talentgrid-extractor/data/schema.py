"""
Fixed extraction schema shared by every component of the pipeline
(synthetic data generation, baseline prompting, fine-tuning, evaluation,
and the Streamlit app). Keeping ONE schema definition in one place is what
lets the fine-tune, the baseline, and the eval script all agree on what
"correct" looks like.

Design choices that map directly onto the client's stated failure modes:
- Every leaf field carries a `confidence` and can be `null` -- this is what
  we train the model to use instead of guessing (fights "Schema Drift").
- `evidence_span` is required whenever a field is non-null -- it forces the
  model to point at the substring in the source text that justifies the
  value, which both discourages fabrication and makes hallucinations easy
  to catch automatically during evaluation (no field should cite evidence
  text that doesn't actually appear in the source).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional

EXTRACTION_SCHEMA_JSON = {
    "type": "object",
    "properties": {
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "years_experience": {"type": ["number", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_span": {"type": ["string", "null"]},
                },
                "required": ["name", "years_experience", "confidence", "evidence_span"],
            },
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "degree": {"type": ["string", "null"]},
                    "institution": {"type": ["string", "null"]},
                    "graduation_year": {"type": ["number", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_span": {"type": ["string", "null"]},
                },
                "required": ["degree", "institution", "graduation_year", "confidence", "evidence_span"],
            },
        },
        "certifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "issuer": {"type": ["string", "null"]},
                    "year": {"type": ["number", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_span": {"type": ["string", "null"]},
                },
                "required": ["name", "issuer", "year", "confidence", "evidence_span"],
            },
        },
        "career_gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_year": {"type": ["number", "null"]},
                    "end_year": {"type": ["number", "null"]},
                    "duration_months": {"type": ["number", "null"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence_span": {"type": ["string", "null"]},
                },
                "required": ["start_year", "end_year", "duration_months", "confidence", "evidence_span"],
            },
        },
    },
    "required": ["skills", "education", "certifications", "career_gaps"],
}

SYSTEM_PROMPT = """You are a structured-data extraction engine for resumes and cover letters.
Extract ONLY the following JSON schema. Rules you MUST follow:
1. Every field you fill in must be directly supported by text in the source document. Copy the
   supporting text into `evidence_span`.
2. If you are not confident about a value, or the document does not mention it, set the value to
   null and set `confidence` low (<= 0.4). Do NOT invent plausible-sounding values (e.g. do not
   guess a graduation year, a skill's years of experience, or a certification issuer).
3. Only include a skill/education/certification/gap entry at all if there is at least some
   textual evidence for it existing -- do not pad the list to look complete.
4. Output valid JSON only, matching this schema exactly, with no prose before or after it:

SCHEMA:
{schema}
""".format(schema=EXTRACTION_SCHEMA_JSON)


@dataclass
class Skill:
    name: str
    years_experience: Optional[float]
    confidence: float
    evidence_span: Optional[str]


@dataclass
class Education:
    degree: Optional[str]
    institution: Optional[str]
    graduation_year: Optional[int]
    confidence: float
    evidence_span: Optional[str]


@dataclass
class Certification:
    name: str
    issuer: Optional[str]
    year: Optional[int]
    confidence: float
    evidence_span: Optional[str]


@dataclass
class CareerGap:
    start_year: Optional[int]
    end_year: Optional[int]
    duration_months: Optional[float]
    confidence: float
    evidence_span: Optional[str]


@dataclass
class CandidateProfile:
    skills: list
    education: list
    certifications: list
    career_gaps: list

    def to_dict(self):
        return asdict(self)
