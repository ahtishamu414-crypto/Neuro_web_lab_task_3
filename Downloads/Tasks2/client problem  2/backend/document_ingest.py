"""
Parses uploaded referral letters / lab reports. Tries text extraction first
(cheap, fast); falls back to OCR for scanned pages. If a page still can't be
read, it's marked low_quality rather than blocking intake — the assistant
will only ask about the specific fields that couldn't be recovered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import uuid

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pytesseract
    from pdf2image import convert_from_path
except ImportError:
    pytesseract = None
    convert_from_path = None


@dataclass
class IngestedDocument:
    document_id: str
    filename: str
    chunks: list[str] = field(default_factory=list)
    quality: str = "ok"          # "ok" | "low_quality" | "unreadable"
    pages_ocr_used: list[int] = field(default_factory=list)


def _chunk_text(text: str, max_chars: int = 500) -> list[str]:
    text = " ".join(text.split())
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)] if text else []


def ingest_pdf(path: str, filename: str) -> IngestedDocument:
    doc_id = str(uuid.uuid4())
    doc = IngestedDocument(document_id=doc_id, filename=filename)

    if pdfplumber is None:
        doc.quality = "unreadable"
        return doc

    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if len(text.strip()) < 20:  # likely scanned / image-only page
                    text = _ocr_page(path, page_num)
                    if text:
                        doc.pages_ocr_used.append(page_num)
                    else:
                        doc.quality = "low_quality"
                        continue
                doc.chunks.extend(_chunk_text(text))
    except Exception:
        doc.quality = "unreadable"

    if not doc.chunks:
        doc.quality = "unreadable"
    return doc


def _ocr_page(path: str, page_num: int) -> str:
    if pytesseract is None or convert_from_path is None:
        return ""
    try:
        images = convert_from_path(path, first_page=page_num + 1, last_page=page_num + 1)
        if not images:
            return ""
        return pytesseract.image_to_string(images[0])
    except Exception:
        return ""
