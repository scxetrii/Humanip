"""
backend/utils/file_parser.py
─────────────────────────────
Secure file upload parser for TXT, DOCX, and PDF formats.

Security measures:
  - Extension whitelist enforced before reading any bytes.
  - File size validated against upload_max_bytes.
  - No execution of uploaded content – pure text extraction.
  - Magic-byte verification for DOCX (ZIP) and PDF.
  - Content sanitised (null bytes removed, length capped).
"""
from __future__ import annotations

import io
import mimetypes
import re
from pathlib import Path
from typing import NamedTuple

from loguru import logger

from config.settings import settings

# ── Magic bytes ───────────────────────────────────────────────────────────────
_MAGIC_PDF  = b"%PDF"
_MAGIC_DOCX = b"PK\x03\x04"   # ZIP – DOCX is a ZIP archive


class ParseResult(NamedTuple):
    text: str
    word_count: int
    char_count: int
    source_format: str


class FileParseError(ValueError):
    """Raised when a file cannot be safely parsed."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_size(data: bytes) -> None:
    if len(data) > settings.upload_max_bytes:
        raise FileParseError(
            f"File too large ({len(data) // 1024} KB). "
            f"Maximum allowed: {settings.upload_max_size_mb} MB."
        )


def _sanitise(text: str) -> str:
    """Remove null bytes and control characters; collapse excessive whitespace."""
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)   # max 3 consecutive blank lines
    return text.strip()


# ── Format parsers ────────────────────────────────────────────────────────────

def _parse_txt(data: bytes) -> str:
    """Detect encoding and decode TXT bytes."""
    import chardet
    detected = chardet.detect(data)
    encoding = detected.get("encoding") or "utf-8"
    try:
        return data.decode(encoding, errors="replace")
    except Exception:
        return data.decode("utf-8", errors="replace")


def _parse_docx(data: bytes) -> str:
    """Extract plain text from a DOCX file (python-docx)."""
    if not data.startswith(_MAGIC_DOCX):
        raise FileParseError("File does not appear to be a valid DOCX (ZIP) file.")

    try:
        import docx
    except ImportError:
        raise FileParseError("python-docx is not installed. Run: pip install python-docx")

    doc = docx.Document(io.BytesIO(data))
    paragraphs: list[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    # Also extract text from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                paragraphs.append(row_text)

    return "\n\n".join(paragraphs)


def _parse_pdf(data: bytes) -> str:
    """Extract plain text from a PDF file (PyMuPDF / fitz)."""
    if not data.startswith(_MAGIC_PDF):
        raise FileParseError("File does not appear to be a valid PDF.")

    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise FileParseError("PyMuPDF is not installed. Run: pip install PyMuPDF")

    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[str] = []

    for page_num in range(min(doc.page_count, 100)):   # cap at 100 pages
        page = doc[page_num]
        page_text = page.get_text("text")
        if page_text.strip():
            pages.append(page_text.strip())

    doc.close()
    return "\n\n".join(pages)


# ── Main entry point ──────────────────────────────────────────────────────────

def parse_uploaded_file(filename: str, data: bytes) -> ParseResult:
    """
    Parse an uploaded file and return its plain-text content.

    Parameters
    ----------
    filename : str
        Original filename with extension (used for type detection).
    data : bytes
        Raw file bytes.

    Returns
    -------
    ParseResult
        Extracted text with metadata.

    Raises
    ------
    FileParseError
        On unsupported format, oversized file, or extraction failure.
    """
    suffix = Path(filename).suffix.lower()

    # ── Whitelist check ──────────────────────────────────────────────────────
    allowed = [ext.lower() for ext in settings.allowed_extensions]
    if suffix not in allowed:
        raise FileParseError(
            f"File type '{suffix}' is not allowed. "
            f"Accepted: {', '.join(allowed)}."
        )

    # ── Size check ───────────────────────────────────────────────────────────
    _validate_size(data)

    logger.info("Parsing uploaded file: {} ({} bytes)", filename, len(data))

    # ── Route to parser ──────────────────────────────────────────────────────
    try:
        if suffix == ".txt":
            raw_text = _parse_txt(data)
            fmt = "TXT"
        elif suffix == ".docx":
            raw_text = _parse_docx(data)
            fmt = "DOCX"
        elif suffix == ".pdf":
            raw_text = _parse_pdf(data)
            fmt = "PDF"
        else:
            raise FileParseError(f"No parser registered for '{suffix}'.")
    except FileParseError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error parsing file '{}'", filename)
        raise FileParseError(f"Could not parse file: {exc}") from exc

    # ── Sanitise ─────────────────────────────────────────────────────────────
    text = _sanitise(raw_text)

    # ── Length guard ─────────────────────────────────────────────────────────
    if len(text) > 50_000:
        logger.warning("File text truncated to 50,000 chars (was {}).", len(text))
        text = text[:50_000] + "\n\n[Document truncated – too long for a single request]"

    word_count = len(text.split())
    char_count = len(text)

    logger.info("File parsed: {} words, {} chars from {}", word_count, char_count, fmt)
    return ParseResult(text=text, word_count=word_count, char_count=char_count, source_format=fmt)


# ── Download helpers ──────────────────────────────────────────────────────────

def text_to_docx_bytes(text: str, title: str = "Humanized Output") -> bytes:
    """Convert plain text back to a DOCX file and return raw bytes."""
    try:
        import docx
    except ImportError:
        raise FileParseError("python-docx is not installed.")

    doc = docx.Document()
    doc.add_heading(title, level=1)

    for para in text.split("\n\n"):
        if para.strip():
            doc.add_paragraph(para.strip())

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()
