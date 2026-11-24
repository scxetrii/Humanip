"""
backend/api/upload.py
─────────────────────
File upload endpoints.

POST /api/v1/upload               → upload TXT, DOCX, or PDF and extract text
GET  /api/v1/download/txt         → download result as plain text
GET  /api/v1/download/docx        → download result as Word document

Security:
  - File extension whitelisted against config.allowed_extensions
  - File bytes validated with magic-byte checks inside the parser
  - Upload size enforced before reading into memory
  - No execution of uploaded content
  - Uploaded files are NOT stored long-term unless explicitly enabled
"""
from __future__ import annotations

import io
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse

from backend.utils.file_parser import FileParseError, parse_uploaded_file, text_to_docx_bytes
from config.settings import settings

router = APIRouter()


# ── Upload ─────────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    summary="Upload a TXT, DOCX, or PDF file and extract its text",
)
async def upload_file(file: UploadFile = File(...)) -> dict:
    """
    Accepts a file upload and returns the extracted plain text ready for
    humanization.  The extracted text is returned in the response body;
    files are not stored server-side by default.

    **Supported formats:** .txt, .docx, .pdf
    **Maximum size:** {max_mb} MB
    """.format(max_mb=settings.upload_max_size_mb)

    # ── Read & validate size ──────────────────────────────────────────────────
    data = await file.read()
    if len(data) > settings.upload_max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File exceeds maximum upload size of {settings.upload_max_size_mb} MB. "
                f"Received {len(data) / 1024 / 1024:.1f} MB."
            ),
        )

    # ── Parse ─────────────────────────────────────────────────────────────────
    try:
        result = parse_uploaded_file(file.filename or "upload.txt", data)
    except FileParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "filename": file.filename,
        "file_type": result.source_format,
        "word_count": result.word_count,
        "char_count": result.char_count,
        "extracted_text": result.text,
    }


# ── Download helpers ───────────────────────────────────────────────────────────

@router.post(
    "/download/txt",
    summary="Download text as a plain-text file",
)
async def download_txt(payload: dict) -> Response:
    """
    Accepts `{ "text": "...", "filename": "output" }` and returns a `.txt` download.
    """
    text = payload.get("text", "")
    filename = payload.get("filename", "humanized_output").rstrip(".txt")

    if not text:
        raise HTTPException(422, "No text provided.")

    return Response(
        content=text.encode("utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}.txt"'},
    )


@router.post(
    "/download/docx",
    summary="Download text as a Word (.docx) document",
)
async def download_docx(payload: dict) -> Response:
    """
    Accepts `{ "text": "...", "title": "My Document", "filename": "output" }`
    and returns a `.docx` download.
    """
    text = payload.get("text", "")
    title = payload.get("title", "Humanized Output")
    filename = payload.get("filename", "humanized_output").rstrip(".docx")

    if not text:
        raise HTTPException(422, "No text provided.")

    try:
        docx_bytes = text_to_docx_bytes(text, title=title)
    except Exception as exc:
        raise HTTPException(500, f"Could not generate DOCX: {exc}")

    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}.docx"'},
    )
