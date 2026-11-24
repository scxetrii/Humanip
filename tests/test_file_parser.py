"""
tests/test_file_parser.py
──────────────────────────
Unit tests for secure file upload parsing.

Run with:  pytest tests/test_file_parser.py -v
"""
from __future__ import annotations

import io
import pytest

from backend.utils.file_parser import (
    FileParseError,
    ParseResult,
    parse_uploaded_file,
    text_to_docx_bytes,
)


# ── TXT Parsing ───────────────────────────────────────────────────────────────

class TestTxtParsing:
    def test_simple_ascii_txt(self):
        content = b"Hello world. This is a test document."
        result = parse_uploaded_file("test.txt", content)
        assert isinstance(result, ParseResult)
        assert "Hello world" in result.text

    def test_utf8_txt(self):
        content = "Résumé with accented characters: café, naïve.".encode("utf-8")
        result = parse_uploaded_file("resume.txt", content)
        assert "Résumé" in result.text or "caf" in result.text  # decoded

    def test_word_count_calculated(self):
        content = b"One two three four five."
        result = parse_uploaded_file("count.txt", content)
        assert result.word_count >= 5

    def test_char_count_positive(self):
        content = b"Some text."
        result = parse_uploaded_file("char.txt", content)
        assert result.char_count > 0

    def test_format_reported_as_txt(self):
        result = parse_uploaded_file("test.txt", b"Hello world.")
        assert result.source_format == "TXT"


# ── Extension Whitelist ───────────────────────────────────────────────────────

class TestExtensionWhitelist:
    @pytest.mark.parametrize("bad_ext", [
        "script.py", "app.exe", "image.jpg", "archive.zip",
        "data.csv", "config.env", "page.html",
    ])
    def test_disallowed_extensions_raise(self, bad_ext):
        with pytest.raises(FileParseError, match="not allowed"):
            parse_uploaded_file(bad_ext, b"content")

    @pytest.mark.parametrize("good_ext", ["file.txt", "document.docx", "report.pdf"])
    def test_allowed_extensions_dont_raise_on_ext_check(self, good_ext):
        # May raise for other reasons (bad magic bytes) but not extension check
        try:
            parse_uploaded_file(good_ext, b"minimal content")
        except FileParseError as e:
            assert "not allowed" not in str(e).lower()


# ── File Size Limit ───────────────────────────────────────────────────────────

class TestFileSizeLimit:
    def test_oversized_file_raises(self):
        # 11 MB — above the 10 MB default limit
        oversized = b"A" * (11 * 1024 * 1024)
        with pytest.raises(FileParseError, match="too large|exceeds"):
            parse_uploaded_file("large.txt", oversized)

    def test_just_under_limit_does_not_raise_on_size(self):
        # 9 MB — under limit
        ok_size = b"A" * (9 * 1024 * 1024)
        try:
            parse_uploaded_file("ok.txt", ok_size)
        except FileParseError as e:
            assert "too large" not in str(e).lower()


# ── Text Sanitisation ─────────────────────────────────────────────────────────

class TestSanitisation:
    def test_null_bytes_removed(self):
        content = b"Hello\x00World"
        result = parse_uploaded_file("null.txt", content)
        assert "\x00" not in result.text

    def test_control_characters_removed(self):
        content = b"Hello\x01\x02\x03World"
        result = parse_uploaded_file("ctrl.txt", content)
        assert "\x01" not in result.text

    def test_excessive_blank_lines_collapsed(self):
        content = ("Paragraph one.\n\n\n\n\n\n\nParagraph two.").encode()
        result = parse_uploaded_file("blanks.txt", content)
        assert "\n\n\n\n" not in result.text


# ── DOCX Magic-Byte Check ─────────────────────────────────────────────────────

class TestDocxMagicBytes:
    def test_fake_docx_raises(self):
        """A .docx file that doesn't start with PK magic bytes should fail."""
        fake_content = b"This is not a ZIP/DOCX file at all."
        with pytest.raises(FileParseError):
            parse_uploaded_file("fake.docx", fake_content)


# ── PDF Magic-Byte Check ──────────────────────────────────────────────────────

class TestPdfMagicBytes:
    def test_fake_pdf_raises(self):
        """A .pdf file that doesn't start with %PDF should fail."""
        fake_content = b"This is not a PDF file."
        with pytest.raises(FileParseError):
            parse_uploaded_file("fake.pdf", fake_content)


# ── DOCX Generation (Download) ────────────────────────────────────────────────

class TestDocxGeneration:
    def test_generates_bytes(self):
        docx_bytes = text_to_docx_bytes("Hello world. This is a paragraph.")
        assert isinstance(docx_bytes, bytes)
        assert len(docx_bytes) > 0

    def test_output_is_valid_zip_magic(self):
        """DOCX files are ZIP archives and start with PK magic bytes."""
        docx_bytes = text_to_docx_bytes("Sample text.")
        assert docx_bytes[:4] == b"PK\x03\x04"

    def test_multiline_text_no_crash(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        docx_bytes = text_to_docx_bytes(text, title="Test Doc")
        assert len(docx_bytes) > 100

    def test_empty_text_no_crash(self):
        docx_bytes = text_to_docx_bytes("")
        assert isinstance(docx_bytes, bytes)
