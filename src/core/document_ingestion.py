"""Utilities for ingesting and normalising translation project documents."""
from __future__ import annotations

from dataclasses import dataclass
import io
import logging
import re
from typing import Iterable, Optional

from pydantic import BaseModel, Field

try:
    from docx import Document as DocxDocument  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    DocxDocument = None  # type: ignore

try:
    from lxml import etree  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    etree = None  # type: ignore

try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    PdfReader = None  # type: ignore

logger = logging.getLogger(__name__)


class DocumentPayload(BaseModel):
    """Normalized representation of an uploaded document."""

    filename: str
    source_language: Optional[str] = Field(default=None)
    target_language: Optional[str] = Field(default=None)
    raw_text: str
    segments: Optional[list[dict]] = Field(default=None, description="Bilingual segments if available")


SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".txt", ".xlf", ".xliff", ".mqxliff"}
INLINE_TAG_PATTERN = re.compile(r"<(/?)(g|x|bx|ex|ph)[^>]*>", re.IGNORECASE)


@dataclass
class XliffContent:
    raw_text: str
    segments: list[dict]
    source_language: Optional[str]
    target_language: Optional[str]


def load_documents(
    uploads: Iterable[tuple[str, bytes]],
    *,
    default_source_lang: Optional[str] = None,
    default_target_lang: Optional[str] = None,
) -> list[DocumentPayload]:
    """Load multiple uploaded files into :class:`DocumentPayload` objects."""

    payloads: list[DocumentPayload] = []
    for filename, file_bytes in uploads:
        payload = load_document(
            filename,
            file_bytes,
            default_source_lang=default_source_lang,
            default_target_lang=default_target_lang,
        )
        if payload:
            payloads.append(payload)
    return payloads


def load_document(
    filename: str,
    file_bytes: bytes,
    *,
    default_source_lang: Optional[str] = None,
    default_target_lang: Optional[str] = None,
) -> Optional[DocumentPayload]:
    """Load a single document based on its extension."""

    extension = filename.lower().rpartition(".")[2]
    extension = f".{extension}" if extension else ""

    if extension not in SUPPORTED_EXTENSIONS:
        logger.warning("Unsupported file extension for %s", filename)
        return None

    if extension in {".xlf", ".xliff", ".mqxliff"}:
        xliff_content = _parse_xliff(file_bytes)
        if xliff_content is None:
            return None
        return DocumentPayload(
            filename=filename,
            source_language=xliff_content.source_language or default_source_lang,
            target_language=xliff_content.target_language or default_target_lang,
            raw_text=xliff_content.raw_text,
            segments=xliff_content.segments,
        )

    if extension == ".docx":
        text = _read_docx(file_bytes)
    elif extension == ".pdf":
        text = _read_pdf(file_bytes)
    else:  # .txt or other plaintext derivatives
        text = file_bytes.decode("utf-8", errors="ignore")

    return DocumentPayload(
        filename=filename,
        source_language=default_source_lang,
        target_language=default_target_lang,
        raw_text=text,
        segments=None,
    )


def _read_docx(file_bytes: bytes) -> str:
    if not DocxDocument:
        raise RuntimeError("python-docx is required to process DOCX files")
    with io.BytesIO(file_bytes) as buffer:
        document = DocxDocument(buffer)
    paragraphs = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    return "\n".join(paragraphs)


def _read_pdf(file_bytes: bytes) -> str:
    if not PdfReader:
        raise RuntimeError("PyPDF2 is required to process PDF files")
    with io.BytesIO(file_bytes) as buffer:
        reader = PdfReader(buffer)
        texts = []
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
            except Exception:  # pragma: no cover - depends on PyPDF2 internals
                page_text = ""
            texts.append(page_text.strip())
    return "\n".join(filter(None, texts))


def _parse_xliff(file_bytes: bytes) -> Optional[XliffContent]:
    if not etree:
        logger.error("lxml is required to parse XLIFF documents")
        return None

    try:
        root = etree.fromstring(file_bytes)
    except etree.XMLSyntaxError as exc:  # type: ignore[attr-defined]
        logger.error("Failed to parse XLIFF: %s", exc)
        return None

    nsmap = root.nsmap.copy()
    nsmap.setdefault(None, root.nsmap.get(None, "urn:oasis:names:tc:xliff:document:1.2"))

    source_language = root.get("source-language") or root.get("srcLang")
    target_language = root.get("target-language") or root.get("trgLang")

    segments: list[dict] = []
    text_fragments: list[str] = []

    trans_units = root.xpath("//ns:trans-unit", namespaces={"ns": nsmap.get(None)})
    if not trans_units:  # Try memoQ specific namespace if default fails
        ns = nsmap.get(None) or "urn:oasis:names:tc:xliff:document:1.2"
        trans_units = root.xpath("//ns:trans-unit", namespaces={"ns": ns})

    for unit in trans_units:
        source_element = unit.find(".//{*}source")
        target_element = unit.find(".//{*}target")
        source_text = _normalise_inline_tags(_element_text(source_element))
        target_text = _normalise_inline_tags(_element_text(target_element))

        text_fragments.append(source_text)
        segments.append(
            {
                "id": unit.get("id"),
                "source": source_text,
                "target": target_text,
            }
        )

    raw_text = "\n".join(filter(None, text_fragments))
    return XliffContent(
        raw_text=raw_text,
        segments=segments,
        source_language=source_language,
        target_language=target_language,
    )


def _normalise_inline_tags(value: Optional[str]) -> str:
    if not value:
        return ""
    value = INLINE_TAG_PATTERN.sub(lambda m: " {" + m.group(2).upper() + ("_END" if m.group(1) else "") + "} ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _element_text(element) -> str:
    if element is None:
        return ""
    text_parts = [element.text or ""]
    for child in element:
        text_parts.append(etree.tostring(child, encoding="unicode", method="text"))
    return "".join(text_parts)

__all__ = [
    "DocumentPayload",
    "SUPPORTED_EXTENSIONS",
    "load_document",
    "load_documents",
]
