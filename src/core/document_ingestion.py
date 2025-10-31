"""Utilities for ingesting and normalising translation project documents."""
from __future__ import annotations

from dataclasses import dataclass
import io
import logging
import os
import re
import tempfile
from typing import Optional, Sequence, TYPE_CHECKING
from xml.etree import ElementTree as ET

from langdetect import detect

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from streamlit.runtime.uploaded_file_manager import UploadedFile

from pydantic import BaseModel, Field

try:
    from docx import Document as DocxDocument  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    DocxDocument = None  # type: ignore

try:
    from PyPDF2 import PdfReader  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    PdfReader = None  # type: ignore

try:  # pragma: no cover - optional dependency guard
    import docx2txt  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    docx2txt = None  # type: ignore

try:
    from lxml import etree  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    etree = None  # type: ignore

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


def detect_langs_from_file(file_obj: io.BufferedIOBase) -> tuple[Optional[str], Optional[str]]:
    """Detect source and target languages from bilingual or monolingual files."""

    name = getattr(file_obj, "name", "").lower()
    try:
        if name.endswith((".xlf", ".xliff", ".mqxliff")):
            file_obj.seek(0)
            tree = ET.parse(file_obj)
            root = tree.getroot()
            src_lang = root.attrib.get("source-language") or root.attrib.get("srcLang")
            tgt_lang = root.attrib.get("target-language") or root.attrib.get("trgLang")
            file_obj.seek(0)
            return src_lang, tgt_lang

        file_obj.seek(0)
        sample = file_obj.read(5000)
        if isinstance(sample, bytes):
            sample_text = sample.decode(errors="ignore")
        else:
            sample_text = str(sample)
        file_obj.seek(0)
        if sample_text.strip():
            return detect(sample_text), None
        return None, None
    except Exception:  # pragma: no cover - heuristic fallbacks only
        file_obj.seek(0)
        return None, None


def extract_text_chunks(
    files: Sequence[io.BufferedIOBase],
    *,
    max_chars: int = 4000,
    overlap: int = 400,
) -> list[str]:
    """Read uploaded files and split their text into overlapping chunks."""

    texts: list[str] = []
    for file_obj in files:
        name = getattr(file_obj, "name", "").lower()
        file_obj.seek(0)
        if name.endswith((".xlf", ".xliff", ".mqxliff")):
            try:
                tree = ET.parse(file_obj)
                sources = [node.text for node in tree.findall(".//{*}source") if node.text]
                texts.extend(sources)
            finally:
                file_obj.seek(0)
        elif name.endswith(".docx"):
            if docx2txt is None:  # pragma: no cover - dependency guard
                raise RuntimeError("docx2txt is required to process DOCX files")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(file_obj.read())
                tmp_path = tmp.name
            try:
                extracted = docx2txt.process(tmp_path) or ""
                texts.append(extracted)
            finally:
                file_obj.seek(0)
                try:
                    os.unlink(tmp_path)
                except OSError:  # pragma: no cover - best effort cleanup
                    pass
        else:
            raw_bytes = file_obj.read()
            if isinstance(raw_bytes, bytes):
                text = raw_bytes.decode(errors="ignore")
            else:
                text = str(raw_bytes)
            texts.append(text)
            file_obj.seek(0)

    joined = "\n".join(part for part in texts if part)
    if not joined:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(joined):
        end = min(start + max_chars, len(joined))
        chunks.append(joined[start:end])
        if end == len(joined):
            break
        start += max(max_chars - overlap, 1)
    return chunks


def load_documents(
    uploads: Sequence["UploadedFile"],
    *,
    default_source_lang: Optional[str] = None,
    default_target_lang: Optional[str] = None,
) -> list[DocumentPayload]:
    """Load multiple uploaded files into :class:`DocumentPayload` objects."""

    payloads: list[DocumentPayload] = []
    for uploaded_file in uploads:
        if uploaded_file is None:
            continue
        filename = getattr(uploaded_file, "name", "uploaded_file")
        file_bytes = uploaded_file.getvalue()
        if not file_bytes:
            logger.warning("Skipping empty upload: %s", filename)
            continue
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
    "detect_langs_from_file",
    "extract_text_chunks",
]
