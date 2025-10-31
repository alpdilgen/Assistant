from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from typing import Iterable, List, Tuple

import docx2txt
from langdetect import DetectorFactory, LangDetectException, detect
from PyPDF2 import PdfReader

DetectorFactory.seed = 42


def _safe_seek(buffer) -> None:
    try:
        buffer.seek(0)
    except Exception:  # pragma: no cover - not all streams support seek
        pass


def _extract_lang_attribute(element: ET.Element, attribute: str) -> str | None:
    value = element.attrib.get(attribute)
    if value:
        return value
    for key, attr_value in element.attrib.items():
        if key.lower().endswith(attribute.lower()):
            return attr_value
    xml_lang = element.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
    if xml_lang:
        return xml_lang
    return None


def detect_langs_from_file(uploaded_file) -> Tuple[str | None, str | None]:
    """Attempt to infer source/target languages from uploaded files."""

    name = getattr(uploaded_file, "name", "").lower()
    source_lang: str | None = None
    target_lang: str | None = None

    try:
        if name.endswith((".xlf", ".xliff", ".mqxliff")):
            _safe_seek(uploaded_file)
            try:
                tree = ET.parse(uploaded_file)
                root = tree.getroot()
                source_lang = _extract_lang_attribute(root, "source-language") or _extract_lang_attribute(
                    root, "srclang"
                )
                target_lang = _extract_lang_attribute(root, "target-language") or _extract_lang_attribute(
                    root, "trglang"
                )
                if not source_lang or not target_lang:
                    for element in root.iter():
                        if not source_lang:
                            source_lang = _extract_lang_attribute(element, "source-language") or _extract_lang_attribute(
                                element, "srclang"
                            )
                        if not target_lang:
                            target_lang = _extract_lang_attribute(element, "target-language") or _extract_lang_attribute(
                                element, "trglang"
                            )
                        tag = element.tag.lower()
                        if not source_lang and tag.endswith("source"):
                            source_lang = _extract_lang_attribute(element, "lang")
                        if not target_lang and tag.endswith("target"):
                            target_lang = _extract_lang_attribute(element, "lang")
                        if source_lang and target_lang:
                            break
            except ET.ParseError:
                source_lang = None
                target_lang = None
            finally:
                _safe_seek(uploaded_file)
        elif name.endswith(".pdf"):
            try:
                _safe_seek(uploaded_file)
                reader = PdfReader(uploaded_file)
                pages_text = []
                for page in reader.pages[:2]:
                    text = page.extract_text() or ""
                    pages_text.append(text)
                sample = "\n".join(pages_text)
                if sample.strip():
                    source_lang = detect(sample)
            except Exception:
                source_lang = None
            finally:
                _safe_seek(uploaded_file)
        else:
            _safe_seek(uploaded_file)
            raw = uploaded_file.read(5000)
            text = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
            if text.strip():
                try:
                    source_lang = detect(text)
                except LangDetectException:
                    source_lang = None
            _safe_seek(uploaded_file)
    except Exception:  # pragma: no cover - defensive fallback
        _safe_seek(uploaded_file)
        return None, None

    return source_lang, target_lang


def extract_from_xliff(uploaded_file) -> List[str]:
    _safe_seek(uploaded_file)
    tree = ET.parse(uploaded_file)
    root = tree.getroot()
    texts: List[str] = []
    for element in root.iter():
        tag = element.tag.lower()
        if tag.endswith("source") and element.text:
            texts.append(element.text)
    _safe_seek(uploaded_file)
    return texts


def extract_from_docx(uploaded_file) -> str:
    _safe_seek(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        temp_path = tmp.name
    try:
        text = docx2txt.process(temp_path)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:  # pragma: no cover - cleanup best effort
            pass
    _safe_seek(uploaded_file)
    return text


def _extract_from_pdf(uploaded_file) -> str:
    _safe_seek(uploaded_file)
    reader = PdfReader(uploaded_file)
    texts: List[str] = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    _safe_seek(uploaded_file)
    return "\n".join(texts)


def extract_plaintext(files: Iterable) -> str:
    parts: List[str] = []
    for file_obj in files:
        name = getattr(file_obj, "name", "").lower()
        try:
            if name.endswith((".xlf", ".xliff", ".mqxliff")):
                parts.extend(extract_from_xliff(file_obj))
            elif name.endswith(".docx"):
                parts.append(extract_from_docx(file_obj))
            elif name.endswith(".pdf"):
                try:
                    parts.append(_extract_from_pdf(file_obj))
                except Exception:
                    _safe_seek(file_obj)
            else:
                _safe_seek(file_obj)
                raw = file_obj.read()
                if isinstance(raw, (bytes, bytearray)):
                    text = raw.decode("utf-8", errors="ignore")
                else:
                    text = str(raw)
                parts.append(text)
        finally:
            _safe_seek(file_obj)
    return "\n".join(part for part in parts if part)


def chunk_text(text: str, max_chars: int = 4000, overlap: int = 400) -> List[str]:
    if not text:
        return []
    length = len(text)
    if length <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    step = max(1, max_chars - overlap)
    while start < length:
        end = min(length, start + max_chars)
        chunks.append(text[start:end])
        if end >= length:
            break
        start += step
    return chunks


__all__ = [
    "detect_langs_from_file",
    "extract_from_xliff",
    "extract_from_docx",
    "extract_plaintext",
    "chunk_text",
]
