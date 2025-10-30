"""Reusable Streamlit UI components and helpers."""
from __future__ import annotations

import io
import json
from typing import Any, Dict

import streamlit as st

try:  # pragma: no cover - optional dependency
    from docx import Document
except Exception:  # pragma: no cover
    Document = None  # type: ignore


ACCEPTED_TYPES = ["docx", "pdf", "txt", "xlf", "xliff", "mqxliff"]


def file_uploader() -> list:
    return st.file_uploader(
        "Upload source document(s)",
        type=ACCEPTED_TYPES,
        accept_multiple_files=True,
        help="Upload DOCX, PDF, TXT, or XLIFF files. memoQ bilingual files (.mqxliff) are also supported.",
    )


def render_json(data: Dict[str, Any]) -> None:
    st.json(data, expanded=False)


def download_json_button(label: str, data: Dict[str, Any], file_name: str) -> None:
    json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button(label, data=json_bytes, file_name=file_name, mime="application/json")


def download_docx_button(label: str, data: Dict[str, Any], file_name: str, title: str) -> None:
    if Document is None:
        st.warning("python-docx not installed; download as JSON instead.")
        download_json_button(label, data, file_name.replace(".docx", ".json"))
        return
    buffer = io.BytesIO()
    doc = Document()
    doc.add_heading(title, level=1)
    _write_dict_to_docx(doc, data)
    doc.save(buffer)
    buffer.seek(0)
    st.download_button(label, data=buffer.getvalue(), file_name=file_name, mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


def _write_dict_to_docx(doc, data: Dict[str, Any], level: int = 2) -> None:
    for key, value in data.items():
        heading_level = min(level, 4)
        if isinstance(value, (dict, list)):
            doc.add_heading(str(key).replace("_", " ").title(), level=heading_level)
            if isinstance(value, dict):
                _write_dict_to_docx(doc, value, level=heading_level + 1)
            else:
                for item in value:
                    if isinstance(item, dict):
                        for sub_key, sub_value in item.items():
                            doc.add_paragraph(f"{sub_key}: {sub_value}", style="List Bullet")
                    else:
                        doc.add_paragraph(str(item), style="List Bullet")
        else:
            doc.add_paragraph(f"{key}: {value}")


__all__ = [
    "file_uploader",
    "render_json",
    "download_json_button",
    "download_docx_button",
]
