from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import yaml

from src.core.document_analyzer import analyze_chunks_hybrid
from src.core.export_doc import analysis_to_docx
from src.core.ingestion import chunk_text, extract_plaintext
from src.core.llm_client import LLMClient

LANGUAGE_LABELS: Dict[str, str] = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bs": "Bosnian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "hr": "Croatian",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ja-jp": "Japanese",
    "ko": "Korean",
    "ko-kr": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mt": "Maltese",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "zh": "Chinese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
}

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


@st.cache_resource
def _load_settings() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@st.cache_resource
def _build_llm_client(settings: Dict[str, Any]) -> LLMClient | None:
    llm_settings = settings.get("llm", {})
    if not llm_settings.get("enabled"):
        return None
    try:
        return LLMClient(
            provider=llm_settings.get("provider", "openai"),
            model=llm_settings.get("model", "gpt-4o-mini"),
            temperature=float(llm_settings.get("temperature", 0.0)),
            max_retries=int(llm_settings.get("max_retries", 2)),
            timeout=float(llm_settings.get("timeout", 60)),
        )
    except Exception as exc:  # pragma: no cover - UI feedback only
        st.sidebar.error(f"Failed to initialise LLM client: {exc}")
        return None


def _language_options(default: str | None = None) -> List[str]:
    options = [""]
    if default and default not in options:
        options.append(default)
    for code in sorted(LANGUAGE_LABELS):
        if code not in options:
            options.append(code)
    return options


def render() -> None:
    st.title("Document Analyzer")

    if "analysis_json" not in st.session_state:
        st.session_state["analysis_json"] = None
    if "analysis_accepted" not in st.session_state:
        st.session_state["analysis_accepted"] = False
    if "project_name" not in st.session_state:
        st.session_state["project_name"] = ""

    settings = _load_settings()
    client = _build_llm_client(settings)

    st.header("Step 1 – Upload files")
    uploaded_files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=["docx", "pdf", "txt", "xlf", "xliff", "mqxliff"],
    )

    st.header("Step 2 – Project details")
    col1, col2 = st.columns(2)
    with col1:
        src_lang = st.selectbox(
            "Source language",
            options=_language_options(st.session_state.get("source_language")),
            index=0,
            key="source_language",
            format_func=lambda code: LANGUAGE_LABELS.get(code, code.upper()) if code else "Select…",
        )
    with col2:
        target_langs = st.multiselect(
            "Target languages",
            options=_language_options(),
            default=st.session_state.get("target_languages", []),
            key="target_languages",
            format_func=lambda code: LANGUAGE_LABELS.get(code, code.upper()) if code else "",
        )

    project_name = st.text_input(
        "Project name",
        value=st.session_state.get("project_name", ""),
        key="project_name",
    )

    pm_notes = st.text_area(
        "PM / client notes (optional)",
        value=st.session_state.get("pm_notes", ""),
        key="pm_notes",
    )

    st.header("Step 3 – Run analysis")
    run_clicked = st.button("Run analysis")

    if run_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one file before running the analysis.")
        else:
            raw_text = extract_plaintext(uploaded_files)
            if not raw_text.strip():
                st.error("No textual content could be extracted from the uploaded files.")
            else:
                chunks = chunk_text(raw_text)
                try:
                    analysis = analyze_chunks_hybrid(
                        chunks=chunks,
                        src_lang=src_lang,
                        tgt_langs=target_langs,
                        pm_notes=pm_notes,
                        llm_client=client,
                    )
                except Exception as exc:  # pragma: no cover - UI feedback only
                    st.error(f"Document analysis failed: {exc}")
                else:
                    metadata = analysis.setdefault("document_metadata", {})
                    metadata["project_name"] = project_name.strip()
                    metadata["source_language"] = src_lang
                    metadata["target_languages"] = target_langs
                    metadata["file_count"] = len(uploaded_files)
                    metadata["word_estimate"] = metadata.get("word_estimate") or len(
                        raw_text.split()
                    )

                    notes_section = analysis.setdefault("pm_notes", {})
                    if pm_notes:
                        notes_section["original"] = pm_notes
                    notes_section.setdefault("system_notes", "")

                    st.session_state["analysis_json"] = analysis
                    st.session_state["analysis_accepted"] = False
                    st.session_state["project_name"] = project_name.strip()
                    st.success("Analysis complete. Review the proposed result below.")

    analysis = st.session_state.get("analysis_json")
    if analysis:
        st.subheader("Proposed analysis (editable)")
        st.json(analysis)

        if st.button("Accept this analysis"):
            st.session_state["analysis_accepted"] = True
            st.success("Analysis accepted. You can now download the report.")

    if st.session_state.get("analysis_accepted") and st.session_state.get("analysis_json"):
        docx_bytes = analysis_to_docx(st.session_state["analysis_json"])
        st.download_button(
            "Download analysis as Word",
            data=docx_bytes,
            file_name=f"{st.session_state.get('project_name', 'analysis') or 'analysis'}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )


__all__ = ["render"]
