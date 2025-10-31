from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

import streamlit as st
import yaml

from src.core.document_analyzer import analyze_chunks_hybrid
from src.core.export_doc import analysis_to_docx
from src.core.ingestion import chunk_text, detect_langs_from_file, extract_plaintext
from src.core.llm_client import LLMClient

LANGUAGE_LABELS: Dict[str, str] = {
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
    "ko": "Korean",
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
    "ar": "Arabic",
    "zh": "Chinese",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja-jp": "Japanese",
    "ko-kr": "Korean",
}

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"

if TYPE_CHECKING:  # pragma: no cover - typing only
    from streamlit.runtime.uploaded_file_manager import UploadedFile


def format_lang(code: str | None) -> str:
    if not code:
        return "— select language —"
    cleaned = code.strip()
    if not cleaned:
        return "— select language —"
    key = cleaned.lower()
    return LANGUAGE_LABELS.get(key, cleaned.upper())


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


def _ensure_state() -> None:
    defaults = {
        "analyzer_files": [],
        "analyzer_file_langs": {},
        "analyzer_chunks": [],
        "analyzer_analysis": None,
        "analyzer_analysis_json": "",
        "analyzer_analysis_accepted": False,
        "analyzer_source_lang": "",
        "analyzer_target_langs": [],
        "analyzer_project_name": "",
        "analyzer_pm_enabled": False,
        "analyzer_pm_notes": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _persist_uploads(uploads: Iterable["UploadedFile"]) -> List[dict]:
    stored: List[dict] = []
    for uploaded in uploads or []:
        if uploaded is None:
            continue
        stored.append({"name": uploaded.name, "data": uploaded.getvalue(), "type": uploaded.type})
    return stored


def _files_from_storage(stored_files: Iterable[dict]) -> List[io.BytesIO]:
    buffers: List[io.BytesIO] = []
    for file_info in stored_files:
        data = file_info.get("data", b"")
        buffer = io.BytesIO(data)
        buffer.name = file_info.get("name", "uploaded_file")
        buffers.append(buffer)
    return buffers


def _language_options(*candidates: str | None) -> List[str]:
    options: List[str] = [""]
    seen: set[str] = {""}
    for candidate in candidates:
        if not candidate:
            continue
        normalised = candidate.strip().lower()
        if not normalised or normalised in seen:
            continue
        options.append(normalised)
        seen.add(normalised)
    for code in LANGUAGE_LABELS:
        if code not in seen:
            options.append(code)
            seen.add(code)
    return options


def _display_language_row(file_name: str, detection: tuple[str | None, str | None]) -> None:
    src_guess, tgt_guess = detection
    language_state = st.session_state.setdefault("analyzer_file_langs", {})
    file_state = language_state.get(file_name, {})

    src_key = f"analyzer_src_{file_name}"
    tgt_key = f"analyzer_tgt_{file_name}"

    default_src = file_state.get("source") or src_guess or ""
    default_tgt = file_state.get("target") or tgt_guess or ""

    options_src = _language_options(default_src, src_guess)
    options_tgt = _language_options(default_tgt, tgt_guess)

    if src_key not in st.session_state:
        st.session_state[src_key] = default_src
    if tgt_key not in st.session_state:
        st.session_state[tgt_key] = default_tgt

    cols = st.columns(2)
    with cols[0]:
        selected_src = st.selectbox(
            "Source language",
            options=options_src,
            format_func=format_lang,
            key=src_key,
        )
    with cols[1]:
        selected_tgt = st.selectbox(
            "Target language",
            options=options_tgt,
            format_func=format_lang,
            key=tgt_key,
        )

    language_state[file_name] = {"source": selected_src, "target": selected_tgt}

    if not src_guess and not selected_src:
        st.warning("Language detection failed for this file. Please select the source language manually.")


def _collect_selected_languages() -> tuple[str, List[str]]:
    language_state = st.session_state.get("analyzer_file_langs", {})
    source_candidates: Counter[str] = Counter()
    target_candidates: Counter[str] = Counter()
    for info in language_state.values():
        src = (info.get("source") or "").strip().lower()
        tgt = (info.get("target") or "").strip().lower()
        if src:
            source_candidates[src] += 1
        if tgt:
            target_candidates[tgt] += 1

    dominant_source = ""
    if source_candidates:
        dominant_source = source_candidates.most_common(1)[0][0]

    stored_source = st.session_state.get("analyzer_source_lang", "")
    if not stored_source and dominant_source:
        st.session_state["analyzer_source_lang"] = dominant_source

    source_options = _language_options(st.session_state.get("analyzer_source_lang"), dominant_source)

    source_lang = st.selectbox(
        "Primary source language",
        options=source_options,
        format_func=format_lang,
        key="analyzer_source_lang",
    )

    detected_targets = [code for code, _ in target_candidates.most_common()]
    defaults = st.session_state.get("analyzer_target_langs", [])
    if not defaults and detected_targets:
        defaults = detected_targets
        st.session_state["analyzer_target_langs"] = defaults

    target_options = _language_options(*(defaults + detected_targets))
    target_langs = st.multiselect(
        "Target languages",
        options=target_options,
        default=defaults,
        format_func=format_lang,
        key="analyzer_target_langs",
    )

    return source_lang, [lang for lang in target_langs if lang]


def render() -> None:
    _ensure_state()

    st.title("Document Analyzer")
    st.caption("Upload your project files, confirm languages, and generate a PM-ready analysis.")

    st.header("Step 1 – Upload files")
    uploaded_files = st.file_uploader(
        "Upload documents",
        accept_multiple_files=True,
        type=["docx", "pdf", "txt", "xlf", "xliff", "mqxliff"],
        key="analyzer_uploader",
    )

    if uploaded_files:
        st.session_state["analyzer_files"] = _persist_uploads(uploaded_files)
        st.session_state["analyzer_analysis"] = None
        st.session_state["analyzer_analysis_json"] = ""
        st.session_state["analyzer_analysis_accepted"] = False

    stored_files = st.session_state.get("analyzer_files", [])
    if not stored_files:
        st.info("Upload DOCX, PDF, TXT or XLIFF files to start the analysis.")
        return

    st.header("Step 2 – Detect languages")
    for file_info in stored_files:
        buffer = io.BytesIO(file_info.get("data", b""))
        buffer.name = file_info.get("name", "uploaded_file")
        detection = detect_langs_from_file(buffer)
        buffer.close()
        with st.container():
            st.subheader(file_info.get("name", "uploaded_file"))
            _display_language_row(file_info.get("name", "uploaded_file"), detection)

    source_lang, target_langs = _collect_selected_languages()

    st.header("Step 3 – Project name (mandatory)")
    project_name = st.text_input(
        "Project name (required)",
        value=st.session_state.get("analyzer_project_name", ""),
        key="analyzer_project_name",
    )
    if not project_name.strip():
        st.warning("Project name is required to continue.")
        st.stop()

    st.header("Step 4 – PM info (optional)")
    pm_choice = st.radio(
        "Do you have client/PM instructions?",
        options=["No", "Yes"],
        index=1 if st.session_state.get("analyzer_pm_enabled") else 0,
        horizontal=True,
        key="analyzer_pm_choice",
    )
    pm_notes = ""
    if pm_choice == "Yes":
        st.session_state["analyzer_pm_enabled"] = True
        pm_notes = st.text_area(
            "Paste PM or client instructions",
            value=st.session_state.get("analyzer_pm_notes", ""),
            key="analyzer_pm_notes",
            height=150,
        )
    else:
        st.session_state["analyzer_pm_enabled"] = False
        st.session_state["analyzer_pm_notes"] = ""

    st.header("Step 5 – Full-text extraction & chunking")
    files_for_processing = _files_from_storage(stored_files)
    text_content = extract_plaintext(files_for_processing)
    if not text_content.strip():
        st.error("No textual content could be extracted from the uploaded files. Please check the documents.")
        return
    chunks = chunk_text(text_content)
    st.session_state["analyzer_chunks"] = chunks
    st.write(f"Extracted {len(chunks)} chunk(s) totaling approximately {len(text_content.split())} words.")

    st.header("Step 6 – Hybrid analysis")
    settings = _load_settings()
    llm_client = _build_llm_client(settings)

    if st.button("Generate analysis", key="analyzer_generate_button"):
        with st.spinner("Running document analysis..."):
            try:
                analysis = analyze_chunks_hybrid(
                    chunks,
                    src_lang=source_lang,
                    tgt_langs=target_langs,
                    pm_notes=pm_notes if st.session_state.get("analyzer_pm_enabled") else None,
                    llm_client=llm_client,
                )
            except Exception as exc:  # pragma: no cover - UI feedback only
                st.error(f"Document analysis failed: {exc}")
                analysis = None
            if analysis:
                metadata = analysis.setdefault("document_metadata", {})
                metadata["project_name"] = project_name.strip()
                metadata["source_language"] = source_lang or ""
                metadata["target_languages"] = target_langs
                metadata["file_count"] = len(stored_files)
                metadata["word_estimate"] = metadata.get("word_estimate") or len(text_content.split())

                notes_section = analysis.setdefault("pm_notes", {})
                if st.session_state.get("analyzer_pm_enabled") and pm_notes:
                    notes_section["original"] = pm_notes
                else:
                    notes_section.setdefault("original", "")
                notes_section.setdefault("system_notes", "")

                st.session_state["analyzer_analysis"] = analysis
                st.session_state["analyzer_analysis_json"] = json.dumps(
                    analysis,
                    indent=2,
                    ensure_ascii=False,
                )
                st.session_state["analyzer_analysis_accepted"] = False
                st.success("Analysis generated. Review the result below.")

    analysis_data = st.session_state.get("analyzer_analysis")
    if not analysis_data:
        st.info("Generate the analysis to review and export the report.")
        return

    st.header("Step 7 – Review & accept")
    with st.expander("Analysis result (editable)", expanded=True):
        edited_json = st.text_area(
            "Analysis JSON",
            value=st.session_state.get("analyzer_analysis_json", ""),
            height=400,
            key="analyzer_analysis_json",
        )
        accept_clicked = st.button("Accept analysis", key="analyzer_accept_button")

    if accept_clicked:
        try:
            parsed = json.loads(edited_json)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
        else:
            st.session_state["analyzer_analysis"] = parsed
            st.session_state["analyzer_analysis_json"] = json.dumps(parsed, indent=2, ensure_ascii=False)
            st.session_state["analyzer_analysis_accepted"] = True
            st.session_state["document_analysis"] = parsed
            st.session_state["document_analysis_json"] = st.session_state["analyzer_analysis_json"]
            st.success("Analysis accepted. You can now export the report.")

    if not st.session_state.get("analyzer_analysis_accepted"):
        st.info("Accept the analysis to enable exports.")
        return

    st.header("Step 8 – Export to DOCX")
    try:
        docx_bytes = analysis_to_docx(st.session_state["analyzer_analysis"])
    except Exception as exc:  # pragma: no cover - UI feedback only
        st.error(f"Failed to generate DOCX export: {exc}")
        return

    st.download_button(
        "Download analysis (Word)",
        data=docx_bytes,
        file_name=f"{project_name.strip() or 'analysis'}_analysis.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    st.success("Analysis ready for PM hand-off.")
