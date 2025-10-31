from __future__ import annotations

import streamlit as st

from src.core.document_analyzer import analyze_chunks_hybrid
from src.core.document_ingestion import detect_langs_from_file, ingest_files
from src.core.export_doc import analysis_to_docx
from src.core.llm_client import load_llm_client


def render():
    st.title("Document Analyzer")

    # 1) SAFE INIT OF SESSION KEYS (create all keys up front)
    if "project_name" not in st.session_state:
        st.session_state["project_name"] = ""
    if "analysis_json" not in st.session_state:
        st.session_state["analysis_json"] = None
    if "analysis_accepted" not in st.session_state:
        st.session_state["analysis_accepted"] = False
    if "pm_notes" not in st.session_state:
        st.session_state["pm_notes"] = ""
    if "source_language" not in st.session_state:
        st.session_state["source_language"] = ""
    if "target_languages" not in st.session_state:
        st.session_state["target_languages"] = ""

    # 2) UI – FILE UPLOAD
    files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=["docx", "pdf", "txt", "xlf", "xliff", "mqxliff"],
    )

    # 3) PROJECT NAME (do NOT write to session every render blindly)
    project_name_input = st.text_input(
        "Project name (required)",
        value=st.session_state["project_name"],
        key="project_name_input",
    )
    if project_name_input != st.session_state["project_name"]:
        st.session_state["project_name"] = project_name_input.strip()

    # 3a) LANGUAGE FIELDS (optional overrides)
    col_src, col_tgt = st.columns(2)
    with col_src:
        source_language_input = st.text_input(
            "Source language (optional)",
            value=st.session_state["source_language"],
            key="source_language_input",
        )
        if source_language_input != st.session_state["source_language"]:
            st.session_state["source_language"] = source_language_input.strip()
    with col_tgt:
        target_languages_input = st.text_input(
            "Target languages (optional, comma separated)",
            value=st.session_state["target_languages"],
            key="target_languages_input",
        )
        if target_languages_input != st.session_state["target_languages"]:
            st.session_state["target_languages"] = target_languages_input.strip()

    # 4) PM NOTES (optional)
    pm_notes_input = st.text_area(
        "PM / client instructions (optional)",
        value=st.session_state["pm_notes"],
        key="pm_notes_input",
    )
    if pm_notes_input != st.session_state["pm_notes"]:
        st.session_state["pm_notes"] = pm_notes_input

    # 5) BUTTON TO RUN ANALYSIS
    run_clicked = st.button("Run analysis")
    if run_clicked:
        if not files:
            st.error("Please upload at least one file first.")
            st.stop()
        if not st.session_state["project_name"]:
            st.error("Please enter a project name before running the analysis.")
            st.stop()

        chunks, combined_text = ingest_files(files)
        if not combined_text.strip():
            st.error("Unable to extract text from the uploaded documents.")
            st.stop()

        llm_client = load_llm_client()
        if llm_client is None:
            st.info("LLM client not configured; using heuristic fallback template where needed.")

        detected_src = None
        detected_tgt = None
        try:
            first_file = files[0]
        except IndexError:
            first_file = None
        if first_file is not None:
            detected_src, detected_tgt = detect_langs_from_file(first_file)

        src_lang = st.session_state["source_language"].strip()
        if not src_lang and detected_src:
            src_lang = detected_src
            st.session_state["source_language"] = detected_src

        raw_targets = st.session_state["target_languages"].split(",") if st.session_state["target_languages"] else []
        tgt_langs = [lang.strip() for lang in raw_targets if lang.strip()]
        if not tgt_langs and detected_tgt:
            tgt_langs = [detected_tgt]
            st.session_state["target_languages"] = detected_tgt

        analysis = analyze_chunks_hybrid(
            chunks=chunks,
            src_lang=src_lang,
            tgt_langs=tgt_langs,
            pm_notes=st.session_state["pm_notes"],
            llm_client=llm_client,
            project_name=st.session_state["project_name"],
            full_text=combined_text,
            file_count=len(files),
        )
        st.session_state["analysis_json"] = analysis
        st.session_state["analysis_accepted"] = False

    # 6) SHOW ANALYSIS IF AVAILABLE
    analysis = st.session_state.get("analysis_json")
    if analysis:
        st.subheader("Proposed analysis")
        st.json(analysis)

        if st.button("Accept this analysis", key="accept_analysis_button"):
            st.session_state["analysis_accepted"] = True

    # 7) EXPORT IF ACCEPTED
    if st.session_state.get("analysis_accepted") and st.session_state.get("analysis_json"):
        docx_bytes = analysis_to_docx(st.session_state["analysis_json"])
        st.download_button(
            "Download analysis as Word",
            data=docx_bytes,
            file_name=f"{st.session_state.get('project_name','analysis')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )


__all__ = ["render"]
