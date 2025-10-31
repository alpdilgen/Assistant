import streamlit as st
from src.core.ingestion import extract_plaintext, chunk_text
from src.core.document_analyzer import analyze_chunks_hybrid
from src.core.export_doc import analysis_to_docx


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
    # only update session if user actually typed something
    if project_name_input:
        st.session_state["project_name"] = project_name_input.strip()

    # 4) PM NOTES (optional)
    pm_notes_input = st.text_area(
        "PM / client instructions (optional)",
        value=st.session_state["pm_notes"],
        key="pm_notes_input",
    )
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

        # extract + chunk
        raw_text = extract_plaintext(files)
        chunks = chunk_text(raw_text)

        # you will need to provide src_lang/tgt_langs from earlier step – here we just mock:
        src_lang = "en"
        tgt_langs = ["sk"]

        analysis = analyze_chunks_hybrid(
            chunks=chunks,
            src_lang=src_lang,
            tgt_langs=tgt_langs,
            pm_notes=st.session_state["pm_notes"],
            llm_client=None,  # <- replace with your real client
        )
        st.session_state["analysis_json"] = analysis
        st.session_state["analysis_accepted"] = False

    # 6) SHOW ANALYSIS IF AVAILABLE
    analysis = st.session_state.get("analysis_json")
    if analysis:
        st.subheader("Proposed analysis")
        st.json(analysis)

        if st.button("Accept this analysis"):
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
