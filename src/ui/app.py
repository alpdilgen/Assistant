import os
import sys
import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

import streamlit as st
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.core.document_ingestion import detect_langs_from_file, extract_text_chunks
from src.core.document_analysis import classify_domain
from src.core.styleguide_builder import build_styleguide, export_styleguide_docx
from src.core.terminology_client import parse_terminology_file, send_to_external_termextractor
from src.core.llm_client import LLMClient

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from streamlit.runtime.uploaded_file_manager import UploadedFile

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"

LANGUAGE_LABELS: Dict[str, str] = {
    "bg": "Bulgarian",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "ga": "Irish",
    "it": "Italian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mt": "Maltese",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "tr": "Turkish",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ar": "Arabic",
    "zh-CN": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "no": "Norwegian",
    "sr": "Serbian",
    "bs": "Bosnian",
}

ALL_LANGS: List[str] = [
    "en",
    "sk",
    "de",
    "fr",
    "it",
    "es",
    "pt",
    "nl",
    "pl",
    "ro",
    "bg",
    "cs",
    "da",
    "et",
    "fi",
    "el",
    "hu",
    "hr",
    "lt",
    "lv",
    "mt",
    "sl",
    "sv",
    "tr",
    "ru",
    "uk",
    "ar",
    "zh-CN",
    "ja",
    "ko",
    "no",
    "sr",
    "bs",
]


@st.cache_resource
def load_settings() -> Dict[str, Any]:
    """Load YAML configuration shared across UI and core modules."""

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_llm_client(settings: Dict[str, Any]) -> LLMClient | None:
    """Instantiate the shared LLM client if configuration enables it."""

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


def _ensure_session_defaults() -> None:
    """Initialise Streamlit session state keys used throughout the workflow."""

    defaults = {
        "stored_files": [],
        "file_languages": {},
        "languages_confirmed": False,
        "project_name": "",
        "pm_notes": "",
        "pm_notes_enabled": False,
        "analysis_chunks": [],
        "analysis_result": None,
        "analysis_confirmed": False,
        "terminology_entries": [],
        "terminology_required": False,
        "styleguide_data": None,
        "styleguide_docx": None,
        "styleguide_mode": "ai",
        "editing_analysis": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _persist_uploaded_files(uploads: Iterable["UploadedFile"]) -> List[dict]:
    """Convert Streamlit UploadedFile objects into session-friendly payloads."""

    stored: List[dict] = []
    for uploaded in uploads:
        if uploaded is None:
            continue
        data = uploaded.getvalue()
        stored.append({"name": uploaded.name, "data": data, "type": uploaded.type})
    return stored


def _files_from_session(stored_files: Iterable[dict]) -> List[io.BytesIO]:
    """Re-create file-like buffers from bytes stored in the session."""

    buffers: List[io.BytesIO] = []
    for file_info in stored_files:
        buffer = io.BytesIO(file_info.get("data", b""))
        buffer.name = file_info.get("name", "uploaded_file")
        buffers.append(buffer)
    return buffers


def _format_language(code: str | None) -> str:
    """Format a language code for display in selectboxes safely handling nullish values."""

    if not code:
        return "— select language —"
    code = code.strip()
    if not code:
        return "— select language —"
    code_lower = code.lower()
    if code_lower in LANGUAGE_LABELS:
        return LANGUAGE_LABELS[code_lower]
    return code.upper()


def _refresh_language_detection(files: List[io.BytesIO]) -> None:
    """Detect languages for each uploaded file and store them in session state."""

    detections: Dict[str, dict] = {}
    for file in files:
        file.seek(0)
        detected_src, detected_tgt = detect_langs_from_file(file)
        file.seek(0)
        targets = [detected_tgt] if detected_tgt else []
        detections[file.name] = {
            "detected_source": detected_src,
            "detected_target": detected_tgt,
            "source": detected_src or "",
            "targets": [target for target in targets if target],
            "is_bilingual": bool(detected_tgt),
        }
    st.session_state["file_languages"] = detections
    st.session_state["languages_confirmed"] = False


def _languages_ready(language_map: Dict[str, dict]) -> bool:
    """Determine whether the language confirmation step can be considered complete."""

    if not language_map:
        return False
    for entry in language_map.values():
        if not entry.get("source"):
            return False
        targets = entry.get("targets", [])
        if entry.get("is_bilingual") and len(targets) != 1:
            return False
        if not entry.get("is_bilingual") and not targets:
            return False
    return True


def _update_progress_sidebar(step_status: Dict[str, bool]) -> None:
    """Render a progress indicator and read-only checklist in the sidebar."""

    total_steps = len(step_status)
    completed_steps = sum(1 for done in step_status.values() if done)
    progress_value = completed_steps / total_steps if total_steps else 0.0
    st.sidebar.progress(progress_value)
    for name, done in step_status.items():
        st.sidebar.checkbox(name, value=done, disabled=True)


def main() -> None:
    """Entry point for the Streamlit workflow orchestrating the localisation pipeline."""

    st.set_page_config(page_title="Assistant Workflow", layout="wide")
    _ensure_session_defaults()

    settings = load_settings()
    llm_client = _build_llm_client(settings)

    st.title("Translation Project Assistant")
    st.caption("Guide localisation projects from intake to a ready-to-share style guide.")

    uploaded_files = st.file_uploader(
        "1️⃣ Upload files",
        type=["docx", "pdf", "txt", "xlf", "xliff", "mqxliff"],
        accept_multiple_files=True,
        help="Upload monolingual or bilingual files to kick off the workflow.",
    )
    if uploaded_files:
        st.session_state["stored_files"] = _persist_uploaded_files(uploaded_files)
        file_buffers = _files_from_session(st.session_state["stored_files"])
        _refresh_language_detection(file_buffers)
    stored_files = st.session_state.get("stored_files", [])
    file_buffers = _files_from_session(stored_files)

    st.divider()
    st.subheader("2️⃣ Confirm languages")
    if not stored_files:
        st.info("Upload at least one document to continue.")
    else:
        language_map = st.session_state.get("file_languages", {})
        if not language_map:
            _refresh_language_detection(file_buffers)
            language_map = st.session_state.get("file_languages", {})

        available_languages = settings.get("languages", {}).get("default_set") or ALL_LANGS
        available_languages = [
            lang
            for lang in available_languages
            if isinstance(lang, str) and lang.strip()
        ]
        for file_name, info in language_map.items():
            with st.expander(file_name, expanded=True):
                detected_src = info.get("detected_source")
                detected_tgt = info.get("detected_target")
                is_bilingual = bool(info.get("is_bilingual"))
                if is_bilingual:
                    if detected_src and detected_tgt:
                        st.markdown(
                            f"Detected: **{detected_src.upper()} → {detected_tgt.upper()}**. Confirm or change below."
                        )
                    else:
                        st.info("At least one language could not be detected from file. Please select below.")
                elif detected_src:
                    st.markdown(
                        f"Detected: **{detected_src.upper()}**. Confirm the source language and choose target(s)."
                    )

                if not detected_src:
                    st.warning(
                        f"Unable to detect language automatically for {file_name}. Please select manually."
                    )

                selectable_sources: List[str] = []
                for candidate in (detected_src, info.get("source")):
                    if candidate and candidate not in selectable_sources:
                        selectable_sources.append(candidate)
                for lang in available_languages:
                    if lang not in selectable_sources:
                        selectable_sources.append(lang)
                selectable_sources = [""] + selectable_sources

                current_source = info.get("source") or detected_src or ""
                source_index = (
                    selectable_sources.index(current_source)
                    if current_source in selectable_sources
                    else 0
                )
                selected_source = st.selectbox(
                    "Source language",
                    options=selectable_sources,
                    index=source_index,
                    format_func=_format_language,
                    key=f"{file_name}_source",
                )
                info["source"] = selected_source.strip() if selected_source else ""

                if is_bilingual:
                    tgt_candidates: List[str] = []
                    existing_targets = [target for target in info.get("targets", []) if target]
                    current_target = existing_targets[0] if existing_targets else (detected_tgt or "")
                    for candidate in (detected_tgt, current_target):
                        if candidate and candidate not in tgt_candidates:
                            tgt_candidates.append(candidate)
                    for lang in available_languages:
                        if lang not in tgt_candidates:
                            tgt_candidates.append(lang)
                    tgt_options = [""] + tgt_candidates
                    target_index = (
                        tgt_options.index(current_target)
                        if current_target in tgt_options
                        else 0
                    )
                    chosen = st.selectbox(
                        "Target language",
                        options=tgt_options,
                        index=target_index,
                        format_func=_format_language,
                        key=f"{file_name}_target",
                    )
                    info["targets"] = [chosen.strip()] if chosen else []
                else:
                    existing = [code for code in info.get("targets", []) if code]
                    info["targets"] = st.multiselect(
                        "Target languages",
                        options=available_languages,
                        default=[code for code in existing if code in available_languages],
                        format_func=_format_language,
                        key=f"{file_name}_targets",
                        help="Select one or more target languages for this monolingual file.",
                    )
        st.session_state["languages_confirmed"] = _languages_ready(language_map)
        if st.session_state["languages_confirmed"]:
            st.success("Languages confirmed for all files.")
        else:
            st.info("Confirm source and target languages to unlock the next steps.")

    st.divider()
    st.subheader("3️⃣ Project details")
    project_name = st.text_input(
        "Enter project name (required)",
        value=st.session_state.get("project_name", ""),
        key="project_name_input",
    )
    st.session_state["project_name"] = project_name.strip()
    if not project_name.strip():
        st.warning("Project name is required before continuing.")

    st.divider()
    st.subheader("4️⃣ PM information (optional)")
    pm_choice = st.radio(
        "Does the PM want to provide project info?",
        options=("No", "Yes"),
        index=1 if st.session_state.get("pm_notes_enabled") else 0,
        horizontal=True,
        key="pm_notes_toggle",
    )
    st.session_state["pm_notes_enabled"] = pm_choice == "Yes"
    if st.session_state["pm_notes_enabled"]:
        st.session_state["pm_notes"] = st.text_area(
            "Enter or paste PM instructions",
            value=st.session_state.get("pm_notes", ""),
            height=180,
            key="pm_notes_text",
        )
    else:
        st.session_state["pm_notes"] = ""
        st.caption("You can provide PM instructions later if needed.")

    st.divider()
    st.subheader("5️⃣ Document analysis")
    analysis_ready = (
        stored_files
        and st.session_state.get("languages_confirmed")
        and bool(project_name.strip())
    )
    chunking_cfg = settings.get("chunking", {"max_chars": 4000, "overlap": 400})
    if not analysis_ready:
        st.info("Upload files, confirm languages, and provide a project name to analyse documents.")
    else:
        if st.button("Analyse uploaded documents", key="analyse_documents"):
            for info in st.session_state.get("file_languages", {}).values():
                if not info.get("source"):
                    st.error("Please select a source language to continue.")
                    st.stop()
            chunks = extract_text_chunks(
                _files_from_session(stored_files),
                max_chars=int(chunking_cfg.get("max_chars", 4000)),
                overlap=int(chunking_cfg.get("overlap", 400)),
            )
            st.session_state["analysis_chunks"] = chunks
            st.session_state["analysis_result"] = classify_domain(chunks, llm_client)
            st.session_state["analysis_confirmed"] = False
            st.session_state["editing_analysis"] = False

        analysis_result = st.session_state.get("analysis_result")
        if analysis_result:
            domain = analysis_result.get("domain", "Unknown")
            subdomains = ", ".join(analysis_result.get("subdomains", [])) or "—"
            related = ", ".join(analysis_result.get("related", [])) or "—"
            col1, col2, col3 = st.columns(3)
            col1.markdown("**Domain**")
            col1.write(domain)
            col2.markdown("**Subdomains**")
            col2.write(subdomains)
            col3.markdown("**Related fields**")
            col3.write(related)

            action_col1, action_col2 = st.columns(2)
            if action_col1.button("Accept analysis", key="accept_analysis"):
                st.session_state["analysis_confirmed"] = True
                st.session_state["editing_analysis"] = False
            if action_col2.button("Edit manually", key="edit_analysis"):
                st.session_state["editing_analysis"] = True

            if st.session_state.get("editing_analysis"):
                with st.form("manual_analysis_edit"):
                    manual_domain = st.text_input(
                        "Main domain",
                        value=analysis_result.get("domain", ""),
                    )
                    manual_subdomains = st.text_input(
                        "Subdomains (comma separated)",
                        value=", ".join(analysis_result.get("subdomains", [])),
                    )
                    manual_related = st.text_area(
                        "Related technical fields (comma separated)",
                        value=", ".join(analysis_result.get("related", [])),
                        height=100,
                    )
                    submitted = st.form_submit_button("Save domain selection")
                    if submitted:
                        st.session_state["analysis_result"] = {
                            "domain": manual_domain.strip() or "General",
                            "subdomains": [part.strip() for part in manual_subdomains.split(",") if part.strip()],
                            "related": [part.strip() for part in manual_related.split(",") if part.strip()],
                        }
                        st.session_state["analysis_confirmed"] = True
                        st.session_state["editing_analysis"] = False
                        st.success("Domain details updated.")
        else:
            st.caption("Run the analysis to populate domain, subdomain, and related field insights.")

    st.divider()
    st.subheader("6️⃣ Terminology extraction")
    term_choice = st.radio(
        "Do you want to extract terminology from these files now?",
        options=("No", "Yes"),
        index=1 if st.session_state.get("terminology_required") else 0,
        horizontal=True,
        key="terminology_toggle",
    )
    st.session_state["terminology_required"] = term_choice == "Yes"

    terminology_entries = st.session_state.get("terminology_entries", [])
    if st.session_state["terminology_required"] and stored_files:
        language_map = st.session_state.get("file_languages", {})
        src_languages = {info.get("source") for info in language_map.values() if info.get("source")}
        tgt_languages = {target for info in language_map.values() for target in info.get("targets", [])}
        primary_src = next(iter(src_languages), "en")
        primary_tgt = next(iter(tgt_languages), "en")
        send_to_external_termextractor(
            _files_from_session(stored_files),
            src_lang=primary_src,
            tgt_lang=primary_tgt,
            project_name=project_name.strip() or "Translation Project",
            base_url=settings.get("external_tools", {}).get("terminology_url"),
        )
        uploaded_terminology = st.file_uploader(
            "Upload extracted terminology file (XLSX/CSV/TBX)",
            type=["xlsx", "csv", "tbx"],
            key="terminology_upload",
        )
        if uploaded_terminology is not None:
            try:
                terminology_entries = parse_terminology_file(uploaded_terminology)
                st.session_state["terminology_entries"] = terminology_entries
                st.success(f"Loaded {len(terminology_entries)} terminology entries.")
            except Exception as exc:  # pragma: no cover - UI feedback only
                st.error(f"Unable to parse terminology file: {exc}")
    elif term_choice == "No":
        st.caption("Terminology extraction can be completed later and uploaded when ready.")

    st.divider()
    st.subheader("7️⃣ Style guide creation")
    if not st.session_state.get("analysis_confirmed"):
        st.info("Accept or edit the document analysis before generating the style guide.")
    else:
        styleguide_mode = st.radio(
            "How should we complete the style guide?",
            options=("Let AI fill missing sections", "I'll add manual answers"),
            index=0 if st.session_state.get("styleguide_mode") == "ai" else 1,
            key="styleguide_mode_selector",
        )
        st.session_state["styleguide_mode"] = "ai" if styleguide_mode.startswith("Let AI") else "manual"
        manual_additions = ""
        if st.session_state["styleguide_mode"] == "manual":
            manual_additions = st.text_area(
                "Add manual answers or clarifications for the questionnaire",
                height=220,
                key="styleguide_manual_notes",
            )
        if st.button("Generate style guide", key="generate_styleguide"):
            language_map = st.session_state.get("file_languages", {})
            sources = sorted({info.get("source") for info in language_map.values() if info.get("source")})
            targets = sorted({target for info in language_map.values() for target in info.get("targets", [])})
            styleguide = build_styleguide(
                analysis=st.session_state.get("analysis_result", {}),
                pm_notes=st.session_state.get("pm_notes", ""),
                terminology=terminology_entries,
                langs={"sources": sources, "targets": targets},
                llm_client=llm_client,
                project_name=project_name.strip(),
                manual_notes=manual_additions,
            )
            st.session_state["styleguide_data"] = styleguide
            st.session_state["styleguide_docx"] = export_styleguide_docx(
                styleguide,
                filename=project_name.strip() or "StyleGuide",
            )
            st.success("Style guide assembled. Proceed to download below.")

        if st.session_state.get("styleguide_data"):
            st.markdown("#### Preview")
            st.json(st.session_state["styleguide_data"])

    st.divider()
    st.subheader("8️⃣ Downloads")
    docx_bytes = st.session_state.get("styleguide_docx")
    if docx_bytes:
        st.download_button(
            "Download Style Guide (Word)",
            data=docx_bytes,
            file_name=f"{project_name.strip() or 'StyleGuide'}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    else:
        st.caption("Generate the style guide to enable downloads.")

    step_status = {
        "1️⃣ Upload files": bool(stored_files),
        "2️⃣ Confirm languages": st.session_state.get("languages_confirmed", False),
        "3️⃣ Enter project name": bool(project_name.strip()),
        "4️⃣ PM info": True,  # optional step always considered complete
        "5️⃣ Analyse documents": bool(st.session_state.get("analysis_result")),
        "6️⃣ Terminology": not st.session_state.get("terminology_required")
        or bool(st.session_state.get("terminology_entries")),
        "7️⃣ Generate style guide": bool(st.session_state.get("styleguide_data")),
        "8️⃣ Download outputs": bool(docx_bytes),
    }
    st.sidebar.header("Workflow progress")
    _update_progress_sidebar(step_status)


if __name__ == "__main__":
    main()
