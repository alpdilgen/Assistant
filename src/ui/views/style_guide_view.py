from __future__ import annotations

import json
from typing import Any, Dict, List

import streamlit as st

from src.core.styleguide_builder import build_cayva_like_styleguide, styleguide_to_docx

LANGUAGE_LABELS: Dict[str, str] = {
    "bg": "Bulgarian",
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
}


def _format_lang(code: str | None) -> str:
    if not code:
        return ""
    normalised = code.strip().lower()
    return LANGUAGE_LABELS.get(normalised, code.upper())


def _ensure_state() -> None:
    defaults = {
        "styleguide_analysis_input": "",
        "styleguide_result": None,
        "styleguide_docx": None,
        "styleguide_target_lang": "",
        "styleguide_file_name": "style_guide.docx",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_analysis_from_session() -> None:
    analysis_json = st.session_state.get("document_analysis_json")
    if analysis_json:
        st.session_state["styleguide_analysis_input"] = analysis_json


def _build_styleguide(
    analysis: Dict[str, Any],
    src_lang: str,
    target_lang: str,
    project_name: str,
    pm_notes: str | None,
) -> None:
    terminology = st.session_state.get("terminology") or []
    styleguide = build_cayva_like_styleguide(
        analysis=analysis,
        pm_notes=pm_notes,
        terminology=terminology,
        src_lang=src_lang or "en",
        tgt_lang=target_lang or "en",
        project_name=project_name or "Style Guide",
    )
    st.session_state["styleguide_result"] = styleguide
    st.session_state["styleguide_docx"] = styleguide_to_docx(styleguide)
    st.session_state["styleguide_file_name"] = f"{project_name or 'style_guide'}_style_guide.docx"
    st.success("Style guide generated. Download it below.")


def render() -> None:
    _ensure_state()

    st.title("Style Guide Creator")
    st.caption("Use the accepted analysis to produce a tailored style guide, or run a quick guide from scratch.")

    mode = st.radio(
        "How would you like to provide project context?",
        ["Paste analysis JSON", "Quick style guide"],
        index=0,
    )

    if mode == "Paste analysis JSON":
        if not st.session_state.get("styleguide_analysis_input") and st.session_state.get("document_analysis_json"):
            _load_analysis_from_session()

        analysis_text = st.text_area(
            "Paste the analysis JSON (from the Document Analyzer export)",
            height=320,
            key="styleguide_analysis_input",
        )

        parsed_analysis: Dict[str, Any] | None = None
        metadata: Dict[str, Any] = {}
        if analysis_text.strip():
            try:
                parsed_analysis = json.loads(analysis_text)
                metadata = parsed_analysis.get("document_metadata", {})
            except json.JSONDecodeError:
                parsed_analysis = None
                metadata = {}

        target_candidates: List[str] = []
        if metadata:
            target_candidates = [lang for lang in metadata.get("target_languages", []) if lang]
            project_name = metadata.get("project_name", "")
            source_lang = metadata.get("source_language", "")
        else:
            project_name = ""
            source_lang = ""

        if target_candidates:
            default_target = st.session_state.get("styleguide_target_lang") or target_candidates[0]
            if default_target not in target_candidates:
                target_candidates.insert(0, default_target)
            target_lang = st.selectbox(
                "Select target language for the style guide",
                options=target_candidates,
                format_func=_format_lang,
                key="styleguide_target_lang",
            )
        else:
            target_lang = st.text_input(
                "Target language (ISO code)",
                value=st.session_state.get("styleguide_target_lang", ""),
                key="styleguide_target_lang",
            )

        if st.button("Generate style guide", key="styleguide_generate_from_json"):
            if not parsed_analysis:
                st.error("Please provide valid analysis JSON before generating the style guide.")
            elif not target_lang:
                st.error("Select or enter a target language.")
            else:
                pm_notes = (parsed_analysis.get("pm_notes", {}) or {}).get("original")
                _build_styleguide(parsed_analysis, source_lang, target_lang, project_name, pm_notes)

    else:  # Quick style guide
        st.subheader("Quick style guide setup")
        quick_project = st.text_input(
            "Project name",
            value=st.session_state.get("styleguide_quick_project", ""),
            key="styleguide_quick_project",
        )
        quick_source = st.text_input(
            "Source language (ISO code)",
            value=st.session_state.get("styleguide_quick_source", "en"),
            key="styleguide_quick_source",
        )
        quick_target = st.text_input(
            "Target language (ISO code)",
            value=st.session_state.get("styleguide_quick_target", ""),
            key="styleguide_quick_target",
        )
        quick_domain = st.text_input(
            "Primary domain",
            value=st.session_state.get("styleguide_quick_domain", "General technical"),
            key="styleguide_quick_domain",
        )
        quick_subdomain = st.text_input(
            "Key subdomain or subject",
            value=st.session_state.get("styleguide_quick_subdomain", ""),
            key="styleguide_quick_subdomain",
        )
        quick_notes = st.text_area(
            "PM instructions (optional)",
            value=st.session_state.get("styleguide_quick_notes", ""),
            key="styleguide_quick_notes",
            height=180,
        )

        if st.button("Generate quick style guide", key="styleguide_generate_quick"):
            analysis_stub = {
                "document_metadata": {
                    "project_name": quick_project,
                    "source_language": quick_source,
                    "target_languages": [quick_target] if quick_target else [],
                    "file_count": 0,
                    "word_estimate": 0,
                },
                "classification": {
                    "domain": quick_domain or "General technical",
                    "subdomains": [quick_subdomain] if quick_subdomain else [],
                    "related_fields": [],
                },
                "complexity": {"level": 2, "drivers": []},
                "translator_profile": {
                    "required_background": "Translator experienced in the specified domain.",
                    "preferred_experience": "Has completed similar projects with EU clients.",
                    "tools": ["memoQ", "Trados", "Xbench", "Verifika"],
                },
                "risks": [],
                "references": ["IATE", "EUR-Lex", "Client termbase"],
                "pm_notes": {"original": quick_notes, "system_notes": ""},
            }
            _build_styleguide(analysis_stub, quick_source, quick_target, quick_project, quick_notes)

    styleguide_data = st.session_state.get("styleguide_result")
    if styleguide_data:
        st.header("Style guide preview")
        st.json(styleguide_data)

    docx_bytes = st.session_state.get("styleguide_docx")
    if docx_bytes:
        st.download_button(
            "Download style guide (Word)",
            data=docx_bytes,
            file_name=st.session_state.get("styleguide_file_name", "style_guide.docx"),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
