import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

import streamlit as st
import yaml

from src.core.document_ingestion import DocumentPayload, load_documents
from src.core.document_analysis import analyze_document
from src.core.styleguide_builder import build_style_guide
from src.core.persona_builder import build_persona
from src.core.pm_brief import build_pm_brief
from src.core.terminology_client import send_to_external_termextractor
from src.core.llm_client import LLMClient
from src.ui import components

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"

LANGUAGE_OPTIONS: Dict[str, str] = {
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


@st.cache_resource
def load_settings() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_llm_client(settings: Dict[str, Any], enabled: bool) -> LLMClient | None:
    llm_settings = settings.get("llm", {})
    if not enabled or not llm_settings.get("enabled"):
        return None
    try:
        return LLMClient(
            provider=llm_settings.get("provider", "openai"),
            model=llm_settings.get("model", "gpt-4o-mini"),
            temperature=float(llm_settings.get("temperature", 0.0)),
            max_retries=int(llm_settings.get("max_retries", 2)),
            timeout=float(llm_settings.get("timeout", 60)),
        )
    except Exception as exc:
        st.sidebar.error(f"Failed to initialise LLM client: {exc}")
        return None


def _language_selector(label: str, session_key: str, default: str) -> str:
    options = list(LANGUAGE_OPTIONS.keys())
    if session_key not in st.session_state or st.session_state[session_key] not in options:
        st.session_state[session_key] = default
    default_index = options.index(st.session_state[session_key])
    selection = st.selectbox(
        label,
        options=options,
        index=default_index,
        format_func=lambda code: f"{LANGUAGE_OPTIONS[code]} ({code})",
        key=f"{session_key}_selector",
    )
    st.session_state[session_key] = selection
    return selection


def _average(values: Sequence[float]) -> float:
    filtered = [value for value in values if value is not None]
    return float(mean(filtered)) if filtered else 0.0


def _combine_document_analyses(analyses: List[dict]) -> dict:
    if not analyses:
        return {}

    domains = [analysis.get("domain") for analysis in analyses if analysis.get("domain")]
    domain_counter = Counter(domains)
    unique_domains = sorted(set(domains))
    primary_domain = domain_counter.most_common(1)[0][0] if domain_counter else "General"

    subdomains = sorted({sub for analysis in analyses for sub in analysis.get("subdomains", []) if sub})
    tone_counter = Counter(analysis.get("tone") for analysis in analyses if analysis.get("tone"))
    tone = tone_counter.most_common(1)[0][0] if tone_counter else "neutral"
    audience_counter = Counter(analysis.get("audience") for analysis in analyses if analysis.get("audience"))
    audience = audience_counter.most_common(1)[0][0] if audience_counter else "general"
    difficulty = max((analysis.get("difficulty_level", 0) for analysis in analyses), default=0)

    signals_list = [analysis.get("complexity_signals", {}) for analysis in analyses]
    aggregated_signals = {
        "avg_sentence_len": round(_average([signals.get("avg_sentence_len") for signals in signals_list]), 2),
        "technical_term_density": round(_average([signals.get("technical_term_density") for signals in signals_list]), 4),
        "abbreviation_density": round(_average([signals.get("abbreviation_density") for signals in signals_list]), 4),
        "numeric_density": round(_average([signals.get("numeric_density") for signals in signals_list]), 4),
        "named_entities": sorted({entity for signals in signals_list for entity in signals.get("named_entities", [])}),
        "terminology_count": int(sum(signals.get("terminology_count", 0) for signals in signals_list)),
    }

    document_summaries = [
        {
            "filename": analysis.get("filename"),
            "summary": analysis.get("summary"),
            "difficulty_level": analysis.get("difficulty_level"),
            "tone": analysis.get("tone"),
        }
        for analysis in analyses
    ]
    combined_summary = "\n\n".join(
        f"{item['filename']}: {item['summary']}".strip()
        for item in document_summaries
        if item.get("summary")
    ).strip()

    return {
        "summary": combined_summary,
        "combined_summary": combined_summary,
        "domain": primary_domain,
        "domains": unique_domains or [primary_domain],
        "subdomains": subdomains,
        "tone": tone,
        "audience": audience,
        "difficulty_level": difficulty,
        "complexity_signals": aggregated_signals,
        "document_breakdown": document_summaries,
        "documents": analyses,
        "total_documents": len(analyses),
    }


def _build_document_analyses(
    payloads: Sequence[DocumentPayload],
    *,
    src_lang: str,
    tgt_lang: str,
    llm_client: LLMClient | None,
    prompts: Dict[str, Any],
) -> List[dict]:
    analysis_prompt = prompts.get("analysis") if llm_client else None
    analyses: List[dict] = []
    for payload in payloads:
        document_analysis = analyze_document(
            payload.raw_text,
            terms=[],
            use_llm_summary=bool(llm_client and analysis_prompt),
            llm_client=llm_client,
            prompt=analysis_prompt,
        )
        document_analysis["filename"] = payload.filename
        document_analysis["source_language"] = payload.source_language or src_lang
        document_analysis["target_language"] = payload.target_language or tgt_lang
        analyses.append(document_analysis)
    return analyses


def main() -> None:
    st.set_page_config(page_title="LSP Assistant", layout="wide")
    settings = load_settings()

    with st.sidebar:
        st.title("Project settings")
        src_lang = _language_selector("Source language", "src_lang", "en")
        tgt_lang = _language_selector("Target language", "tgt_lang", "es")
        document_type = st.selectbox("Document type", options=settings.get("document_types", []))
        use_alignment = st.checkbox("Show bilingual segments", value=True)
        generate_style_guide = st.checkbox("Generate style guide", value=True)
        generate_persona = st.checkbox("Generate translator persona", value=True)
        generate_pm_brief = st.checkbox("Generate PM brief", value=True)
        enable_llm = st.checkbox(
            "Use LLM prompts when available",
            value=settings.get("llm", {}).get("enabled", False),
        )

    llm_client = _build_llm_client(settings, enable_llm)
    prompts = settings.get("prompts", {})

    st.header("AI-Powered Localization Assistant")
    st.caption(
        "Upload one or more documents to analyse content, build translation style guides, and brief linguists and project managers."
    )

    project_name = st.text_input(
        "Project or client name (optional)",
        value=st.session_state.get("project_name", ""),
        help="Used for style guide headers and terminology hand-offs.",
        key="project_name_input",
    )
    st.session_state["project_name"] = project_name

    uploaded_files = st.file_uploader(
        "Upload one or more files",
        type=["docx", "pdf", "txt", "xlf", "xliff", "mqxliff"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload at least one document to begin.")
        return

    payloads = load_documents(
        uploaded_files,
        default_source_lang=src_lang,
        default_target_lang=tgt_lang,
    )

    if not payloads:
        st.error("No supported documents were uploaded.")
        return

    document_analyses = _build_document_analyses(
        payloads,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        llm_client=llm_client,
        prompts=prompts,
    )
    combined_analysis = _combine_document_analyses(document_analyses)
    combined_analysis["document_type"] = document_type
    combined_analysis["source_language"] = src_lang
    combined_analysis["target_language"] = tgt_lang
    combined_analysis["project_name"] = project_name

    st.markdown("### Terminology planning")
    terminology_choice = st.radio(
        "Do you want to create terminology from these documents using the external Termextractor?",
        options=("No", "Yes"),
        horizontal=True,
        key="terminology_choice",
    )
    termextractor_url: str | None = None
    if terminology_choice == "Yes":
        termextractor_url = send_to_external_termextractor(
            uploaded_files,
            src_lang,
            tgt_lang,
            project_name=project_name or combined_analysis.get("domains", ["Project"])[0],
        )
    else:
        st.caption("Terminology extraction can be initiated later if required.")

    terms: List[dict] = []

    style_guide: Dict[str, Any] | None = None
    allow_edit = False
    if generate_style_guide:
        st.markdown("### Style guide inputs")
        pm_choice = st.radio(
            "Do you already have style guide inputs from the client/PM?",
            options=("No", "Yes"),
            horizontal=True,
            key="pm_inputs_choice",
        )
        pm_inputs: Dict[str, str] | None = None
        if pm_choice == "Yes":
            st.markdown("Provide the information captured in the Translation Style Guide Questionnaire.")
            pm_inputs = {
                "company_name": st.text_input(
                    "Company / project name",
                    value=st.session_state.get("styleguide_company", project_name),
                    key="styleguide_company",
                ),
                "existing_glossaries": st.text_input(
                    "Existing glossaries?",
                    value=st.session_state.get("styleguide_glossaries", ""),
                    key="styleguide_glossaries",
                ),
                "tone_voice": st.text_area(
                    "Tone & voice",
                    value=st.session_state.get("styleguide_tone", ""),
                    height=100,
                    key="styleguide_tone",
                ),
                "dnt_list": st.text_area(
                    "Do-not-translate list",
                    value=st.session_state.get("styleguide_dnt", ""),
                    height=100,
                    key="styleguide_dnt",
                ),
                "number_rules": st.text_area(
                    "Number/date/currency rules",
                    value=st.session_state.get("styleguide_numbers", ""),
                    height=100,
                    key="styleguide_numbers",
                ),
                "ui_rules": st.text_area(
                    "UI-specific rules",
                    value=st.session_state.get("styleguide_ui", ""),
                    height=100,
                    key="styleguide_ui",
                ),
                "sign_off_person": st.text_input(
                    "Sign-off person",
                    value=st.session_state.get("styleguide_signoff", ""),
                    key="styleguide_signoff",
                ),
            }
        allow_edit = st.checkbox(
            "Allow PM to edit before download",
            value=st.session_state.get("styleguide_allow_edit", False),
            key="styleguide_allow_edit",
        )
        style_prompt = prompts.get("style_guide") if llm_client else None
        style_guide = build_style_guide(
            combined_analysis,
            src_lang,
            tgt_lang,
            terms=terms,
            pm_inputs=pm_inputs,
            project_name=(pm_inputs or {}).get("company_name") or project_name,
            llm_client=llm_client,
            prompt=style_prompt,
        )

        if allow_edit and style_guide:
            st.markdown("#### Editable style guide draft (YAML)")
            default_yaml = st.session_state.get("styleguide_editor_text")
            if not default_yaml:
                default_yaml = yaml.safe_dump(style_guide, sort_keys=False, allow_unicode=True)
            edited_yaml = st.text_area(
                "Update the style guide as needed, then share or download.",
                value=default_yaml,
                height=420,
                key="styleguide_editor_text",
            )
            try:
                parsed = yaml.safe_load(edited_yaml) or {}
                if isinstance(parsed, dict):
                    style_guide = parsed
                else:
                    st.warning("Edited content must resolve to a dictionary; keeping the previous version.")
            except yaml.YAMLError as exc:
                st.warning(f"Unable to parse edited content: {exc}")

    persona: Dict[str, Any] | None = None
    if generate_persona:
        persona_prompt = prompts.get("persona") if llm_client else None
        persona = build_persona(
            combined_analysis,
            style_guide or {},
            src_lang,
            tgt_lang,
            llm_client=llm_client,
            prompt=persona_prompt,
        )

    pm_brief: Dict[str, Any] | None = None
    if generate_pm_brief:
        brief_prompt = prompts.get("pm_brief") if llm_client else None
        pm_brief = build_pm_brief(
            combined_analysis,
            terms,
            llm_client=llm_client,
            prompt=brief_prompt,
        )

    tab_titles = [
        "Document analysis",
        "Style guide",
        "Translator persona",
        "Terminology",
        "PM brief",
    ]
    tabs = st.tabs(tab_titles)

    with tabs[0]:
        st.subheader("Document analysis")
        components.render_json(combined_analysis)
        components.download_json_button("Download analysis JSON", combined_analysis, "analysis.json")
        components.download_docx_button("Download analysis DOCX", combined_analysis, "analysis.docx", "Document Analysis")
        if document_analyses:
            st.markdown("#### Per-document insights")
            for document_analysis in document_analyses:
                with st.expander(document_analysis.get("filename", "Document")):
                    components.render_json(document_analysis)
        if use_alignment:
            segmented_payloads = [payload for payload in payloads if payload.segments]
            if segmented_payloads:
                st.markdown("#### Bilingual segments preview")
                for payload in segmented_payloads:
                    st.markdown(f"**{payload.filename}**")
                    st.dataframe(payload.segments[:20])

    with tabs[1]:
        st.subheader("Style guide")
        if style_guide:
            components.render_json(style_guide)
            components.download_json_button("Download style guide JSON", style_guide, "style_guide.json")
            components.download_docx_button(
                "Download style guide DOCX",
                style_guide,
                "style_guide.docx",
                "Translation Style Guide",
            )
        else:
            st.info("Enable style guide generation to view this tab.")

    with tabs[2]:
        st.subheader("Translator persona")
        if persona:
            components.render_json(persona)
            components.download_json_button("Download persona JSON", persona, "translator_persona.json")
            components.download_docx_button(
                "Download persona DOCX",
                persona,
                "translator_persona.docx",
                "Translator Persona",
            )
        else:
            st.info("Enable persona generation to view this tab.")

    with tabs[3]:
        st.subheader("Terminology")
        if terminology_choice == "Yes":
            st.success("Terminology extraction delegated to the Termextractor app.")
            if termextractor_url:
                st.markdown(f"[Open Termextractor]({termextractor_url})")
        else:
            st.info("Terminology extraction was skipped for now.")

    with tabs[4]:
        st.subheader("Project manager brief")
        if pm_brief:
            components.render_json(pm_brief)
            components.download_json_button("Download PM brief JSON", pm_brief, "pm_brief.json")
            components.download_docx_button("Download PM brief DOCX", pm_brief, "pm_brief.docx", "PM Brief")
        else:
            st.info("Enable PM brief generation to view this tab.")


if __name__ == "__main__":
    main()
