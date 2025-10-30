from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
import yaml

from src.core.document_ingestion import load_documents
from src.core.document_analysis import analyze_document
from src.core.styleguide_builder import build_style_guide
from src.core.persona_builder import build_persona
from src.core.pm_brief import build_pm_brief
from src.core.terminology_client import get_terminology
from src.core.llm_client import LLMClient
from src.ui import components

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


@st.cache_resource
def load_settings() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _language_codes(settings: Dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    options = settings.get("languages", [])
    labels = [item.get("label") for item in options]
    mapping = {item.get("label"): item.get("code") for item in options}
    return labels, mapping


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


def main() -> None:
    st.set_page_config(page_title="LSP Assistant", layout="wide")
    settings = load_settings()

    language_labels, language_mapping = _language_codes(settings)

    with st.sidebar:
        st.title("Project Settings")
        src_label = st.selectbox("Source language", options=language_labels, index=0)
        tgt_label = st.selectbox("Target language", options=language_labels, index=1 if len(language_labels) > 1 else 0)
        document_type = st.selectbox("Document type", options=settings.get("document_types", []))
        use_alignment = st.checkbox("Use bilingual alignment", value=True)
        call_terminology = st.checkbox("Call external terminology service", value=True)
        generate_style_guide = st.checkbox("Generate style guide", value=True)
        generate_persona = st.checkbox("Generate translator persona", value=True)
        generate_pm_brief = st.checkbox("Generate PM brief", value=True)
        enable_llm = st.checkbox("Use LLM prompts when available", value=settings.get("llm", {}).get("enabled", False))

    llm_client = _build_llm_client(settings, enable_llm)

    st.header("AI-Powered Localization Assistant")
    st.caption("Upload source documents to analyse, build style guides, define translator personas, and brief project managers.")

    uploaded_files = components.file_uploader()
    if not uploaded_files:
        st.info("Upload at least one document to begin.")
        return

    uploads: List[tuple[str, bytes]] = [(file.name, file.getvalue()) for file in uploaded_files]
    payloads = load_documents(
        uploads,
        default_source_lang=language_mapping.get(src_label),
        default_target_lang=language_mapping.get(tgt_label),
    )

    if not payloads:
        st.error("No supported documents were uploaded.")
        return

    combined_text = "\n\n".join(payload.raw_text for payload in payloads if payload.raw_text)

    terms: List[dict] = []
    if call_terminology:
        with st.spinner("Fetching terminology from external service..."):
            for (filename, file_bytes), payload in zip(uploads, payloads):
                try:
                    term_list = get_terminology(
                        file_bytes,
                        filename,
                        payload.source_language or language_mapping.get(src_label) or "",
                        payload.target_language or language_mapping.get(tgt_label) or "",
                    )
                except Exception as exc:  # pragma: no cover - external
                    st.warning(f"Terminology extraction failed for {filename}: {exc}")
                    term_list = []
                terms.extend(term_list)
        # Deduplicate by term text
        unique_terms = {}
        for term in terms:
            key = term.get("term") or json.dumps(term, sort_keys=True)
            unique_terms[key] = term
        terms = list(unique_terms.values())

    analysis_prompt = settings.get("prompts", {}).get("analysis") if llm_client else None
    analysis = analyze_document(
        combined_text,
        terms=terms,
        use_llm_summary=bool(llm_client and analysis_prompt),
        llm_client=llm_client,
        prompt=analysis_prompt,
    )
    analysis["document_type"] = document_type

    style_guide = None
    if generate_style_guide:
        style_prompt = settings.get("prompts", {}).get("style_guide") if llm_client else None
        style_guide = build_style_guide(
            analysis,
            terms,
            language_mapping.get(src_label) or src_label,
            language_mapping.get(tgt_label) or tgt_label,
            llm_client=llm_client,
            prompt=style_prompt,
        )

    persona = None
    if generate_persona:
        persona_prompt = settings.get("prompts", {}).get("persona") if llm_client else None
        persona = build_persona(
            analysis,
            style_guide or {},
            language_mapping.get(src_label) or src_label,
            language_mapping.get(tgt_label) or tgt_label,
            llm_client=llm_client,
            prompt=persona_prompt,
        )

    pm_brief = None
    if generate_pm_brief:
        pm_prompt = settings.get("prompts", {}).get("pm_brief") if llm_client else None
        pm_brief = build_pm_brief(
            analysis,
            terms,
            llm_client=llm_client,
            prompt=pm_prompt,
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
        components.render_json(analysis)
        components.download_json_button("Download analysis JSON", analysis, "analysis.json")

    with tabs[1]:
        st.subheader("Style guide")
        if style_guide:
            components.render_json(style_guide)
            components.download_json_button("Download style guide JSON", style_guide, "style_guide.json")
            components.download_docx_button("Download style guide DOCX", style_guide, "style_guide.docx", "Translation Style Guide")
        else:
            st.info("Enable style guide generation in the sidebar to view this tab.")

    with tabs[2]:
        st.subheader("Translator persona")
        if persona:
            components.render_json(persona)
            components.download_json_button("Download persona JSON", persona, "translator_persona.json")
        else:
            st.info("Enable persona generation in the sidebar to view this tab.")

    with tabs[3]:
        st.subheader("Terminology")
        if terms:
            st.dataframe(terms)
            components.download_json_button("Download terminology JSON", {"terms": terms}, "terminology.json")
        else:
            st.info("Terminology extraction disabled or no terms found.")
        if use_alignment:
            segmented_payloads = [payload for payload in payloads if payload.segments]
            if segmented_payloads:
                st.markdown("### Bilingual segments preview")
                for payload in segmented_payloads:
                    st.markdown(f"**{payload.filename}**")
                    st.dataframe(payload.segments[:20])

    with tabs[4]:
        st.subheader("Project manager brief")
        if pm_brief:
            components.render_json(pm_brief)
            components.download_json_button("Download PM brief JSON", pm_brief, "pm_brief.json")
            components.download_docx_button("Download PM brief DOCX", pm_brief, "pm_brief.docx", "PM Brief")
        else:
            st.info("Enable PM brief generation in the sidebar to view this tab.")


if __name__ == "__main__":
    main()
