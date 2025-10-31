"""Build and export project-specific translation style guides."""
from __future__ import annotations

import io
import re
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Sequence

from docx import Document

if TYPE_CHECKING:  # pragma: no cover - typing helpers for editors
    from src.core.llm_client import LLMClient

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
try:  # pragma: no cover - configuration loader
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        SETTINGS = yaml.safe_load(handle)
except FileNotFoundError:  # pragma: no cover - packaging fallback
    SETTINGS = {}

STYLEGUIDE_PROMPT = SETTINGS.get("prompts", {}).get(
    "styleguide_refinement",
    (
        "You are a senior localisation PM. Refine the provided style guide JSON, enriching empty fields "
        "while preserving the structure. Respond with JSON only."
    ),
)

SECTION_TITLES = OrderedDict(
    [
        ("project_overview", "Project Overview"),
        ("content_summary", "Content Summary"),
        ("language_profile", "Language Profile"),
        ("tone_and_voice", "Tone and Voice"),
        ("audience", "Audience"),
        ("terminology", "Terminology"),
        ("do_not_translate", "Do Not Translate"),
        ("formatting", "Formatting"),
        ("ui_copy", "UI Copy"),
        ("units_and_measurements", "Units & Measurements"),
        ("quality_checks", "Quality Checks"),
        ("review_process", "Review Process"),
    ]
)


def _normalise_text(value: str) -> str:
    return value.strip() if value else ""


def _normalise_list(items: Iterable[str]) -> List[str]:
    return sorted({item.strip() for item in items if item and item.strip()})


def _extract_brand_terms(texts: Sequence[str]) -> List[str]:
    """Identify capitalised words that may represent brand names."""

    pattern = re.compile(r"\b[A-Z][A-Z0-9\-]{2,}\b")
    brands: set[str] = set()
    for text in texts:
        if not text:
            continue
        for match in pattern.findall(text):
            brands.add(match)
    return sorted(brands)


def _normalise_terminology(entries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalised: List[Dict[str, Any]] = []
    for entry in entries or []:
        term = _normalise_text(str(entry.get("term", "")))
        translation = _normalise_text(str(entry.get("translation", "")))
        notes = _normalise_text(str(entry.get("notes", "")))
        category = _normalise_text(str(entry.get("category", "")))
        dnt_flag = bool(entry.get("dnt")) or str(entry.get("status", "")).lower() == "dnt"
        if not term:
            continue
        normalised.append(
            {
                "term": term,
                "translation": translation,
                "notes": notes,
                "category": category,
                "dnt": dnt_flag,
            }
        )
    return normalised


def _derive_dnt_terms(terminology: Sequence[Mapping[str, Any]], pm_notes: str) -> List[str]:
    flagged = [entry.get("term", "") for entry in terminology if entry.get("dnt")]
    brands = _extract_brand_terms([pm_notes])
    return _normalise_list(list(flagged) + brands)


def _build_template(
    project_name: str,
    analysis: Mapping[str, Any],
    languages: Mapping[str, Sequence[str]],
    pm_notes: str,
    terminology: List[Dict[str, Any]],
    manual_notes: str,
) -> OrderedDict[str, Any]:
    domain = analysis.get("domain") or "General"
    subdomains = analysis.get("subdomains", []) or []
    related = analysis.get("related", []) or []

    template: OrderedDict[str, Any] = OrderedDict()
    template["project_overview"] = {
        "project_name": project_name,
        "primary_domain": domain,
        "subdomains": subdomains,
        "related_fields": related,
        "pm_notes": pm_notes,
    }
    template["content_summary"] = {
        "summary": analysis.get("summary") or "",  # backwards compatibility if provided
        "manual_notes": manual_notes,
    }
    template["language_profile"] = {
        "source_languages": list(languages.get("sources", [])),
        "target_languages": list(languages.get("targets", [])),
    }
    template["tone_and_voice"] = {
        "default_tone": analysis.get("tone", "Professional"),
        "notes": "Align with brand personality; adjust based on PM notes when present.",
    }
    template["audience"] = {
        "primary_audience": analysis.get("audience", "General"),
        "difficulty_level": analysis.get("difficulty_level"),
    }
    template["terminology"] = {
        "entries": terminology,
        "guidance": "Validate extracted terms with the client. Prefer consistent terminology across assets.",
    }
    template["do_not_translate"] = {
        "terms": _derive_dnt_terms(terminology, pm_notes),
        "instructions": "Preserve brand names, product names, and interface placeholders in source language.",
    }
    template["formatting"] = {
        "numbers_dates": "Follow locale-specific number and date formats. Confirm measurement conversions with PM.",
        "punctuation": "Respect local punctuation rules and spacing before/after symbols.",
        "capitalisation": "Retain source casing for trademarks and UI strings unless instructed otherwise.",
    }
    template["ui_copy"] = {
        "length_constraints": "Stay within UI character limits; favour concise actionable phrasing.",
        "placeholders": "Keep placeholders and tags unchanged. Verify spacing around variables.",
        "buttons_links": "Use imperative verbs and consistent casing across UI elements.",
    }
    template["units_and_measurements"] = {
        "measurement_system": "Use metric/imperial conversions as required for the target locale.",
        "validation": "Double-check figures and units during QA; escalate discrepancies to PM.",
    }
    template["quality_checks"] = {
        "pre_delivery": [
            "Spellcheck and grammar review",
            "Terminology verification against glossary",
            "Cross-check numbers, dates, and placeholders",
        ],
        "tools": "Use QA automation tools when available (Xbench, Verifika, etc.).",
    }
    template["review_process"] = {
        "workflow": [
            "Translator self-review",
            "Peer or LQA review focusing on terminology and tone",
            "PM sign-off",
        ],
        "contacts": "Document PM and reviewer contacts for escalation.",
    }
    return template


def build_styleguide(
    *,
    analysis: Mapping[str, Any],
    pm_notes: str,
    terminology: Sequence[Mapping[str, Any]],
    langs: Mapping[str, Sequence[str]],
    llm_client: "LLMClient" | None,
    project_name: str,
    manual_notes: str = "",
) -> OrderedDict[str, Any]:
    """Combine analysis, notes, and terminology into a structured style guide."""

    normalised_terminology = _normalise_terminology(terminology)
    template = _build_template(project_name, analysis, langs, pm_notes, normalised_terminology, manual_notes)

    if llm_client:
        try:
            prompt = STYLEGUIDE_PROMPT if "{payload}" in STYLEGUIDE_PROMPT else f"{STYLEGUIDE_PROMPT}\n{{payload}}"
            enriched = llm_client.complete_json(prompt, {"styleguide": template})
            if isinstance(enriched, dict):
                for key, value in enriched.items():
                    template[key] = value
        except Exception:  # pragma: no cover - graceful degradation when LLM unavailable
            pass
    return template


def export_styleguide_docx(data: Mapping[str, Any], filename: str) -> bytes:
    """Export the style guide data as a DOCX document and return bytes."""

    document = Document()
    document.add_heading(f"{filename} Translation Style Guide", level=0)
    for section_key, section_title in SECTION_TITLES.items():
        document.add_heading(section_title, level=1)
        section_value = data.get(section_key, {})
        if isinstance(section_value, Mapping):
            for field, value in section_value.items():
                if isinstance(value, list):
                    document.add_paragraph(field.replace("_", " ").title() + ":")
                    for item in value:
                        document.add_paragraph(str(item), style="List Bullet")
                elif isinstance(value, Mapping):
                    document.add_paragraph(field.replace("_", " ").title() + ":")
                    for sub_field, sub_value in value.items():
                        document.add_paragraph(f"{sub_field.title()}: {sub_value}")
                else:
                    text = str(value).strip()
                    if text:
                        document.add_paragraph(f"{field.replace('_', ' ').title()}: {text}")
        elif isinstance(section_value, list):
            for item in section_value:
                document.add_paragraph(str(item), style="List Bullet")
        elif section_value:
            document.add_paragraph(str(section_value))

    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


__all__ = ["build_styleguide", "export_styleguide_docx"]
