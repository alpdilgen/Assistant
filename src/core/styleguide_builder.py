"""Style guide assembly informed by PM inputs and automated analysis."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional

LANGUAGE_CONVENTIONS: dict[str, dict[str, str]] = {
    "ro": {
        "dates": "Use DD.MM.YYYY and include Romanian diacritics (ă, â, î, ș, ț) consistently.",
        "numbers": "Use comma as decimal separator and period as thousands separator. Non-breaking space before currency (e.g. 150 RON).",
        "address": "Prefer polite plural forms (dumneavoastră) for formal communication.",
        "quotation": "Use Romanian quotation marks („“), fallback to double quotes when UI constraints apply.",
    },
    "de": {
        "dates": "Use DD.MM.YYYY and keep nouns capitalised.",
        "numbers": "Comma as decimal separator and period for thousands (1.000,50).",
        "address": "Use formal Sie unless specified otherwise.",
        "quotation": "Use „…“ or »…« depending on client preference.",
    },
    "fr": {
        "dates": "Use DD/MM/YYYY and insert non-breaking spaces before ; : ? !",
        "numbers": "Comma decimal separator, space for thousands (1 000,50).",
        "address": "Use vouvoiement unless instructed otherwise.",
        "quotation": "Use guillemets (« … »).",
    },
    "es": {
        "dates": "Use DD/MM/YYYY and spell out months when space allows.",
        "numbers": "Comma decimal separator and period for thousands.",
        "address": "Use formal usted for professional communication unless persona demands otherwise.",
        "quotation": "Use «…» or “…” depending on platform.",
    },
    "it": {
        "dates": "Use DD/MM/YYYY and Italian month names when expanded.",
        "numbers": "Comma decimal separator, period for thousands.",
        "address": "Use formal Lei for hospitality and professional contexts.",
        "quotation": "Use «…» for long text and double quotes in UI.",
    },
    "ru": {
        "dates": "Use DD.MM.YYYY and genitive month names when spelling out.",
        "numbers": "Comma decimal separator, space for thousands.",
        "address": "Use formal Вы for B2B communication.",
        "quotation": "Use «…» with inner „…“.",
    },
    "uk": {
        "dates": "Use DD.MM.YYYY and include apostrophe in year abbreviations.",
        "numbers": "Comma decimal separator, space for thousands.",
        "address": "Use formal Ви with capital letter.",
        "quotation": "Use «…» or “…” depending on medium.",
    },
    "tr": {
        "dates": "Use DD.MM.YYYY and month names in lowercase.",
        "numbers": "Comma decimal separator, period for thousands.",
        "address": "Use polite second person plural (Siz) in formal contexts.",
        "quotation": "Use double quotes for UI and «…» for long form.",
    },
    "ar": {
        "dates": "Use DD/MM/YYYY and Arabic month names when possible.",
        "numbers": "Use Eastern Arabic numerals when required by client; otherwise Arabic digits with comma decimal separator.",
        "address": "Maintain formal address and gender agreement.",
        "quotation": "Use «…» respecting right-to-left layout.",
    },
    "zh-cn": {
        "dates": "Use YYYY年M月D日 format or YYYY-MM-DD in UI.",
        "numbers": "Use Arabic numerals with comma separators; maintain units after numbers.",
        "address": "Keep polite tone and use full-width punctuation where appropriate.",
        "quotation": "Use full-width Chinese quotes （“……”） or 「……」 depending on platform.",
    },
    "ja": {
        "dates": "Use YYYY年M月D日; use era notation if client requests.",
        "numbers": "Use Arabic numerals with comma separators and include full-width characters in vertical text.",
        "address": "Use polite language (です・ます調) unless casual tone specified.",
        "quotation": "Use Japanese corner brackets 「…」 or 『…』.",
    },
    "ko": {
        "dates": "Use YYYY년 M월 D일 or YYYY.MM.DD for UI.",
        "numbers": "Use comma as thousands separator; place currency before amount (₩50,000).",
        "address": "Use polite form (~습니다) unless instructed otherwise.",
        "quotation": "Use Korean quotes 「…」 / 『…』 or double quotes depending on platform.",
    },
    "no": {
        "dates": "Use DD.MM.YYYY and capitalise weekdays.",
        "numbers": "Use comma as decimal separator and space for thousands.",
        "address": "Use formal De in official contexts, otherwise du for informal communications.",
        "quotation": "Use «…» or double quotes depending on medium.",
    },
    "sr": {
        "dates": "Use DD.MM.YYYY and specify Cyrillic/Latin requirements.",
        "numbers": "Comma decimal separator and period for thousands.",
        "address": "Clarify script preference; default to formal Ви.",
        "quotation": "Use „…“ for Cyrillic and “...” for Latin.",
    },
    "bs": {
        "dates": "Use DD.MM.YYYY and specify whether Latin script is required.",
        "numbers": "Comma decimal separator, period for thousands.",
        "address": "Use formal Vi in professional contexts.",
        "quotation": "Use „…“ or “...” depending on platform.",
    },
}

DEFAULT_LANGUAGE_GUIDANCE = {
    "dates": "Follow local conventions for dates (prefer numeric format unless client specifies otherwise).",
    "numbers": "Use locale-appropriate separators and keep measurement units consistent with the source.",
    "address": "Adopt the standard level of formality for business communication in the target locale.",
    "quotation": "Use native quotation marks where supported; otherwise fall back to double quotes.",
}


def _language_guidance(language_code: str) -> dict[str, str]:
    return LANGUAGE_CONVENTIONS.get(language_code.lower(), DEFAULT_LANGUAGE_GUIDANCE)


def _base_template(
    project_name: str,
    src_lang: str,
    tgt_lang: str,
    domains: Iterable[str],
    document_type: Optional[str] = None,
) -> "OrderedDict[str, Any]":
    domain_list = [domain for domain in domains if domain]
    domain_summary = ", ".join(domain_list) if domain_list else "General"
    template = OrderedDict(
        {
            "general_information": {
                "project_name": project_name or "",
                "languages": {"source": src_lang, "target": tgt_lang},
                "domain_focus": domain_summary,
                "document_type": document_type or "",
                "notes": (
                    "Structure mirrors the Translation Style Guide Questionnaire used for hospitality/spa projects. "
                    "Update each section as client feedback arrives."
                ),
            },
            "references": {
                "existing_glossaries": "",
                "reference_materials": "",
                "style_samples": "",
            },
            "tone": {
                "desired_voice": "",
                "audience": "",
                "examples": [],
            },
            "terminology": {
                "sources": [],
                "preferred_terms": [],
                "forbidden_terms": [],
                "notes": "",
            },
            "formatting": {
                "numbers": "",
                "dates": "",
                "capitalisation": "Maintain source casing for brand names; otherwise sentence case unless client specifies.",
                "punctuation": "",
            },
            "ui_rules": {
                "placeholder_handling": "Preserve placeholders and tags exactly as in source (e.g. {TAG}).",
                "length_constraints": "Keep translations within UI character limits; prefer concise wording.",
                "button_copy": "Use actionable verbs; capitalisation follows platform norms.",
            },
            "do_not_translate": {
                "list": [],
                "validation": "Confirm with PM before removing any candidate from the DNT list.",
            },
            "units_and_measurements": {
                "measurement_system": "",
                "conversion_rules": "Clarify whether metric to imperial conversions are required.",
                "numeric_checks": "Run QA for numbers and units before delivery.",
            },
            "branding": {
                "key_messages": "",
                "taglines": [],
                "third_party_brands": "",
            },
            "locale_conventions": {
                "formality": "",
                "address_formats": "",
                "quotation_marks": "",
                "additional_notes": "",
            },
            "review_process": {
                "review_steps": [
                    "Linguist self-QA",
                    "Peer or LQA review focusing on terminology and tone",
                    "PM sign-off",
                ],
                "sign_off": "",
            },
            "language_specific": {
                "orthography": "",
                "proofing_focus": "",
                "common_mistakes": [],
            },
        }
    )
    return template


def _normalise_list_field(value: str) -> list[str]:
    if not value:
        return []
    separators = [",", "\n", ";"]
    working = value
    for separator in separators:
        working = working.replace(separator, "\n")
    items = [item.strip() for item in working.splitlines() if item.strip()]
    return items


def _apply_pm_inputs(
    template: "OrderedDict[str, Any]",
    pm_inputs: Dict[str, str],
    *,
    language_guidance: dict[str, str],
) -> None:
    general = template["general_information"]
    if pm_inputs.get("company_name"):
        general["project_name"] = pm_inputs["company_name"].strip()

    references = template["references"]
    if pm_inputs.get("existing_glossaries"):
        references["existing_glossaries"] = pm_inputs["existing_glossaries"].strip()

    tone_section = template["tone"]
    if pm_inputs.get("tone_voice"):
        tone_section["desired_voice"] = pm_inputs["tone_voice"].strip()

    terminology = template["terminology"]
    if pm_inputs.get("existing_glossaries"):
        terminology_sources = terminology.get("sources", [])
        terminology_sources.insert(0, f"Client-provided glossaries: {pm_inputs['existing_glossaries'].strip()}")
        terminology["sources"] = terminology_sources

    dnt_section = template["do_not_translate"]
    if pm_inputs.get("dnt_list"):
        dnt_section["list"] = _normalise_list_field(pm_inputs["dnt_list"])

    formatting = template["formatting"]
    if pm_inputs.get("number_rules"):
        rule = pm_inputs["number_rules"].strip()
        formatting["numbers"] = rule
        formatting["dates"] = rule or language_guidance.get("dates", "")

    ui_rules = template["ui_rules"]
    if pm_inputs.get("ui_rules"):
        ui_rules["platform_specific"] = pm_inputs["ui_rules"].strip()

    review = template["review_process"]
    if pm_inputs.get("sign_off_person"):
        review["sign_off"] = pm_inputs["sign_off_person"].strip()

    locale = template["locale_conventions"]
    locale.setdefault("additional_notes", "")
    locale.setdefault("formality", "")
    locale.setdefault("address_formats", "")
    locale.setdefault("quotation_marks", "")
    locale["formality"] = language_guidance.get("address", "")
    locale["quotation_marks"] = language_guidance.get("quotation", "")
    locale["additional_notes"] = pm_inputs.get("additional_notes", "").strip() if pm_inputs.get("additional_notes") else locale["additional_notes"]


def _auto_populate(
    template: "OrderedDict[str, Any]",
    analysis: dict,
    terms: Optional[List[dict]],
    language_guidance: dict[str, str],
    project_name: str,
) -> None:
    general = template["general_information"]
    summary = analysis.get("summary") or analysis.get("combined_summary") or ""
    if project_name and not general.get("project_name"):
        general["project_name"] = project_name
    general["project_overview"] = summary

    audience = analysis.get("audience", "general")
    tone = analysis.get("tone", "neutral")
    domains = analysis.get("domains") or [analysis.get("domain")]

    references = template["references"]
    references["reference_materials"] = (
        "Leverage any hospitality/spa collateral that reflects the detected domains: "
        f"{', '.join(domains or ['General'])}."
    )
    references["style_samples"] = "Use previous launch material or marketing collateral as tone benchmarks when available."

    tone_section = template["tone"]
    tone_section["desired_voice"] = f"{tone.title()} tone anchored in {', '.join(domains or ['general'])}."
    tone_section["audience"] = f"Primary audience: {audience}."
    tone_section["examples"] = [
        "Keep messaging warm and service-oriented, mirroring the hospitality/spa sample questionnaire.",
        "Highlight benefits and wellbeing outcomes for guests.",
    ]

    terminology = template["terminology"]
    terminology_sources = [
        "Use external Termextractor output for baseline terminology.",
        "Incorporate existing hospitality/spa glossaries when provided by PM.",
    ]
    terminology["sources"] = terminology_sources
    preferred_terms = []
    if terms:
        for entry in terms:
            if isinstance(entry, dict) and entry.get("term"):
                preferred_terms.append(
                    {
                        "term": entry.get("term"),
                        "preferred_translation": entry.get("translation"),
                        "notes": entry.get("note"),
                    }
                )
    terminology["preferred_terms"] = preferred_terms
    terminology["notes"] = "If terminology extraction is pending, schedule a follow-up with the PM once the Termextractor export is ready."

    formatting = template["formatting"]
    formatting["numbers"] = language_guidance.get("numbers", DEFAULT_LANGUAGE_GUIDANCE["numbers"])
    formatting["dates"] = language_guidance.get("dates", DEFAULT_LANGUAGE_GUIDANCE["dates"])
    formatting["punctuation"] = language_guidance.get("quotation", DEFAULT_LANGUAGE_GUIDANCE["quotation"])

    ui_rules = template["ui_rules"]
    ui_rules.setdefault("platform_specific", "")
    ui_rules["platform_specific"] = "Replicate the sample questionnaire order: navigation labels, booking flow, amenity descriptions."

    dnt_section = template["do_not_translate"]
    named_entities = []
    signals = analysis.get("complexity_signals", {})
    if signals:
        named_entities = signals.get("named_entities", [])
    dnt_section["list"] = named_entities
    if not named_entities:
        dnt_section["list"] = ["Client brand names", "Spa package names", "Room categories"]

    units = template["units_and_measurements"]
    numeric_density = signals.get("numeric_density") if signals else None
    units["measurement_system"] = "Use metric units unless client specifies imperial." if (numeric_density is None or numeric_density >= 0) else units["measurement_system"]
    units["conversion_rules"] = "Mirror the sample questionnaire: keep temperatures in °C and treatment durations in minutes."

    branding = template["branding"]
    branding["key_messages"] = "Focus on relaxation, premium guest experience, and wellbeing outcomes."
    branding["taglines"] = ["Rejuvenate with us", "Wellness tailored to every guest"]
    branding["third_party_brands"] = "Retain partner spa product names in source language unless instructed otherwise."

    locale = template["locale_conventions"]
    locale["formality"] = language_guidance.get("address", DEFAULT_LANGUAGE_GUIDANCE["address"])
    locale["address_formats"] = "Follow national postal standards when addresses appear in content."
    locale["quotation_marks"] = language_guidance.get("quotation", DEFAULT_LANGUAGE_GUIDANCE["quotation"])
    locale["additional_notes"] = "Use locale-specific diacritics and ensure currency placement reflects local conventions."

    review = template["review_process"]
    review["sign_off"] = "PM or client brand manager"

    language_specific = template["language_specific"]
    language_specific["orthography"] = language_guidance.get("address", DEFAULT_LANGUAGE_GUIDANCE["address"])
    language_specific["proofing_focus"] = "Ensure spa terminology and wellness-specific benefits remain consistent across documents."
    language_specific["common_mistakes"] = [
        "Dropping diacritics in the target language",
        "Literal translation of wellness package names",
        "Overly casual tone for premium hospitality brand",
    ]


def build_style_guide(
    analysis: dict,
    src_lang: str,
    tgt_lang: str,
    *,
    terms: Optional[List[dict]] = None,
    pm_inputs: Optional[Dict[str, str]] = None,
    project_name: str | None = None,
    llm_client=None,
    prompt: str | None = None,
) -> Dict[str, Any]:
    """Build a style guide following the hospitality questionnaire structure."""

    domains = analysis.get("domains") or [analysis.get("domain")]
    document_type = analysis.get("document_type")
    template = _base_template(project_name or "", src_lang, tgt_lang, domains, document_type=document_type)
    language_guidance = _language_guidance(tgt_lang)

    if pm_inputs:
        _auto_populate(template, analysis, terms, language_guidance, project_name or "")
        _apply_pm_inputs(template, pm_inputs, language_guidance=language_guidance)
    else:
        _auto_populate(template, analysis, terms, language_guidance, project_name or "")

    style_guide: Dict[str, Any] = dict(template)

    if llm_client and prompt:
        enriched = llm_client.complete_json(
            prompt,
            {
                "analysis": analysis,
                "style_guide": style_guide,
                "pm_inputs": pm_inputs or {},
            },
        )
        if isinstance(enriched, dict):
            style_guide = enriched

    return style_guide


__all__ = ["build_style_guide"]
