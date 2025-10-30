"""Generate project manager briefs from analysis data."""
from __future__ import annotations

from typing import Any, Dict, List


RISK_THRESHOLD = {
    "term_density": 0.02,
    "abbreviation_density": 0.02,
    "sentence_length": 25,
}


def build_pm_brief(
    analysis: dict,
    terms: List[dict],
    *,
    llm_client=None,
    prompt: str | None = None,
) -> Dict[str, Any]:
    """Construct a project manager briefing document."""

    signals = analysis.get("complexity_signals", {})
    difficulty_level = analysis.get("difficulty_level")
    domain = analysis.get("domain")
    term_density = signals.get("technical_term_density", 0)
    abbreviation_density = signals.get("abbreviation_density", 0)
    avg_sentence_len = signals.get("avg_sentence_len", 0)

    risk_flags = []
    if term_density > RISK_THRESHOLD["term_density"]:
        risk_flags.append("High terminology density; confirm glossary coverage.")
    if abbreviation_density > RISK_THRESHOLD["abbreviation_density"]:
        risk_flags.append("Frequent abbreviations; request expanded forms from client.")
    if avg_sentence_len > RISK_THRESHOLD["sentence_length"]:
        risk_flags.append("Long sentences; consider additional QA for readability.")
    if domain in {"Legal", "Medical", "Finance"}:
        risk_flags.append("Regulatory content: ensure compliance review.")

    prerequisites = [
        "Confirm availability of reference materials and previous translations.",
        "Request non-translatable lists and branding guidelines.",
        "Align on CAT tool package requirements (memoQ, Trados, etc.).",
    ]

    linguist_profile = {
        "domain": domain,
        "experience": f">= 5 years in {', '.join(analysis.get('subdomains', []) or [domain or 'target domain'])}",
        "tools": ["memoQ", "Trados", "Phrase"],
        "mtpe_experience": domain in {"Software", "Marketing"},
    }

    prerequisites.extend(
        ["Terminology list received" if terms else "Request terminology extraction before kickoff."]
    )

    brief = {
        "difficulty_level": difficulty_level,
        "difficulty_rationale": {
            "avg_sentence_len": avg_sentence_len,
            "term_density": term_density,
            "abbreviation_density": abbreviation_density,
        },
        "human_linguist_profile": linguist_profile,
        "prerequisites": prerequisites,
        "pm_focus_points": [
            "Confirm unit conventions (metric/imperial) with client before launch.",
            "Validate proper nouns and abbreviation expansions with stakeholders.",
            "Ensure inline tags are locked/protected in the CAT tool.",
        ],
        "risk_flags": risk_flags,
        "suggested_steps": [
            "Schedule SME review for critical terminology.",
            "Plan LQA pass focusing on tone and regulatory statements.",
            "Configure terminology QA checks in the CAT tool.",
        ],
    }

    if llm_client and prompt:
        enriched = llm_client.complete_json(prompt, {"analysis": analysis, "brief": brief})
        if isinstance(enriched, dict):
            brief = enriched

    return brief


__all__ = ["build_pm_brief"]
