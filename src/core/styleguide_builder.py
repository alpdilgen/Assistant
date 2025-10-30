"""Generate draft style guides for translation projects."""
from __future__ import annotations

from typing import Any, Dict, List


DEFAULT_TONE_RULES = {
    "formal": "Use professional and precise language. Avoid colloquialisms.",
    "neutral": "Keep sentences concise and informative with minimal flair.",
    "marketing": "Highlight benefits and persuasive messaging while staying truthful.",
    "technical": "Prioritise clarity, include steps and parameter names exactly as provided.",
}


def build_style_guide(
    analysis: dict,
    terms: List[dict],
    src_lang: str,
    tgt_lang: str,
    *,
    llm_client=None,
    prompt: str | None = None,
) -> Dict[str, Any]:
    """Build a structured style guide from the analysis and terminology."""

    tone = analysis.get("tone", "neutral")
    tone_rules = DEFAULT_TONE_RULES.get(tone, DEFAULT_TONE_RULES["neutral"])

    terminology_entries = [
        {
            "term": term.get("term"),
            "part_of_speech": term.get("pos"),
            "preferred_translation": term.get("translation"),
            "notes": term.get("note"),
        }
        for term in terms
        if term.get("term")
    ]

    base_style = {
        "languages": {"source": src_lang, "target": tgt_lang},
        "voice_tone": {
            "preferred": tone,
            "guidance": tone_rules,
        },
        "terminology": terminology_entries,
        "formatting": {
            "numbers": "Use locale-appropriate decimal separators and preserve units.",
            "dates": "Follow ISO 8601 unless client specifies otherwise.",
            "capitalisation": "Keep brand names and UI labels as in source; sentence case for headings unless stated.",
        },
        "do_dont": {
            "do": [
                "Respect inline tags and placeholders.",
                "Cross-check technical terminology against provided list.",
                "Maintain consistency of tone with domain expectations.",
            ],
            "dont": [
                "Do not invent new terminology.",
                "Avoid literal translation of idioms that reduce clarity.",
                "Do not modify measurement units without confirmation.",
            ],
        },
        "client_specific": {
            "brand_voice": "",
            "non_translatables": "",
            "reference_materials": "",
        },
    }

    if llm_client and prompt:
        enriched = llm_client.complete_json(
            prompt,
            {
                "analysis": analysis,
                "style_guide": base_style,
            },
        )
        if isinstance(enriched, dict):
            base_style = enriched

    return base_style


__all__ = ["build_style_guide"]
