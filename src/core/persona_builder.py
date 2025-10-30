"""Builder for LLM-ready virtual translator personas."""
from __future__ import annotations

from typing import Any, Dict


def build_persona(
    analysis: dict,
    style_guide: dict,
    src_lang: str,
    tgt_lang: str,
    *,
    llm_client=None,
    prompt: str | None = None,
) -> Dict[str, Any]:
    """Create a persona specification that can be used as an LLM system prompt."""

    persona = {
        "persona_name": f"Senior {analysis.get('domain', 'General')} Translator",
        "languages": {"source": src_lang, "target": tgt_lang},
        "domain_expertise": analysis.get("domain"),
        "subdomains": analysis.get("subdomains", []),
        "tone_target": style_guide.get("voice_tone", {}).get("preferred"),
        "instructions": (
            "You are an experienced linguist responsible for producing high-quality translations. "
            "Respect the provided style guide, use the approved terminology, and keep placeholders intact."
        ),
        "constraints": [
            "Respect inline tags and placeholders such as {TAG} markers.",
            "Use terminology provided by the Termextractor output when available.",
            "Do not invent product names or modify regulatory statements.",
            "Perform a consistency check before delivering the translation.",
        ],
        "workflow_suggestions": [
            "Review the source text for complex terminology before translating.",
            "Draft translation in a CAT tool supporting bilingual segmentation.",
            "Apply QA checks for terminology, numbers, and placeholders.",
        ],
    }

    if llm_client and prompt:
        enriched = llm_client.complete_json(
            prompt,
            {
                "analysis": analysis,
                "style_guide": style_guide,
                "persona": persona,
            },
        )
        if isinstance(enriched, dict):
            persona = enriched

    return persona


__all__ = ["build_persona"]
