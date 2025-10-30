"""Document analysis module for domain, tone, audience, and complexity signals."""
from __future__ import annotations

import math
import re
from collections import Counter
from statistics import mean
from typing import Iterable, Optional

from .terminology_client import get_terminology  # re-exported for convenience

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
WORD_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)
ABBREVIATION_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
NUMERIC_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")

DOMAIN_KEYWORDS = {
    "Legal": {"hereby", "whereas", "herein", "plaintiff", "defendant", "compliance"},
    "Medical": {"patient", "diagnosis", "treatment", "clinical", "dose", "symptom"},
    "Automotive": {"engine", "diagnostic", "transmission", "torque", "vehicle", "sensor"},
    "Software": {"login", "interface", "deployment", "API", "configuration", "database"},
    "Marketing": {"campaign", "brand", "engagement", "audience", "conversion"},
    "Finance": {"equity", "portfolio", "investment", "liability", "asset"},
}

AUDIENCE_KEYWORDS = {
    "technicians": {"diagnostic", "torque", "sensor", "firmware", "circuit"},
    "end users": {"click", "download", "setup", "account", "support"},
    "management": {"strategy", "kpi", "budget", "growth", "revenue"},
    "regulators": {"compliance", "certification", "standards", "regulation"},
}

TONE_KEYWORDS = {
    "formal": {"hereby", "therefore", "pursuant", "shall"},
    "neutral": {"step", "note", "information", "update"},
    "marketing": {"exciting", "discover", "experience", "exclusive"},
    "technical": {"parameter", "configuration", "protocol"},
}


def tokenize(text: str) -> list[str]:
    return WORD_PATTERN.findall(text)


def _detect_domain(token_counts: Counter[str]) -> str:
    scores: dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(token_counts.get(keyword.lower(), 0) for keyword in keywords)
    if not scores:
        return "General"
    winner, value = max(scores.items(), key=lambda kv: kv[1])
    return winner if value > 0 else "General"


def _detect_subdomains(domain: str, token_counts: Counter[str]) -> list[str]:
    if domain == "Automotive":
        options = {
            "Diagnostics": {"obd", "fault", "sensor"},
            "Software UI": {"interface", "button", "screen"},
            "Manufacturing": {"assembly", "plant", "production"},
        }
    elif domain == "Medical":
        options = {
            "Pharma": {"dosage", "tablet", "injection"},
            "Clinical": {"patient", "trial", "symptom"},
            "Medical devices": {"device", "implant", "sterile"},
        }
    elif domain == "Legal":
        options = {
            "Contracts": {"agreement", "party", "liability"},
            "Compliance": {"regulation", "compliance", "audit"},
        }
    else:
        options = {
            "General": set(),
        }
    ranked = []
    for name, keywords in options.items():
        score = sum(token_counts.get(word.lower(), 0) for word in keywords)
        if score:
            ranked.append((name, score))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in ranked[:3]] or (["General"] if domain == "General" else [])


def _detect_tone(token_counts: Counter[str]) -> str:
    tone_scores: dict[str, int] = {}
    for tone, keywords in TONE_KEYWORDS.items():
        tone_scores[tone] = sum(token_counts.get(keyword.lower(), 0) for keyword in keywords)
    if not tone_scores:
        return "neutral"
    tone, score = max(tone_scores.items(), key=lambda kv: kv[1])
    return tone if score else "neutral"


def _detect_audience(token_counts: Counter[str]) -> str:
    audience_scores: dict[str, int] = {}
    for audience, keywords in AUDIENCE_KEYWORDS.items():
        audience_scores[audience] = sum(token_counts.get(keyword.lower(), 0) for keyword in keywords)
    if not audience_scores:
        return "general"
    audience, score = max(audience_scores.items(), key=lambda kv: kv[1])
    return audience if score else "general"


def _sentence_lengths(text: str) -> list[int]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    lengths = []
    for sentence in sentences:
        if sentence:
            lengths.append(len(tokenize(sentence)))
    return lengths


def _complexity_signals(text: str, tokens: list[str], terms: Optional[list[dict]] = None) -> dict:
    sentence_lengths = _sentence_lengths(text)
    avg_sentence_len = mean(sentence_lengths) if sentence_lengths else 0
    token_count = len(tokens) or 1
    technical_terms = {term["term"] for term in terms or [] if isinstance(term, dict) and term.get("term")}
    technical_term_density = len(technical_terms) / token_count
    abbreviations = ABBREVIATION_PATTERN.findall(text)
    abbreviation_density = len(abbreviations) / token_count
    named_entities = sorted({token for token in tokens if token.istitle() and len(token) > 3})[:20]

    numeric_tokens = NUMERIC_PATTERN.findall(text)
    numeric_density = len(numeric_tokens) / token_count

    return {
        "avg_sentence_len": round(avg_sentence_len, 2),
        "technical_term_density": round(technical_term_density, 4),
        "abbreviation_density": round(abbreviation_density, 4),
        "numeric_density": round(numeric_density, 4),
        "named_entities": named_entities,
        "terminology_count": len(technical_terms),
    }


def _difficulty_score(signals: dict) -> int:
    sentence_factor = min(signals.get("avg_sentence_len", 0) / 20, 1)
    term_factor = min(signals.get("technical_term_density", 0) * 200, 1)
    abbreviation_factor = min(signals.get("abbreviation_density", 0) * 200, 1)
    numeric_factor = min(signals.get("numeric_density", 0) * 100, 1)

    raw_score = 3 + 4 * sentence_factor + 2 * term_factor + abbreviation_factor + numeric_factor
    bounded = max(1, min(10, int(round(raw_score))))
    return bounded


def analyze_document(
    text: str,
    *,
    terms: Optional[list[dict]] = None,
    use_llm_summary: bool = False,
    llm_client=None,
    prompt: Optional[str] = None,
) -> dict:
    """Analyse the document content and produce structured metadata."""

    tokens = [token.lower() for token in tokenize(text)]
    token_counts = Counter(tokens)

    domain = _detect_domain(token_counts)
    subdomains = _detect_subdomains(domain, token_counts)
    tone = _detect_tone(token_counts)
    audience = _detect_audience(token_counts)
    signals = _complexity_signals(text, tokens, terms=terms)
    difficulty_level = _difficulty_score(signals)

    if use_llm_summary and llm_client and prompt:
        summary = llm_client.complete_json(prompt, {"text": text, "signals": signals})
        summary_text = summary.get("summary", "") if isinstance(summary, dict) else ""
    else:
        summary_text = _heuristic_summary(text)

    return {
        "summary": summary_text,
        "domain": domain,
        "subdomains": subdomains,
        "difficulty_level": difficulty_level,
        "tone": tone,
        "audience": audience,
        "complexity_signals": signals,
    }


def _heuristic_summary(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    first_sentences = sentences[:3]
    summary = " ".join(first_sentences)
    return summary[:700]


__all__ = ["analyze_document"]
