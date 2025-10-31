"""Document analysis module for domain, tone, audience, and complexity signals."""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from src.core.llm_client import LLMClient

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
try:  # pragma: no cover - configuration loader
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        SETTINGS = yaml.safe_load(handle)
except FileNotFoundError:  # pragma: no cover - fallback for packaging
    SETTINGS = {}

DOMAIN_PROMPT = SETTINGS.get("prompts", {}).get(
    "domain_classification",
    (
        "Analyze the following text and determine main domain, subdomain, and related technical fields. "
        "Respond with JSON using keys 'domain', 'subdomains', and 'related'. Text:\n{payload}"
    ),
)

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


def _heuristic_classification(text: str) -> dict:
    tokens = [token.lower() for token in tokenize(text)]
    token_counts = Counter(tokens)
    domain = _detect_domain(token_counts)
    subdomains = _detect_subdomains(domain, token_counts)
    related = [item for item, _ in Counter(tokens).most_common(10) if len(item) > 5][:5]
    return {"domain": domain, "subdomains": subdomains, "related": related}


def classify_domain(chunks: Iterable[str], client: "LLMClient" | None) -> dict[str, list[str] | str]:
    """Classify domain, subdomains, and related fields from text chunks."""

    prompt = DOMAIN_PROMPT
    results: list[dict] = []
    for chunk in chunks:
        chunk_text = (chunk or "").strip()
        if not chunk_text:
            continue
        if client:
            try:
                response = client.complete_json(prompt, {"text": chunk_text})
            except Exception:  # pragma: no cover - fallback to heuristic
                response = _heuristic_classification(chunk_text)
        else:
            response = _heuristic_classification(chunk_text)
        if isinstance(response, dict):
            results.append(response)

    if not results:
        return {"domain": "General", "subdomains": [], "related": []}

    domain_counts: Counter[str] = Counter()
    subdomain_counts: Counter[str] = Counter()
    related_counts: Counter[str] = Counter()

    for result in results:
        domain = result.get("domain")
        if isinstance(domain, str):
            domain_counts[domain] += 1
        for subdomain in result.get("subdomains", []) or []:
            if isinstance(subdomain, str):
                subdomain_counts[subdomain] += 1
        for related in result.get("related", []) or []:
            if isinstance(related, str):
                related_counts[related] += 1

    primary_domain = domain_counts.most_common(1)[0][0] if domain_counts else "General"
    ranked_subdomains = [item for item, _ in subdomain_counts.most_common(5)]
    ranked_related = [item for item, _ in related_counts.most_common(5)]

    return {
        "domain": primary_domain,
        "subdomains": ranked_subdomains,
        "related": ranked_related,
    }


__all__ = ["analyze_document", "classify_domain"]
