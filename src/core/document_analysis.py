"""Document analysis module for domain, tone, audience, and complexity signals."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from src.core.llm_client import LLMClient

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
try:  # pragma: no cover - configuration loader
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        SETTINGS = yaml.safe_load(handle)
except FileNotFoundError:  # pragma: no cover - fallback for packaging
    SETTINGS = {}

ANALYSIS_PROMPT = SETTINGS.get(
    "prompts", {}
).get(
    "document_chunk_analysis",
    (
        "You are a senior localisation strategist. Use the payload JSON to analyse translation material, "
        "considering any PM notes. Respond strictly in JSON with keys: domain (string), subdomains (list), "
        "related (list), primary_purpose (string), target_audience (string), tone (string), register (string), "
        "keywords (list of strings), technical_symbols (list of strings), difficulty_level (int 1-10), "
        "summary (string).\n{payload}"
    ),
)

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
WORD_PATTERN = re.compile(r"[\w'-]+", re.UNICODE)
ABBREVIATION_PATTERN = re.compile(r"\b[A-Z]{2,}\b")
NUMERIC_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")
CHEMICAL_PATTERN = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")

TECHNICAL_TOKEN_CANDIDATES = {
    "°C",
    "°F",
    "°K",
    "m³",
    "Nm³",
    "mg/Nm³",
    "mg/m³",
    "μg/m³",
    "ppm",
    "kWh",
    "MW",
    "CO₂",
    "NOx",
    "SO₂",
    "pH",
    "Hz",
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "from",
    "your",
    "have",
    "this",
    "will",
    "into",
    "about",
    "their",
    "when",
    "shall",
    "there",
    "which",
    "while",
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


DOMAIN_KEYWORDS = {
    "Legal": {"hereby", "whereas", "herein", "plaintiff", "defendant", "compliance"},
    "Medical": {"patient", "diagnosis", "treatment", "clinical", "dose", "symptom"},
    "Automotive": {"engine", "diagnostic", "transmission", "torque", "vehicle", "sensor"},
    "Software": {"login", "interface", "deployment", "api", "configuration", "database"},
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


PURPOSE_BY_DOMAIN = {
    "Legal": "Ensure legal compliance and unambiguous contractual communication.",
    "Medical": "Convey clinical and patient information safely and accurately.",
    "Automotive": "Provide technical servicing or diagnostics guidance for vehicles.",
    "Software": "Explain product functionality and guide users through digital workflows.",
    "Marketing": "Persuade the audience to engage with the product or campaign.",
    "Finance": "Report financial performance and provide regulatory disclosures.",
}


REGISTER_BY_TONE = {
    "formal": "formal business register",
    "neutral": "neutral professional",
    "marketing": "persuasive marketing",
    "technical": "technical and instructional",
}


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
        options = {"General": set()}
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


def _detect_technical_symbols(text: str) -> list[str]:
    found = {token for token in TECHNICAL_TOKEN_CANDIDATES if token in text}
    found.update(CHEMICAL_PATTERN.findall(text))
    return sorted(found)


def _complexity_signals(text: str, tokens: list[str], terms: Optional[list[dict]] = None) -> dict:
    sentence_lengths = _sentence_lengths(text)
    avg_sentence_len = mean(sentence_lengths) if sentence_lengths else 0
    token_count = len(tokens) or 1
    technical_terms = {
        term["term"]
        for term in terms or []
        if isinstance(term, dict) and term.get("term")
    }
    technical_term_density = len(technical_terms) / token_count
    abbreviations = ABBREVIATION_PATTERN.findall(text)
    abbreviation_density = len(abbreviations) / token_count
    named_entities = sorted({token for token in tokens if token.istitle() and len(token) > 3})[:20]
    numeric_tokens = NUMERIC_PATTERN.findall(text)
    numeric_density = len(numeric_tokens) / token_count
    technical_symbols = _detect_technical_symbols(text)

    return {
        "avg_sentence_len": round(avg_sentence_len, 2),
        "technical_term_density": round(technical_term_density, 4),
        "abbreviation_density": round(abbreviation_density, 4),
        "numeric_density": round(numeric_density, 4),
        "named_entities": named_entities,
        "terminology_count": len(technical_terms),
        "technical_symbols": technical_symbols,
    }


def _difficulty_score(signals: dict) -> int:
    sentence_factor = min(signals.get("avg_sentence_len", 0) / 20, 1)
    term_factor = min(signals.get("technical_term_density", 0) * 200, 1)
    abbreviation_factor = min(signals.get("abbreviation_density", 0) * 200, 1)
    numeric_factor = min(signals.get("numeric_density", 0) * 100, 1)

    raw_score = 3 + 4 * sentence_factor + 2 * term_factor + abbreviation_factor + numeric_factor
    bounded = max(1, min(10, int(round(raw_score))))
    return bounded


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


def _top_keywords(tokens: Sequence[str], limit: int = 10) -> list[str]:
    cleaned = [token.lower() for token in tokens if len(token) > 3 and not token.isdigit()]
    cleaned = [token for token in cleaned if token not in STOPWORDS]
    counts = Counter(cleaned)
    return [token for token, _ in counts.most_common(limit)]


def _infer_purpose(domain: str) -> str:
    return PURPOSE_BY_DOMAIN.get(domain, "Provide clear, accurate information tailored to the end audience.")


def _infer_register(tone: str) -> str:
    return REGISTER_BY_TONE.get(tone, "neutral professional")


def analyze_document(
    text: str,
    *,
    terms: Optional[list[dict]] = None,
    use_llm_summary: bool = False,
    llm_client: "LLMClient" | None = None,
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
    keywords = _top_keywords(tokens)
    related = _heuristic_classification(text).get("related", [])

    if use_llm_summary and llm_client and prompt:
        try:
            summary = llm_client.complete_json(prompt, {"text": text, "signals": signals})
            summary_text = summary.get("summary", "") if isinstance(summary, dict) else ""
        except Exception:  # pragma: no cover - degrade gracefully if model unavailable
            summary_text = _heuristic_summary(text)
    else:
        summary_text = _heuristic_summary(text)

    return {
        "summary": summary_text,
        "domain": domain,
        "subdomains": subdomains,
        "related": related,
        "difficulty_level": difficulty_level,
        "tone": tone,
        "audience": audience,
        "purpose": _infer_purpose(domain),
        "register": _infer_register(tone),
        "signals": signals,
        "keywords": keywords,
    }


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            if isinstance(item, str) and item.strip():
                result.append(item.strip())
        return result
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def build_analysis_prompt(chunk: str, pm_notes: str | None = None) -> str:
    """Return the LLM prompt used for chunk-level document analysis."""

    _ = pm_notes  # Included for signature parity and future customisation
    return ANALYSIS_PROMPT


def _normalise_partial_result(raw: Mapping[str, Any], chunk: str) -> dict:
    domain = str(raw.get("domain") or raw.get("primary_domain") or "General")
    subdomains = _ensure_list(raw.get("subdomains"))
    related = _ensure_list(raw.get("related"))
    tone = str(raw.get("tone") or raw.get("voice") or "neutral")
    audience = str(raw.get("audience") or raw.get("target_audience") or "general")
    purpose = str(raw.get("primary_purpose") or raw.get("purpose") or _infer_purpose(domain))
    register = str(raw.get("register") or raw.get("style") or _infer_register(tone))
    difficulty = int(raw.get("difficulty_level") or raw.get("difficulty") or 0)
    keywords = _ensure_list(raw.get("keywords"))
    technical_symbols = set(_ensure_list(raw.get("technical_symbols")))
    summary = str(raw.get("summary") or _heuristic_summary(chunk))

    signals = raw.get("signals") or {}
    avg_sentence_len = float(signals.get("avg_sentence_len") or raw.get("avg_sentence_length") or 0)
    abbreviation_density = float(signals.get("abbreviation_density") or raw.get("abbreviation_density") or 0)
    numeric_density = float(signals.get("numeric_density") or raw.get("numeric_density") or 0)
    terminology_density = float(signals.get("technical_term_density") or raw.get("terminology_density") or 0)
    terminology_count = int(signals.get("terminology_count") or raw.get("terminology_count") or 0)
    named_entities = set(signals.get("named_entities") or raw.get("named_entities") or [])
    technical_symbols.update(signals.get("technical_symbols", []))

    if not keywords:
        keywords = _top_keywords(tokenize(chunk))

    return {
        "domain": domain,
        "subdomains": subdomains,
        "related": related,
        "tone": tone,
        "audience": audience,
        "purpose": purpose,
        "register": register,
        "difficulty_level": difficulty,
        "keywords": keywords,
        "technical_symbols": sorted({symbol for symbol in technical_symbols if symbol}),
        "summary": summary,
        "avg_sentence_len": avg_sentence_len,
        "abbreviation_density": abbreviation_density,
        "numeric_density": numeric_density,
        "terminology_density": terminology_density,
        "terminology_count": terminology_count,
        "named_entities": sorted(named_entities),
    }


def merge_partial_analyses(partials: Sequence[Mapping[str, Any]]) -> dict:
    """Combine chunk-level analyses into a single project-wide summary."""

    if not partials:
        return {
            "domain": "General",
            "subdomains": [],
            "related": [],
            "primary_purpose": _infer_purpose("General"),
            "audience": "general",
            "tone": "neutral",
            "register": _infer_register("neutral"),
            "keywords": [],
            "technical_symbols": [],
            "chunk_count": 0,
            "avg_sentence_length": 0.0,
            "abbreviation_density": 0.0,
            "numeric_density": 0.0,
            "terminology_density": 0.0,
            "terminology_count": 0,
            "difficulty_level": 1,
            "summary": "",
            "named_entities": [],
        }

    domain_counts: Counter[str] = Counter()
    subdomain_counts: Counter[str] = Counter()
    related_counts: Counter[str] = Counter()
    tone_counts: Counter[str] = Counter()
    audience_counts: Counter[str] = Counter()
    purpose_counts: Counter[str] = Counter()
    register_counts: Counter[str] = Counter()
    keyword_counts: Counter[str] = Counter()

    technical_symbols: set[str] = set()
    named_entities: set[str] = set()
    avg_sentence_values: list[float] = []
    abbreviation_values: list[float] = []
    numeric_values: list[float] = []
    terminology_density_values: list[float] = []
    terminology_counts: list[int] = []
    difficulty_values: list[int] = []
    summary_parts: list[str] = []

    for partial in partials:
        domain_counts[partial.get("domain", "General")] += 1
        for subdomain in partial.get("subdomains", []):
            subdomain_counts[subdomain] += 1
        for related in partial.get("related", []):
            related_counts[related] += 1
        tone_counts[partial.get("tone", "neutral")] += 1
        audience_counts[partial.get("audience", "general")] += 1
        purpose_counts[partial.get("purpose", _infer_purpose(partial.get("domain", "General")))] += 1
        register_counts[partial.get("register", _infer_register(partial.get("tone", "neutral")))] += 1
        keyword_counts.update(partial.get("keywords", []))
        technical_symbols.update(partial.get("technical_symbols", []))
        named_entities.update(partial.get("named_entities", []))

        avg_sentence_values.append(float(partial.get("avg_sentence_len", 0)))
        abbreviation_values.append(float(partial.get("abbreviation_density", 0)))
        numeric_values.append(float(partial.get("numeric_density", 0)))
        terminology_density_values.append(float(partial.get("terminology_density", 0)))
        terminology_counts.append(int(partial.get("terminology_count", 0)))
        difficulty_values.append(int(partial.get("difficulty_level", 0)))

        summary = partial.get("summary")
        if summary:
            summary_parts.append(summary)

    primary_domain = domain_counts.most_common(1)[0][0]
    primary_purpose = purpose_counts.most_common(1)[0][0]
    primary_tone = tone_counts.most_common(1)[0][0]
    primary_register = register_counts.most_common(1)[0][0]
    primary_audience = audience_counts.most_common(1)[0][0]

    summary_text = " ".join(summary_parts)
    summary_text = summary_text[:1200]

    return {
        "domain": primary_domain,
        "subdomains": [item for item, _ in subdomain_counts.most_common(5)],
        "related": [item for item, _ in related_counts.most_common(5)],
        "primary_purpose": primary_purpose,
        "audience": primary_audience,
        "tone": primary_tone,
        "register": primary_register,
        "keywords": [item for item, _ in keyword_counts.most_common(12)],
        "technical_symbols": sorted(technical_symbols),
        "chunk_count": len(partials),
        "avg_sentence_length": round(mean(avg_sentence_values), 2) if avg_sentence_values else 0.0,
        "abbreviation_density": round(mean(abbreviation_values), 4) if abbreviation_values else 0.0,
        "numeric_density": round(mean(numeric_values), 4) if numeric_values else 0.0,
        "terminology_density": round(mean(terminology_density_values), 4)
        if terminology_density_values
        else 0.0,
        "terminology_count": int(round(mean(terminology_counts))) if terminology_counts else 0,
        "difficulty_level": max(1, min(10, int(round(mean(difficulty_values))))) if difficulty_values else 1,
        "summary": summary_text,
        "named_entities": sorted(named_entities)[:25],
    }


def analyze_document_chunks(
    chunks: Sequence[str],
    llm_client: "LLMClient" | None,
    pm_notes: str | None = None,
) -> dict:
    """Run chunked analysis using the configured LLM client or heuristic fallbacks."""

    partials: list[dict] = []
    prompt = build_analysis_prompt("", pm_notes)
    for chunk in chunks:
        chunk_text = (chunk or "").strip()
        if not chunk_text:
            continue
        result: Mapping[str, Any]
        if llm_client:
            try:
                result = llm_client.complete_json(
                    prompt,
                    {"text": chunk_text, "pm_notes": pm_notes or ""},
                )
            except Exception:  # pragma: no cover - degrade gracefully
                result = analyze_document(chunk_text)
        else:
            result = analyze_document(chunk_text)
        partials.append(_normalise_partial_result(result, chunk_text))
    return merge_partial_analyses(partials)


def classify_domain(chunks: Iterable[str], client: "LLMClient" | None) -> dict[str, list[str] | str]:
    """Classify domain, subdomains, and related fields from text chunks."""

    chunk_list = [chunk for chunk in chunks if chunk and chunk.strip()]
    analysis = analyze_document_chunks(chunk_list, client, pm_notes=None)
    return {
        "domain": analysis.get("domain", "General"),
        "subdomains": analysis.get("subdomains", []),
        "related": analysis.get("related", []),
    }


__all__ = [
    "analyze_document",
    "build_analysis_prompt",
    "merge_partial_analyses",
    "analyze_document_chunks",
    "classify_domain",
]
