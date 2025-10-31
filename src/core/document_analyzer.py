from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

logger = logging.getLogger(__name__)

GENERAL_DOMAIN = "General / Technical"
VALID_COMPLEXITY_DRIVERS = ("terminology_density", "regulatory_refs", "tables", "formulas")

DOMAIN_KEYWORDS: Dict[str, set[str]] = {
    "Medical / Clinical": {
        "anamnesis",
        "cardiology",
        "clinical",
        "diagnosis",
        "epikriza",
        "hematology",
        "hospital",
        "medical",
        "nephrology",
        "oncology",
        "patient",
        "pharmaceutical",
        "prescription",
        "procedure",
        "prostate",
        "radiology",
        "surgery",
        "therapy",
        "treatment",
        "urology",
    },
    "Environmental / Industrial": {
        "abatement",
        "bref",
        "cement",
        "emission",
        "furnace",
        "industrial",
        "kiln",
        "lime",
        "magnesium",
        "magnesium oxide",
        "mg o",
        "mgo",
        "pollutant",
        "process",
        "stack",
        "waste gas",
    },
    "Legal / Regulatory": {
        "article",
        "compliance",
        "directive",
        "law",
        "legal",
        "paragraph",
        "directive (eu)",
        "regulation (eu)",
        "decree",
        "regulation",
        "section",
        "subsection",
    },
}

DOMAIN_METADATA: Dict[str, Dict[str, List[str]]] = {
    "Medical / Clinical": {
        "default_subdomains": ["Clinical documentation"],
        "default_related": ["Healthcare services", "Patient care"],
        "default_references": ["WHO terminology", "ICD-10", "SNOMED CT"],
    },
    "Environmental / Industrial": {
        "default_subdomains": ["Industrial emissions"],
        "default_related": ["Process engineering", "Environmental compliance"],
        "default_references": ["BREF portal", "EU ETS guidance"],
    },
    "Legal / Regulatory": {
        "default_subdomains": ["Legislative documentation"],
        "default_related": ["Policy", "Compliance"],
        "default_references": ["EUR-Lex", "Official Journal of the EU"],
    },
    GENERAL_DOMAIN: {
        "default_subdomains": ["Technical documentation"],
        "default_related": ["General technology", "Business communication"],
        "default_references": ["IATE", "Client reference material"],
    },
}

RISK_LIBRARY: Dict[str, List[Dict[str, str]]] = {
    "Medical / Clinical": [
        {
            "name": "Clinical terminology accuracy",
            "description": "Ensure diagnoses, procedures, and drug names match established medical terminology.",
            "mitigation": "Engage a subject-matter expert or medically certified linguist for review.",
        },
        {
            "name": "Patient data sensitivity",
            "description": "Protect patient-identifying information and adhere to confidentiality requirements.",
            "mitigation": "Apply data protection guidelines and confirm secure handling with the client.",
        },
    ],
    "Environmental / Industrial": [
        {
            "name": "Process terminology alignment",
            "description": "Industrial process steps and equipment must be translated consistently across the project.",
            "mitigation": "Establish a terminology base and enforce QA using terminology tools.",
        },
        {
            "name": "Emission compliance references",
            "description": "Regulatory citations must be accurate for compliance and audit readiness.",
            "mitigation": "Cross-check references against client-provided compliance documentation.",
        },
    ],
    "Legal / Regulatory": [
        {
            "name": "Legislative citation precision",
            "description": "Directive and regulation identifiers must remain exact to avoid legal issues.",
            "mitigation": "Validate citations on EUR-Lex and confirm with client SMEs.",
        },
        {
            "name": "Terminology consistency",
            "description": "Legal terms must be used consistently across all deliverables.",
            "mitigation": "Maintain a bilingual glossary and apply QA checks before delivery.",
        },
    ],
    GENERAL_DOMAIN: [
        {
            "name": "General QA",
            "description": "Large technical projects risk inconsistent tone and formatting across translators.",
            "mitigation": "Provide unified briefing notes and run bilingual QA before delivery.",
        }
    ],
}

PROMPT_TEMPLATE = (
    "You are assisting a localisation project manager. Analyse the provided document chunk using the"
    " metadata supplied below. Respond strictly in JSON with UTF-8 encoding and double quotes. The"
    " JSON must include: domain (string), subdomains (list of strings), related_fields (list of"
    " strings), regulatory_references (list of strings), complexity_drivers (list of values from"
    " ['terminology_density','regulatory_refs','tables','formulas']), risks (list of objects with"
    " name, description, mitigation), and recommended_references (optional list of strings).\n"
    "Metadata and text to analyse (JSON):\n{payload}"
)

REGULATORY_PATTERNS = [
    re.compile(r"Directive\s+\d{4}/\d{1,4}", re.IGNORECASE),
    re.compile(r"Regulation\s+\(EU\)[^)]+\d{4}", re.IGNORECASE),
    re.compile(r"Decision\s+\(EU\)[^)]+\d{4}", re.IGNORECASE),
    re.compile(r"Article\s+\d+", re.IGNORECASE),
]

UPPERCASE_PATTERN = re.compile(r"\b[A-Z]{3,}\b")
TABLE_PATTERN = re.compile(r"Table\s+\d", re.IGNORECASE)
FORMULA_PATTERN = re.compile(r"\b[A-Z][a-z]?[0-9]{1,3}[A-Z][a-z]?[0-9]{0,3}\b")


def _unique(values: Iterable[Any]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        if not value:
            continue
        normalised = str(value).strip()
        if not normalised or normalised.lower() in seen:
            continue
        seen.add(normalised.lower())
        ordered.append(normalised)
    return ordered


def _keyword_hits(text: str, keywords: Iterable[str]) -> int:
    lower = text.lower()
    return sum(lower.count(str(keyword)) for keyword in keywords)


def guess_domain(full_text: str) -> str:
    if not full_text or not full_text.strip():
        return GENERAL_DOMAIN

    scores = {
        domain: _keyword_hits(full_text, keywords)
        for domain, keywords in DOMAIN_KEYWORDS.items()
    }
    best_domain, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score == 0:
        return GENERAL_DOMAIN
    return best_domain


def _heuristic_features(text: str) -> Dict[str, Any]:
    regulatory_refs: List[str] = []
    for pattern in REGULATORY_PATTERNS:
        regulatory_refs.extend(pattern.findall(text))

    uppercase_terms = UPPERCASE_PATTERN.findall(text)
    tables_present = bool(TABLE_PATTERN.search(text)) or "|" in text or "\t" in text
    formulas_present = bool(FORMULA_PATTERN.search(text))

    drivers: set[str] = set()
    if len(uppercase_terms) >= 10:
        drivers.add("terminology_density")
    if regulatory_refs:
        drivers.add("regulatory_refs")
    if tables_present:
        drivers.add("tables")
    if formulas_present:
        drivers.add("formulas")

    return {
        "regulatory_references": _unique(regulatory_refs),
        "drivers": drivers,
    }


def _normalise_string_list(values: Any) -> List[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    if isinstance(values, (list, tuple, set)):
        return _unique(str(item) for item in values if item)
    return []


def _normalise_risks(risks: Any) -> List[Dict[str, str]]:
    normalised: List[Dict[str, str]] = []
    if not isinstance(risks, Iterable):
        return normalised
    for risk in risks:
        if not isinstance(risk, dict):
            continue
        name = str(risk.get("name", "")).strip()
        description = str(risk.get("description", "")).strip()
        mitigation = str(risk.get("mitigation", "")).strip()
        if name and description and mitigation:
            normalised.append(
                {
                    "name": name,
                    "description": description,
                    "mitigation": mitigation,
                }
            )
    return normalised


def _merge_unique_risks(*risk_groups: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    merged: List[Dict[str, str]] = []
    for group in risk_groups:
        for risk in group or []:
            key = (risk.get("name", ""), risk.get("description", ""), risk.get("mitigation", ""))
            if not all(key):
                continue
            if key in seen:
                continue
            merged.append({"name": key[0], "description": key[1], "mitigation": key[2]})
            seen.add(key)
    return merged


def _fallback_chunk_analysis(
    domain_hint: str,
    regulatory_refs: Sequence[str],
    drivers: Iterable[str],
) -> Dict[str, Any]:
    metadata = DOMAIN_METADATA.get(domain_hint, DOMAIN_METADATA[GENERAL_DOMAIN])
    default_risks = RISK_LIBRARY.get(domain_hint, RISK_LIBRARY[GENERAL_DOMAIN])
    return {
        "domain": domain_hint,
        "subdomains": metadata["default_subdomains"],
        "related_fields": metadata["default_related"],
        "regulatory_references": _unique(regulatory_refs),
        "complexity_drivers": _unique(drivers),
        "risks": [dict(risk) for risk in default_risks],
        "references": metadata["default_references"],
    }


def _normalise_chunk_result(result: Dict[str, Any], domain_hint: str) -> Dict[str, Any]:
    domain_value = str(result.get("domain", "")).strip() or domain_hint
    subdomains = _normalise_string_list(result.get("subdomains"))
    related = _normalise_string_list(result.get("related_fields"))
    regulatory_refs = _normalise_string_list(result.get("regulatory_references"))
    drivers = [driver for driver in _normalise_string_list(result.get("complexity_drivers")) if driver in VALID_COMPLEXITY_DRIVERS]
    risks = _normalise_risks(result.get("risks"))
    references = _normalise_string_list(result.get("recommended_references"))
    return {
        "domain": domain_value,
        "subdomains": subdomains,
        "related_fields": related,
        "regulatory_references": regulatory_refs,
        "complexity_drivers": drivers,
        "risks": risks,
        "references": references,
    }


def _analyse_chunk_with_llm(
    llm_client,
    payload: Dict[str, Any],
    domain_hint: str,
    heuristics: Dict[str, Any],
) -> tuple[Dict[str, Any], bool]:
    llm_result: Dict[str, Any] | None = None
    if llm_client is not None:
        try:
            llm_result = llm_client.complete_json(PROMPT_TEMPLATE, payload)
        except Exception as exc:  # pragma: no cover - depends on external service
            logger.warning("LLM analysis failed; falling back to template: %s", exc)
            llm_result = None
    if not isinstance(llm_result, dict) or not llm_result:
        fallback = _fallback_chunk_analysis(
            domain_hint,
            heuristics.get("regulatory_references", []),
            heuristics.get("drivers", []),
        )
        return fallback, True
    return _normalise_chunk_result(llm_result, domain_hint), False


def _most_common_or_default(counter: Counter[str], default: Sequence[str]) -> List[str]:
    if counter:
        return _unique(item for item, _ in counter.most_common())
    return list(default)


def _determine_complexity(word_estimate: int, drivers: Sequence[str]) -> Dict[str, Any]:
    level = 1
    if word_estimate > 1500:
        level += 1
    if word_estimate > 4000:
        level += 1
    if word_estimate > 8000:
        level += 1
    level += min(len(list(drivers)), 2)
    level = max(1, min(level, 5))
    ordered_drivers = [driver for driver in VALID_COMPLEXITY_DRIVERS if driver in set(drivers)]
    return {"level": level, "drivers": ordered_drivers}


def _build_translator_profile(domain: str, drivers: Sequence[str]) -> Dict[str, Any]:
    domain_lower = domain.lower()
    driver_set = set(drivers)
    tools = ["memoQ", "Trados Studio", "Xbench", "Verifika"]

    if "medical" in domain_lower or "clinical" in domain_lower:
        background = "Certified medical translator or healthcare professional with localisation expertise."
        experience = "Experience handling clinical documentation, discharge summaries, and patient records."
        tools.append("Multiterm")
    elif "environmental" in domain_lower or "industrial" in domain_lower:
        background = "Environmental or process engineer with strong localisation background."
        experience = "Prior work on industrial emissions, environmental compliance, and technical process documentation."
    elif "legal" in domain_lower:
        background = "Legal translator specialising in EU and regulatory materials."
        experience = "Experience translating directives, regulations, and compliance frameworks."
    else:
        background = "Senior technical translator comfortable with complex subject matter."
        experience = "Demonstrated history working on technical, business, or IT documentation."

    if "terminology_density" in driver_set:
        experience += " Able to manage dense terminology and maintain glossary alignment."
    if "regulatory_refs" in driver_set:
        experience += " Capable of validating legislative and standards references."

    return {
        "required_background": background,
        "preferred_experience": experience,
        "tools": _unique(tools),
    }


def _build_system_notes(
    final_domain: str,
    heuristic_domain: str,
    fallback_count: int,
    chunk_count: int,
    pm_notes: str | None,
) -> str:
    notes: List[str] = []
    if pm_notes:
        notes.append("PM instructions supplied; ensure they are reflected in downstream deliverables.")
    if final_domain:
        notes.append(f"Domain determined from document content: {final_domain}.")
    if heuristic_domain and heuristic_domain != final_domain:
        notes.append(
            f"Heuristic keyword guess ({heuristic_domain}) differed from the merged analysis; validate with the client."
        )
    if fallback_count:
        notes.append(
            f"{fallback_count} of {chunk_count} chunk(s) used a fallback template because no LLM response was available."
        )
    notes.append("Review the classification before assigning linguists.")
    return "\n".join(notes)


def analyze_chunks_hybrid(
    chunks: Sequence[str],
    src_lang: str | None,
    tgt_langs: Sequence[str] | None,
    pm_notes: str | None,
    llm_client,
    *,
    project_name: str | None = None,
    full_text: str | None = None,
    file_count: int | None = None,
) -> Dict[str, Any]:
    """Analyse document chunks using LLM results with heuristic fallbacks."""

    chunks = [chunk for chunk in chunks if chunk]
    combined_text = full_text if full_text is not None else "\n".join(chunks)
    domain_guess = guess_domain(combined_text)

    chunk_results: List[Dict[str, Any]] = []
    fallback_counter = 0
    for chunk in chunks:
        heuristics = _heuristic_features(chunk)
        payload = {
            "chunk": chunk,
            "domain_hint": domain_guess,
            "source_language": src_lang or "",
            "target_languages": list(tgt_langs or []),
            "pm_notes": pm_notes or "",
            "regulatory_reference_hints": heuristics.get("regulatory_references", []),
            "complexity_driver_hints": list(heuristics.get("drivers", [])),
        }
        chunk_analysis, used_template = _analyse_chunk_with_llm(llm_client, payload, domain_guess, heuristics)
        if used_template:
            fallback_counter += 1
        chunk_results.append(chunk_analysis)

    domain_counter: Counter[str] = Counter()
    subdomain_counter: Counter[str] = Counter()
    related_counter: Counter[str] = Counter()
    regulatory_counter: Counter[str] = Counter()
    driver_counter: Counter[str] = Counter()
    reference_counter: Counter[str] = Counter()
    risk_aggregate: List[Dict[str, str]] = []

    for result in chunk_results:
        domain_value = result.get("domain")
        if domain_value:
            domain_counter[str(domain_value)] += 1
        for subdomain in result.get("subdomains", []):
            subdomain_counter[str(subdomain)] += 1
        for related in result.get("related_fields", []):
            related_counter[str(related)] += 1
        for ref in result.get("regulatory_references", []):
            regulatory_counter[str(ref)] += 1
        for driver in result.get("complexity_drivers", []):
            if driver in VALID_COMPLEXITY_DRIVERS:
                driver_counter[driver] += 1
        references = _normalise_string_list(result.get("references", []))
        reference_counter.update(references)
        risk_aggregate = _merge_unique_risks(risk_aggregate, result.get("risks", []))

    final_domain = domain_counter.most_common(1)[0][0] if domain_counter else domain_guess
    metadata_defaults = DOMAIN_METADATA.get(final_domain, DOMAIN_METADATA[GENERAL_DOMAIN])

    final_subdomains = _most_common_or_default(subdomain_counter, metadata_defaults["default_subdomains"])
    final_related = _most_common_or_default(related_counter, metadata_defaults["default_related"])
    final_reg_refs = _unique([item for item, _ in regulatory_counter.most_common()])

    final_drivers = [driver for driver in VALID_COMPLEXITY_DRIVERS if driver_counter[driver] > 0]
    if not final_drivers:
        final_drivers = metadata_defaults.get("default_drivers", [])

    final_references = _unique(
        [item for item, _ in reference_counter.most_common()]
        + metadata_defaults["default_references"]
        + final_reg_refs
    )

    if not risk_aggregate:
        risk_aggregate = [dict(risk) for risk in RISK_LIBRARY.get(final_domain, RISK_LIBRARY[GENERAL_DOMAIN])]

    word_estimate = len(re.findall(r"\w+", combined_text))
    complexity = _determine_complexity(word_estimate, final_drivers)
    translator_profile = _build_translator_profile(final_domain, complexity["drivers"])

    chunk_count = len(chunks)
    analysis: Dict[str, Any] = {
        "document_metadata": {
            "project_name": project_name or "",
            "source_language": (src_lang or "").strip(),
            "target_languages": _unique(str(lang).strip() for lang in (tgt_langs or [])),
            "file_count": file_count if file_count is not None else chunk_count,
            "word_estimate": word_estimate,
        },
        "classification": {
            "domain": final_domain,
            "subdomains": final_subdomains,
            "related_fields": final_related,
        },
        "complexity": complexity,
        "translator_profile": translator_profile,
        "risks": risk_aggregate,
        "references": final_references,
        "pm_notes": {
            "original": pm_notes or "",
            "system_notes": _build_system_notes(
                final_domain,
                domain_guess,
                fallback_counter,
                chunk_count,
                pm_notes,
            ),
        },
    }

    return analysis


__all__ = ["analyze_chunks_hybrid", "guess_domain"]
