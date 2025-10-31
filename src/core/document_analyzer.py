from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

BREF_KEYWORDS = {
    "bref",
    "best available techniques",
    "cement",
    "clinker",
    "kiln",
    "lime",
    "magnesium oxide",
    "mgo",
}

DOMAIN_LIBRARY = [
    {
        "domain": "Environmental / Industrial processes",
        "subdomain": "Industrial Emissions (BAT)",
        "related": ["Environmental compliance", "Process engineering"],
        "keywords": [
            "emission",
            "abatement",
            "best available technique",
            "industrial emissions",
            "stack",
            "kiln",
            "dust collector",
            "sinter",
            "waste gas",
        ],
    },
    {
        "domain": "Legal / Regulatory",
        "subdomain": "EU legislation",
        "related": ["Policy", "Compliance", "Contracts"],
        "keywords": [
            "directive",
            "regulation",
            "article",
            "paragraph",
            "commission",
            "annex",
        ],
    },
    {
        "domain": "Technical / Engineering",
        "subdomain": "Process engineering",
        "related": ["Manufacturing", "Automation"],
        "keywords": [
            "process",
            "installation",
            "equipment",
            "maintenance",
            "heat exchanger",
            "filter",
            "operation",
        ],
    },
]

REGULATORY_PATTERNS = [
    r"Directive\s+\d{4}/\d{1,4}",
    r"Regulation\s+\(EU\)[^)]+\d{4}",
    r"Decision\s+\(EU\)[^)]+\d{4}",
    r"ISO\s?\d{3,5}",
    r"EN\s?\d{3,5}",
]

CHEMICAL_SYMBOLS = {
    "so2",
    "nox",
    "co2",
    "cao",
    "mgo",
    "hcl",
    "hf",
    "nh3",
    "h2o",
    "o2",
}

DRIVER_ORDER = ["terminology_density", "regulatory_refs", "tables", "formulas"]


def _unique(sequence: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for item in sequence:
        if not item:
            continue
        normalised = item.strip()
        if normalised and normalised not in seen:
            ordered.append(normalised)
            seen.add(normalised)
    return ordered


def _analyse_chunk_low_level(text: str) -> Dict[str, Any]:
    lower_text = text.lower()

    domains: List[str] = []
    subdomains: List[str] = []
    related: List[str] = []
    for entry in DOMAIN_LIBRARY:
        if any(keyword in lower_text for keyword in entry["keywords"]):
            domains.append(entry["domain"])
            subdomains.append(entry["subdomain"])
            related.extend(entry["related"])

    bref_hits = sum(1 for keyword in BREF_KEYWORDS if keyword in lower_text)
    if bref_hits:
        domains.append("Environmental / Industrial processes")
        subdomains.append("Cement, Lime and MgO")
        related.extend(["Industrial emissions", "Environmental compliance"])

    regulatory_refs: List[str] = []
    for pattern in REGULATORY_PATTERNS:
        regulatory_refs.extend(re.findall(pattern, text, flags=re.IGNORECASE))

    uppercase_terms = re.findall(r"\b[A-Z]{3,}\b", text)
    tables_present = bool(re.search(r"Table\s+\d", text, flags=re.IGNORECASE)) or "|" in text or "\t" in text
    formulas_present = bool(re.search(r"\b[A-Z][a-z]?\d{1,3}[A-Z][a-z]?\d{0,3}\b", text))
    chemical_hits = {symbol for symbol in CHEMICAL_SYMBOLS if symbol in lower_text}

    drivers: set[str] = set()
    if len(uppercase_terms) >= 10:
        drivers.add("terminology_density")
    if regulatory_refs:
        drivers.add("regulatory_refs")
    if tables_present:
        drivers.add("tables")
    if formulas_present or chemical_hits:
        drivers.add("formulas")

    return {
        "domains": domains,
        "subdomains": subdomains,
        "related": related,
        "regulatory_refs": _unique(regulatory_refs),
        "drivers": drivers,
        "chemicals": chemical_hits,
        "bref_hits": bref_hits,
    }


def _run_llm_analysis(llm_client, chunk: str) -> Dict[str, Any]:
    if llm_client is None:
        return {}
    prompt = (
        "You are assisting a localisation project manager. Analyse the provided document chunk and respond "
        "with JSON containing these keys: domain (string), subdomain (string), related_fields (list of strings), "
        "regulatory_references (list of strings), complexity_drivers (list of values drawn from ['terminology_density', "
        "'regulatory_refs', 'tables', 'formulas']), and risks (list of objects with name, description, mitigation).\n"
        "Chunk: {payload}"
    )
    try:
        result = llm_client.complete_json(prompt, {"chunk": chunk})
    except Exception:  # pragma: no cover - depends on external service
        return {}
    return result or {}


def _merge_drivers(*driver_sets: Iterable[str]) -> List[str]:
    aggregated: List[str] = []
    for drivers in driver_sets:
        for driver in drivers or []:
            if driver in DRIVER_ORDER and driver not in aggregated:
                aggregated.append(driver)
    return aggregated


def _determine_complexity(word_estimate: int, drivers: Iterable[str]) -> Dict[str, Any]:
    driver_list = _merge_drivers(drivers)
    driver_set = set(driver_list)

    level = 1
    if word_estimate > 1500:
        level += 1
    if word_estimate > 4000:
        level += 1
    level += len(driver_set)
    level = max(1, min(5, level))

    ordered_drivers = [driver for driver in DRIVER_ORDER if driver in driver_set]
    return {"level": level, "drivers": ordered_drivers}


def _build_translator_profile(domain: str, drivers: Iterable[str]) -> Dict[str, Any]:
    domain_lower = (domain or "").lower()
    drivers_set = set(drivers or [])
    if "environmental" in domain_lower or "industrial" in domain_lower:
        background = "Environmental or process engineer with strong localisation background."
        experience = "Prior work on EU BREF and heavy-industry environmental compliance projects."
    elif "legal" in domain_lower:
        background = "Legal translator specialising in EU regulatory frameworks."
        experience = "Experience translating directives, regulations, and compliance documentation."
    else:
        background = "Technical translator comfortable with complex process documentation."
        experience = "Experience with manufacturing or engineering materials at EU institutions."

    if "terminology_density" in drivers_set:
        experience += " Able to maintain dense terminology consistency across large projects."
    if "regulatory_refs" in drivers_set:
        experience += " Familiarity with legislative citation standards is required."

    tools = ["memoQ", "Trados", "Xbench", "Verifika"]
    return {
        "required_background": background,
        "preferred_experience": experience,
        "tools": tools,
    }


def _build_risks(drivers: Iterable[str], regulatory_refs: Sequence[str]) -> List[Dict[str, str]]:
    driver_set = set(drivers or [])
    risks: List[Dict[str, str]] = []

    if "terminology_density" in driver_set:
        risks.append(
            {
                "name": "Terminology consistency",
                "description": "Dense specialised terminology requires consistent treatment across all files.",
                "mitigation": "Establish approved glossary entries early and enforce QA with terminology QA tools.",
            }
        )
    if "regulatory_refs" in driver_set or regulatory_refs:
        risks.append(
            {
                "name": "Regulatory references",
                "description": "EU legislation and standards must remain accurate and up to date.",
                "mitigation": "Double-check citations against EUR-Lex and client references before delivery.",
            }
        )
    if "tables" in driver_set:
        risks.append(
            {
                "name": "Structured content",
                "description": "Tables and structured data increase formatting and QA complexity.",
                "mitigation": "Align with CAT tool table handling and run final layout QA in the delivery format.",
            }
        )
    if "formulas" in driver_set:
        risks.append(
            {
                "name": "Chemical and numeric precision",
                "description": "Chemical formulas and measurements leave no room for transcription errors.",
                "mitigation": "Have a subject-matter reviewer verify formulas and numeric units before sign-off.",
            }
        )

    if not risks:
        risks.append(
            {
                "name": "General QA",
                "description": "Large technical projects risk inconsistent tone and formatting across translators.",
                "mitigation": "Provide unified briefing notes and run bilingual QA before delivery.",
            }
        )
    return risks


def _system_notes(domain: str, bref_hits: int, pm_notes: str | None) -> str:
    notes: List[str] = []
    if pm_notes:
        notes.append("PM instructions provided; ensure they are reflected in the translator brief and style guide.")
    if bref_hits >= 2:
        notes.append(
            "Detected repeated BREF/Cement references. Treat the project as Environmental / Industrial processes → Cement, Lime, MgO."
        )
    if not domain:
        notes.append("Domain determined heuristically; validate classification before distribution to linguists.")
    return "\n".join(notes)


def analyze_chunks_hybrid(
    chunks: List[str],
    src_lang: str,
    tgt_langs: List[str],
    pm_notes: str | None,
    llm_client,
) -> Dict[str, Any]:
    """Combine heuristic signals with optional LLM feedback to produce a fixed-schema analysis."""

    combined_text = "\n".join(chunks or [])
    word_estimate = len(re.findall(r"\w+", combined_text))

    domain_votes: Counter[str] = Counter()
    subdomain_votes: Counter[str] = Counter()
    related_fields_counter: Counter[str] = Counter()
    regulatory_refs_counter: Counter[str] = Counter()
    driver_counter: Counter[str] = Counter()
    chemical_mentions: set[str] = set()
    bref_chunk_hits = 0
    llm_risks: List[Dict[str, str]] = []

    for chunk in chunks:
        low_level = _analyse_chunk_low_level(chunk)
        for domain in low_level["domains"]:
            domain_votes[domain] += 1
        for subdomain in low_level["subdomains"]:
            subdomain_votes[subdomain] += 1
        for related in low_level["related"]:
            related_fields_counter[related] += 1
        for ref in low_level["regulatory_refs"]:
            regulatory_refs_counter[ref] += 1
        for driver in low_level["drivers"]:
            driver_counter[driver] += 1
        chemical_mentions.update(low_level["chemicals"])
        if low_level["bref_hits"]:
            bref_chunk_hits += 1

        llm_result = _run_llm_analysis(llm_client, chunk)
        if llm_result:
            domain = llm_result.get("domain")
            subdomain = llm_result.get("subdomain")
            if domain:
                domain_votes[str(domain)] += 1
            if subdomain:
                subdomain_votes[str(subdomain)] += 1
            for related in llm_result.get("related_fields", []) or []:
                if isinstance(related, str):
                    related_fields_counter[str(related)] += 1
            for ref in llm_result.get("regulatory_references", []) or []:
                if isinstance(ref, str):
                    regulatory_refs_counter[str(ref)] += 1
            for driver in llm_result.get("complexity_drivers", []) or []:
                if isinstance(driver, str):
                    driver_counter[str(driver)] += 1
            for risk in llm_result.get("risks", []) or []:
                if isinstance(risk, dict):
                    name = str(risk.get("name", ""))
                    description = str(risk.get("description", ""))
                    mitigation = str(risk.get("mitigation", ""))
                    if name and description and mitigation:
                        llm_risks.append({
                            "name": name,
                            "description": description,
                            "mitigation": mitigation,
                        })

    if bref_chunk_hits >= 2:
        domain_value = "Environmental / Industrial processes → Cement, Lime, MgO"
        subdomains = ["Cement, Lime and MgO"]
        related_fields_counter.update({"Industrial emissions": 2, "Environmental compliance": 2})
    else:
        domain_value = domain_votes.most_common(1)[0][0] if domain_votes else "Technical / Engineering"
        subdomains = [item for item, _ in subdomain_votes.most_common()]
        if not subdomains:
            subdomains = ["Process engineering"] if "technical" in domain_value.lower() else []

    related_fields = [item for item, _ in related_fields_counter.most_common()]
    regulatory_refs = [item for item, _ in regulatory_refs_counter.most_common()]

    drivers = [driver for driver, count in driver_counter.items() if count > 0]
    complexity = _determine_complexity(word_estimate, drivers)

    profile = _build_translator_profile(domain_value, complexity["drivers"])

    combined_risks = _build_risks(complexity["drivers"], regulatory_refs)
    if llm_risks:
        combined_risks.extend(llm_risks)
        combined_risks = [dict(t) for t in {tuple(sorted(risk.items())): risk for risk in combined_risks}.values()]

    references = _unique(
        [
            "IATE",
            "EUR-Lex",
            "BREF (Cement, Lime and Magnesium Oxide)",
            *regulatory_refs,
            *(symbol.upper() for symbol in chemical_mentions),
        ]
    )

    pm_section = {
        "original": pm_notes or "",
        "system_notes": _system_notes(domain_value, bref_chunk_hits, pm_notes),
    }

    analysis: Dict[str, Any] = {
        "document_metadata": {
            "project_name": "",
            "source_language": (src_lang or "").strip().lower(),
            "target_languages": _unique((lang or "").strip().lower() for lang in tgt_langs or []),
            "file_count": len(chunks),
            "word_estimate": word_estimate,
        },
        "classification": {
            "domain": domain_value,
            "subdomains": _unique(subdomains),
            "related_fields": _unique(related_fields),
        },
        "complexity": complexity,
        "translator_profile": profile,
        "risks": combined_risks,
        "references": references,
        "pm_notes": pm_section,
    }

    return analysis


__all__ = ["analyze_chunks_hybrid"]
