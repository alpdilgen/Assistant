from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List

from docx import Document


def analysis_to_docx(analysis: Dict[str, Any]) -> bytes:
    """Convert a document analysis dictionary into DOCX bytes."""

    doc = Document()
    meta = analysis.get("document_metadata", {})
    clsf = analysis.get("classification", {})
    cx = analysis.get("complexity", {})
    prof = analysis.get("translator_profile", {})
    risks: List[Dict[str, Any]] = analysis.get("risks", []) or []
    refs = analysis.get("references", []) or []
    pm_notes = analysis.get("pm_notes", {}) or {}

    doc.add_heading("Document Analysis Report", level=1)
    doc.add_paragraph(f"Generated: {datetime.utcnow().isoformat()} UTC")

    # 1. Metadata
    doc.add_heading("1. Document Metadata", level=2)
    doc.add_paragraph(f"Project name: {meta.get('project_name', '')}")
    doc.add_paragraph(f"Source language: {meta.get('source_language', '')}")
    doc.add_paragraph(
        "Target languages: "
        + ", ".join(meta.get("target_languages", []) or [])
    )
    doc.add_paragraph(f"File count: {meta.get('file_count', '')}")
    doc.add_paragraph(f"Word estimate: {meta.get('word_estimate', '')}")

    # 2. Classification
    doc.add_heading("2. Classification", level=2)
    doc.add_paragraph(f"Domain: {clsf.get('domain', '')}")
    doc.add_paragraph(
        f"Subdomains: {', '.join(clsf.get('subdomains', []) or [])}"
    )
    doc.add_paragraph(
        f"Related fields: {', '.join(clsf.get('related_fields', []) or [])}"
    )

    # 3. Complexity
    doc.add_heading("3. Complexity", level=2)
    doc.add_paragraph(f"Level: {cx.get('level', '')}")
    doc.add_paragraph("Drivers:")
    for driver in cx.get("drivers", []) or []:
        doc.add_paragraph(f"- {driver}", style="List Bullet")

    # 4. Translator profile
    doc.add_heading("4. Translator Profile", level=2)
    doc.add_paragraph(
        f"Required background: {prof.get('required_background', '')}"
    )
    doc.add_paragraph(
        f"Preferred experience: {prof.get('preferred_experience', '')}"
    )
    doc.add_paragraph("Recommended tools:")
    for tool in prof.get("tools", []) or []:
        doc.add_paragraph(f"- {tool}", style="List Bullet")

    # 5. Risks
    doc.add_heading("5. Risks & Mitigations", level=2)
    for risk in risks:
        doc.add_paragraph(
            f"- {risk.get('name', '')}: {risk.get('description', '')}"
        )
        doc.add_paragraph(
            f"  Mitigation: {risk.get('mitigation', '')}"
        )

    # 6. References
    doc.add_heading("6. References", level=2)
    for ref in refs:
        doc.add_paragraph(f"- {ref}", style="List Bullet")

    # 7. PM Notes
    doc.add_heading("7. PM Notes", level=2)
    if pm_notes.get("original"):
        doc.add_paragraph(f"From PM: {pm_notes['original']}")
    doc.add_paragraph(f"System notes: {pm_notes.get('system_notes', '')}")

    bio = BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.read()


__all__ = ["analysis_to_docx"]
