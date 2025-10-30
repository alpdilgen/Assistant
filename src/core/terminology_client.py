"""Connector to the alpdilgen/Termextractor project."""
from __future__ import annotations

import json
import logging
import os
from typing import List

import requests

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from termextractor import TermExtractor  # type: ignore
except Exception:  # pragma: no cover - absence is expected in CI
    TermExtractor = None  # type: ignore

_LOCAL_EXTRACTOR = None


def _get_local_extractor():
    global _LOCAL_EXTRACTOR
    if TermExtractor is None:
        return None
    if _LOCAL_EXTRACTOR is None:
        try:
            _LOCAL_EXTRACTOR = TermExtractor()
        except Exception as exc:  # pragma: no cover - depends on external package
            logger.error("Failed to initialise TermExtractor: %s", exc)
            _LOCAL_EXTRACTOR = None
    return _LOCAL_EXTRACTOR


def _call_local(doc_bytes: bytes, filename: str, src_lang: str, tgt_lang: str) -> List[dict] | None:
    extractor = _get_local_extractor()
    if extractor is None:
        return None

    text_payload = doc_bytes.decode("utf-8", errors="ignore")

    for method_name in ("extract_from_bytes", "extract", "__call__"):
        method = getattr(extractor, method_name, None)
        if callable(method):
            try:
                result = method(
                    text_payload if method_name == "extract" else doc_bytes,
                    source_language=src_lang,
                    target_language=tgt_lang,
                )
                if isinstance(result, dict) and "terms" in result:
                    return list(result["terms"])
                if isinstance(result, list):
                    return result
            except TypeError:
                try:
                    result = method(text_payload)
                    if isinstance(result, dict) and "terms" in result:
                        return list(result["terms"])
                    if isinstance(result, list):
                        return result
                except Exception:  # pragma: no cover - best effort fallback
                    continue
            except Exception as exc:  # pragma: no cover - depends on external impl
                logger.error("Local term extraction failed via %s: %s", method_name, exc)
                continue
    return None


def _call_remote(doc_bytes: bytes, filename: str, src_lang: str, tgt_lang: str) -> List[dict]:
    url = os.getenv("TERMINOLOGY_SERVICE_URL", "https://termextractor-service/api/extract")
    try:
        response = requests.post(
            url,
            files={"file": (filename, doc_bytes)},
            data={"source_language": src_lang, "target_language": tgt_lang},
            timeout=float(os.getenv("TERMINOLOGY_SERVICE_TIMEOUT", "30")),
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and "terms" in payload:
            return list(payload["terms"])
        if isinstance(payload, list):
            return payload
        logger.warning("Unexpected payload from terminology service: %s", payload)
    except requests.RequestException as exc:
        logger.error("Remote terminology service error: %s", exc)
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode terminology response: %s", exc)
    return []


def get_terminology(doc_bytes: bytes, filename: str, src_lang: str, tgt_lang: str) -> List[dict]:
    """Return a list of term dictionaries from the Termextractor service."""

    if not doc_bytes:
        return []

    local_terms = _call_local(doc_bytes, filename, src_lang, tgt_lang)
    if local_terms is not None:
        return local_terms

    return _call_remote(doc_bytes, filename, src_lang, tgt_lang)


__all__ = ["get_terminology"]
