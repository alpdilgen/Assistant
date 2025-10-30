"""Integration helpers for delegating terminology extraction to Termextractor."""
from __future__ import annotations

from typing import Optional, Sequence, TYPE_CHECKING
from urllib.parse import urlencode

import streamlit as st

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from streamlit.runtime.uploaded_file_manager import UploadedFile

TERMEXTRACTOR_BASE_URL = "https://termtool.streamlit.app/"


def send_to_external_termextractor(
    files: Sequence["UploadedFile"],
    src_lang: str,
    tgt_lang: str,
    project_name: Optional[str] = None,
) -> str:
    """Render a link guiding the user to the hosted Termextractor application."""

    params: dict[str, str] = {
        "source_language": src_lang,
        "target_language": tgt_lang,
    }
    if project_name:
        params["project_name"] = project_name

    query = urlencode(params)
    url = f"{TERMEXTRACTOR_BASE_URL}?{query}" if query else TERMEXTRACTOR_BASE_URL

    st.info(
        "Terminology extraction is handled by the external Termextractor app. "
        "Open the tool in a new tab and upload the same files to generate the glossary."
    )

    if hasattr(st, "link_button"):
        st.link_button("Go to Termextractor", url)
    else:  # pragma: no cover - fallback for older Streamlit versions
        st.markdown(f"[Go to Termextractor]({url})")

    if files:
        st.caption("Files prepared for terminology hand-off:")
        for uploaded_file in files:
            if uploaded_file is not None:
                st.write(f"• {uploaded_file.name}")

    return url


__all__ = ["send_to_external_termextractor"]
