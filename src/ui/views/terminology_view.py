from __future__ import annotations

from typing import List

import streamlit as st

from src.core.terminology_client import parse_terminology_file


def _ensure_state() -> None:
    if "terminology" not in st.session_state:
        st.session_state["terminology"] = []


def render() -> None:
    _ensure_state()

    st.title("Terminology Extractor")
    st.caption("Hand off terminology extraction to the dedicated Termextractor app and attach the results here.")

    st.markdown(
        "This workspace delegates terminology extraction to the external [Termextractor](https://termtool.streamlit.app/)."
    )
    if hasattr(st, "link_button"):
        st.link_button("Open Termextractor", "https://termtool.streamlit.app/")
    else:  # pragma: no cover - fallback for older Streamlit
        st.markdown("[Open Termextractor](https://termtool.streamlit.app/)")

    st.divider()
    st.subheader("Attach extracted terminology")
    uploaded_terms = st.file_uploader(
        "Upload terminology export (CSV, XLSX, TBX)",
        type=["csv", "xlsx", "xlsm", "tbx"],
        key="terminology_uploader",
    )

    if uploaded_terms is not None:
        try:
            entries = parse_terminology_file(uploaded_terms)
        except Exception as exc:  # pragma: no cover - depends on file format
            st.error(f"Failed to parse terminology file: {exc}")
        else:
            st.session_state["terminology"] = entries
            st.success(f"Loaded {len(entries)} terminology entries.")

    entries: List[dict] = st.session_state.get("terminology", [])
    if entries:
        st.write(f"{len(entries)} terms currently attached to this project.")
        with st.expander("Preview terminology"):
            st.table(entries[:50])
    else:
        st.info("No terminology attached yet. Upload the output from Termextractor when it's ready.")
