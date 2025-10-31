import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st

from src.ui.views import document_analyzer_view, style_guide_view, terminology_view


def main() -> None:
    st.set_page_config(page_title="AI Translation Workspace", layout="wide")

    st.sidebar.title("AI Translation Tools")
    app_choice = st.sidebar.radio(
        "Choose your tool",
        ["Document Analyzer", "Style Guide Creator", "Terminology Extractor"],
        index=0,
    )

    if app_choice == "Document Analyzer":
        document_analyzer_view.render()
    elif app_choice == "Style Guide Creator":
        style_guide_view.render()
    elif app_choice == "Terminology Extractor":
        terminology_view.render()


if __name__ == "__main__":
    main()
