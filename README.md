# LSP AI Assistant

An AI-powered Streamlit application that helps language service providers analyse source documents, build draft style guides, prepare translator personas, and generate project manager briefs.

## Features
- Upload multiple DOCX, PDF, TXT, XLIFF, or memoQ MQXLIFF files in one go.
- Automatic document analysis (domain, tone, audience, difficulty) with combined reporting.
- Draft style guide generation aligned with the Translation Style Guide Questionnaire, including optional PM inputs.
- LLM-ready translator persona creation and project manager briefs.
- JSON-first outputs with Word (`.docx`) exports for all key artefacts.
- Quick hand-off to the hosted [Termextractor](https://termtool.streamlit.app/) tool for terminology creation.

## Project Structure
```
src/
  ui/
    app.py                # Streamlit entry point
    components.py         # Reusable UI helpers
  core/
    document_ingestion.py # File parsing for supported formats
    document_analysis.py  # Heuristic & LLM-assisted analysis
    styleguide_builder.py # Style guide assembly
    persona_builder.py    # Translator persona generator
    pm_brief.py           # Project manager brief generator
    terminology_client.py # Connector to Termextractor
    llm_client.py         # JSON-enforcing wrapper for OpenAI/Anthropic
  config/
    settings.yaml         # Application defaults and LLM prompts
```

## Installation
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Provide any required secrets (LLM keys, etc.) via environment variables or `st.secrets`.

## Running Locally
Run commands from the project root so that Python can resolve the `src` package correctly.

```bash
streamlit run src/ui/app.py
```

The helper script `python run_app.py` simply executes the same command.

## Running on Streamlit Cloud
- Repository root must contain the app.
- Entry point: `src/ui/app.py`.
- Imports are package-based under `src/`, so no additional path tweaks are required beyond cloning the repository.

### Docker
You can also run the application via Docker:
```bash
docker build -t lsp-assistant .
docker run -p 8501:8501 lsp-assistant
```

## Terminology Extraction
Terminology generation is delegated to the external Termextractor application: <https://termtool.streamlit.app/>. The Streamlit UI links directly to the hosted tool and forwards the selected language pair so the PM can upload the same files there.

## LLM Configuration & Secrets
LLM usage is optional and controlled by `src/config/settings.yaml` and a sidebar toggle. To enable:
1. Set `llm.enabled` to `true` in `settings.yaml`.
2. Provide the appropriate API key via environment variables or `st.secrets`.
   - `OPENAI_API_KEY` for OpenAI models.
   - `ANTHROPIC_API_KEY` for Anthropic models.

The prompts used for analysis, style guide enrichment, persona creation, and PM briefs are defined in `settings.yaml`. They can be customised per deployment or overridden via `st.secrets`.

## Deploying to Streamlit Cloud
1. Push this repository to your Git hosting provider.
2. On [Streamlit Cloud](https://streamlit.io/cloud), create a new app and point it to `src/ui/app.py`.
3. Configure the required environment variables (API keys, terminology endpoint) in the Streamlit Cloud secrets.
4. (Optional) Adjust `settings.yaml` or override values using `st.secrets`.

## Notes
- Terminology extraction is never re-implemented locally; the app delegates entirely to the Termextractor package or service.
- All generated outputs originate as JSON before being rendered or exported.
