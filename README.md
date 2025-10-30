# LSP AI Assistant

An AI-powered Streamlit application that helps language service providers analyse source documents, build draft style guides, prepare translator personas, and generate project manager briefs.

## Features
- Upload DOCX, PDF, TXT, XLIFF, or memoQ MQXLIFF files.
- Automatic document analysis (domain, tone, audience, difficulty).
- Optional connection to the external [alpdilgen/Termextractor](https://github.com/alpdilgen/Termextractor) project for up-to-date terminology.
- Draft style guide generation using extracted terminology.
- LLM-ready translator persona creation.
- Project manager brief with risks, prerequisites, and QA recommendations.
- JSON-first outputs with download options for JSON and DOCX artefacts.

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
3. (Optional) Install the `termextractor` package locally if you want to use the local extraction strategy.

## Running Locally
Run commands from the project root so that Python can resolve the `src` package correctly.

```bash
python run_app.py
```

This helper simply executes `streamlit run src/ui/app.py` with the correct working directory. You can also invoke Streamlit directly with the same command if you prefer. The app will be served on `http://localhost:8501` by default.

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

## Configuring Terminology Extraction
The assistant first attempts to import `TermExtractor` from the `termextractor` Python package. Install it in the same environment to use the local strategy. If the import fails, the app falls back to calling a REST endpoint specified by the `TERMINOLOGY_SERVICE_URL` environment variable (default: `https://termextractor-service/api/extract`).

Additional environment variables:
- `TERMINOLOGY_SERVICE_TIMEOUT` (seconds, default `30`)

## LLM Configuration
LLM usage is optional and controlled by `src/config/settings.yaml` and a sidebar toggle. To enable:
1. Set `llm.enabled` to `true` in `settings.yaml`.
2. Provide the appropriate API key via environment variable:
   - `OPENAI_API_KEY` for OpenAI models.
   - `ANTHROPIC_API_KEY` for Anthropic models.

The prompts used for analysis, style guide enrichment, persona creation, and PM briefs are defined in `settings.yaml`. They can be customised per deployment.

## Deploying to Streamlit Cloud
1. Push this repository to your Git hosting provider.
2. On [Streamlit Cloud](https://streamlit.io/cloud), create a new app and point it to `src/ui/app.py`.
3. Configure the required environment variables (API keys, terminology endpoint) in the Streamlit Cloud secrets.
4. (Optional) Adjust `settings.yaml` or override values using `st.secrets`.

## Notes
- Terminology extraction is never re-implemented locally; the app delegates entirely to the Termextractor package or service.
- All generated outputs originate as JSON before being rendered or exported.
