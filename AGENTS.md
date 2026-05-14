# Repository Guidelines

## Project Structure & Module Organization

Core Python packages live in `src/`: `config/` loads `.env` settings, `llm/` wraps model providers, `indexing/` builds GraphRAG indexes, `retrieval/` searches memoir context, `generation/` produces literary background text, and `evaluation/` computes retrieval, factuality, and long-form quality metrics. Gradio and FastAPI entry points are in `web/app.py` and `web/api.py`; `run_web.py` selects the runtime mode. Data and generated indexes are under `data/`, scripts and diagnostics under `scripts/`, focused unit tests under `tests/`, and slower end-to-end generation checks under `generation_test/`.

## Build, Test, and Development Commands

Install dependencies after creating a virtual environment:

```powershell
pip install -r requirements.txt
```

Run the Gradio UI:

```powershell
python run_web.py gradio --port 8001
```

Run the API server:

```powershell
python run_web.py api
```

Build or rebuild the knowledge index when input data changes:

```powershell
python scripts\rebuild_index.py
```

Run the main test suite:

```powershell
python -m pytest tests -q
```

Use `python -m py_compile src\**\*.py web\*.py` or targeted `py_compile` commands for quick syntax checks after edits.

## Coding Style & Naming Conventions

Use Python 3.10+ style with 4-space indentation, type hints for public functions, and dataclasses or Pydantic models for structured data. Keep module names lowercase with underscores, classes in `PascalCase`, functions and variables in `snake_case`, and constants in `UPPER_SNAKE_CASE`. Prefer existing adapter, retriever, generator, and evaluator abstractions over ad hoc provider-specific logic in UI callbacks.

## Testing Guidelines

Tests use `pytest`. Name new files `test_<feature>.py` and test functions `test_<behavior>()`. Put fast deterministic tests in `tests/`; reserve LLM, index, or long-form integration scenarios for `generation_test/` or scripts with clear runtime expectations. Mock or isolate network/model calls where practical, and document required `.env` keys when a test needs live providers.

## Commit & Pull Request Guidelines

Recent history uses short imperative summaries and occasional scoped prefixes, for example `feat(web+eval): ...` or `modified evaluation metrics`. Prefer concise messages that name the affected area and behavior. Pull requests should include a summary, test commands run, configuration changes, linked issue or task when available, and screenshots for Gradio UI changes.

## Security & Configuration Tips

Do not commit real API keys. Keep secrets in `.env`, based on `env.example` or `.env.example`. Important variables include `DEFAULT_LLM_PROVIDER`, `DEFAULT_LLM_MODEL`, provider API keys, `GRAPHRAG_INPUT_DIR`, and `GRAPHRAG_OUTPUT_DIR`. When debugging, avoid pasting full keys from logs or screenshots.
