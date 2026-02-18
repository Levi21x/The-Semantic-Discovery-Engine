# Semantic Discovery Engine

Semantic Discovery Engine is an NLP-powered recommender that solves the cold-start problem by understanding item content (text) and performing semantic search over vector embeddings.

This repository contains:

- `pipeline.py` — ETL that builds `data/processed_movies.csv` from raw CSVs.
- `engine.py`   — Encodes text into embeddings and stores them in ChromaDB.
- `api.py`      — FastAPI microservice that exposes `/recommend` and `/stats`.
- `app.py`      — Streamlit demo frontend.
- `requirements.txt` — pinned Python dependencies.

CI: GitHub Actions (lint + syntax check)

## Quick start (development)

1. Create a virtual environment and activate it (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Prepare data (if not already present in `data/`):

```bash
python pipeline.py
```

3. Index (one-time, stores vectors in `chroma_store/`):

```bash
python engine.py        # add --force to re-index
```

4. Run the API server:

```bash
uvicorn api:app --reload --port 8000
```

Open Swagger UI: `http://127.0.0.1:8000/docs`

5. Run the Streamlit UI (in another terminal):

```bash
streamlit run app.py
```

Or run the UI in standalone mode (no separate API):

```bash
STANDALONE=1 streamlit run app.py
```

## Packaging & persistence

- ChromaDB store is persisted to `chroma_store/` (ignored by `.gitignore`).
- Processed CSV is written to `data/processed_movies.csv`.

## Notes

- The project uses `all-MiniLM-L6-v2` (sentence-transformers) to produce 384-d vectors.
- Model weights are downloaded lazily on first run; indexing can take time on CPU.
- CI is conservative: it runs linting and a syntax check (no heavy model downloads).

## Contributing

PRs welcome. If adding tests, place them under `tests/` and update CI accordingly.

---

Made with ❤️ — the Cold-Start Solver.
