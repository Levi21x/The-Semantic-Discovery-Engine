"""
api.py  —  Segment 3: The Service Layer (FastAPI Microservice)
==============================================================
Wraps the SemanticEngine in a REST API.

Endpoints:
  GET  /                         → health check
  GET  /stats                    → collection stats
  GET  /recommend?query=…&n=10   → semantic recommendations
  POST /recommend                → same, via JSON body

Run:
    uvicorn api:app --reload --port 8000

Swagger UI:  http://127.0.0.1:8000/docs
ReDoc:       http://127.0.0.1:8000/redoc
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine import SemanticEngine

# ── App lifecycle  ────────────────────────────────────────────────────────────
# The engine is loaded once at startup and shared across all requests.

engine_instance: Optional[SemanticEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the engine when the server starts; clean up on shutdown."""
    global engine_instance
    print("[API] Initialising Semantic Engine …")
    engine_instance = SemanticEngine()
    # Ensure the index exists (no-op if already indexed)
    engine_instance.index()
    print("[API] Engine ready.")
    yield
    print("[API] Shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Semantic Discovery Engine",
    description = (
        "An NLP-powered movie recommender that solves the **Cold Start** problem. "
        "It understands the *meaning* of your query — no user history required."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# Allow the Streamlit frontend (any origin in dev) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class MovieResult(BaseModel):
    title   : str
    year    : str
    genres  : str
    tags    : str
    score   : float = Field(description="Cosine similarity 0-1 (higher = better match)")
    distance: float


class RecommendRequest(BaseModel):
    query    : str  = Field(..., example="sad robot falling in love")
    n_results: int  = Field(10, ge=1, le=50, description="Number of results (1-50)")


class RecommendResponse(BaseModel):
    query       : str
    n_results   : int
    results     : list[MovieResult]


class StatsResponse(BaseModel):
    collection   : str
    total_movies : int
    model        : str
    vector_dim   : int


# ── Helper ────────────────────────────────────────────────────────────────────
def _get_engine() -> SemanticEngine:
    if engine_instance is None:
        raise HTTPException(status_code=503, detail="Engine not initialised yet.")
    return engine_instance


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is alive."""
    return {"status": "ok", "message": "Semantic Discovery Engine is running."}


@app.get("/stats", response_model=StatsResponse, tags=["Info"])
def get_stats():
    """Return metadata about the vector collection."""
    return _get_engine().stats()


@app.get(
    "/recommend",
    response_model = RecommendResponse,
    tags           = ["Recommendations"],
    summary        = "Get recommendations via query string",
)
def recommend_get(
    query    : str = Query(..., example="funny movie about friendship", min_length=2),
    n        : int = Query(10, ge=1, le=50, description="Number of results"),
):
    """
    Semantic search via **GET**.

    - `query` — any natural-language description (e.g. *"action movie in space"*)
    - `n` — how many results to return (default 10, max 50)
    """
    engine  = _get_engine()
    results = engine.recommend(query=query, n_results=n)
    return RecommendResponse(query=query, n_results=len(results), results=results)


@app.post(
    "/recommend",
    response_model = RecommendResponse,
    tags           = ["Recommendations"],
    summary        = "Get recommendations via JSON body",
)
def recommend_post(payload: RecommendRequest):
    """
    Semantic search via **POST** (JSON body).

    Useful when the query is long or contains special characters.
    ```json
    { "query": "I want a comedy about love and dogs", "n_results": 5 }
    ```
    """
    engine  = _get_engine()
    results = engine.recommend(query=payload.query, n_results=payload.n_results)
    return RecommendResponse(
        query     = payload.query,
        n_results = len(results),
        results   = results,
    )
