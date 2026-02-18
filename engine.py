"""
engine.py  —  Segment 2: The Cognitive + Storage Layer
=======================================================
• Loads processed_movies.csv
• Encodes every movie's 'soup' field with sentence-transformers
  (all-MiniLM-L6-v2  →  384-dim vectors)
• Stores vectors in a persistent ChromaDB collection
• Exposes a recommend() function used by both the API and the UI

Run (one-time indexing):
    python engine.py

The ChromaDB data is written to  ./chroma_store/  so indexing
only needs to happen once; subsequent runs skip it automatically.
"""

import os
import math
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(__file__)
PROCESSED_FILE  = os.path.join(BASE_DIR, "data", "processed_movies.csv")
CHROMA_DIR      = os.path.join(BASE_DIR, "chroma_store")
COLLECTION_NAME = "movies"
MODEL_NAME      = "all-MiniLM-L6-v2"
BATCH_SIZE      = 512          # rows per ChromaDB upsert call


# ── Engine Class ──────────────────────────────────────────────────────────────
class SemanticEngine:
    """
    Singleton-style class that:
      1. Loads the SentenceTransformer model once.
      2. Opens (or creates) the ChromaDB persistent collection.
      3. Provides recommend(query, n_results) for any caller.
    """

    def __init__(self):
        print("[Engine] Loading sentence-transformer model …")
        self.model = SentenceTransformer(MODEL_NAME)

        print("[Engine] Connecting to ChromaDB …")
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )

    # ── Index ─────────────────────────────────────────────────────────────────
    def index(self, force: bool = False) -> None:
        """
        Encode all movies and upsert them into ChromaDB.
        Skips if the collection already has data and force=False.
        """
        existing = self.collection.count()
        if existing > 0 and not force:
            print(f"[Engine] Collection already has {existing:,} items — skipping index.")
            print("[Engine] Pass force=True to re-index.")
            return

        print(f"[Engine] Loading processed data from: {PROCESSED_FILE}")
        df = pd.read_csv(PROCESSED_FILE)
        df["soup"] = df["soup"].fillna("").astype(str)

        total   = len(df)
        n_batch = math.ceil(total / BATCH_SIZE)
        print(f"[Engine] Encoding {total:,} movies in {n_batch} batches …")

        for i in tqdm(range(n_batch), desc="Indexing batches"):
            batch     = df.iloc[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            soups     = batch["soup"].tolist()
            ids       = batch["movieId"].astype(str).tolist()
            metadatas = batch[["title", "clean_title", "year", "genres_clean", "tags"]]\
                            .fillna("")\
                            .to_dict(orient="records")

            embeddings = self.model.encode(soups, show_progress_bar=False).tolist()

            self.collection.upsert(
                ids        = ids,
                embeddings = embeddings,
                documents  = soups,
                metadatas  = metadatas,
            )

        print(f"[Engine] ✓ Indexed {self.collection.count():,} movies into ChromaDB.")

    # ── Recommend ─────────────────────────────────────────────────────────────
    def recommend(self, query: str, n_results: int = 10) -> list[dict]:
        """
        Encode the free-text query and return the n closest movies.

        Returns a list of dicts:
          {
            "title"       : str,
            "year"        : str,
            "genres"      : str,
            "tags"        : str,
            "score"       : float,   # 0-1, higher = more similar
            "distance"    : float,   # raw cosine distance from ChromaDB
          }
        """
        query_vec = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings = query_vec,
            n_results        = n_results,
            include          = ["metadatas", "distances"],
        )

        recommendations = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            score = round(1 - dist, 4)   # cosine similarity  (1 = identical)
            recommendations.append({
                "title"   : meta.get("title",        "N/A"),
                "year"    : meta.get("year",          "N/A"),
                "genres"  : meta.get("genres_clean",  "N/A"),
                "tags"    : meta.get("tags",          ""),
                "score"   : score,
                "distance": round(dist, 4),
            })

        return recommendations

    # ── Stats ─────────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        """Return basic collection stats."""
        return {
            "collection"   : COLLECTION_NAME,
            "total_movies" : self.collection.count(),
            "model"        : MODEL_NAME,
            "vector_dim"   : 384,
        }


# ── CLI: one-time indexing ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    force_reindex = "--force" in sys.argv

    print("=" * 60)
    print("  SEMANTIC DISCOVERY ENGINE — Indexer")
    print("=" * 60)

    engine = SemanticEngine()
    engine.index(force=force_reindex)

    # Quick smoke-test
    print("\n[Test] Querying: 'sad robot in space' …")
    hits = engine.recommend("sad robot in space", n_results=5)
    for rank, h in enumerate(hits, 1):
        print(f"  {rank}. {h['title']} ({h['year']})  score={h['score']}")

    print("\n  Engine ready.\n")
