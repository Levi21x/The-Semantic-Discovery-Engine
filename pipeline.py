"""
pipeline.py  —  Segment 1: The Ingestion / ETL Layer
=====================================================
Reads the raw MovieLens CSVs from archive (9)/, cleans and
enriches the text metadata, then saves a single processed
file  →  data/processed_movies.csv  that the engine uses.

Run:
    python pipeline.py
"""

import re
import os
import pandas as pd
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join(os.path.dirname(__file__), "archive (9)")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE   = os.path.join(PROCESSED_DIR, "processed_movies.csv")


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_title(title: str) -> str:
    """Remove year from title and strip extra whitespace."""
    title = re.sub(r"\(\d{4}\)\s*$", "", title).strip()
    return title


def extract_year(title: str) -> str:
    """Extract the 4-digit year that usually lives at the end of a title."""
    match = re.search(r"\((\d{4})\)\s*$", title)
    return match.group(1) if match else "Unknown"


def format_genres(genres: str) -> str:
    """Convert 'Action|Crime|Thriller' → 'Action Crime Thriller'."""
    if pd.isna(genres) or genres == "(no genres listed)":
        return ""
    return genres.replace("|", " ")


def aggregate_tags(tags_df: pd.DataFrame) -> pd.Series:
    """
    For every movieId, join all user-supplied tags into a single
    lowercased, deduplicated string.
    """
    def _join(grp):
        unique_tags = grp["tag"].dropna().astype(str).str.lower().unique()
        return " ".join(unique_tags)

    return tags_df.groupby("movieId").apply(_join, include_groups=False).rename("tags")


def build_soup(row: pd.Series) -> str:
    """
    Merge title + genres + tags into one rich text field (the 'soup')
    that the transformer will encode.
    """
    parts = [
        row["clean_title"],
        row["genres_clean"],
        row.get("tags", ""),
    ]
    return " ".join(p for p in parts if p).strip()


# ── Main ──────────────────────────────────────────────────────────────────────
def run_pipeline() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 60)
    print("  SEMANTIC DISCOVERY ENGINE — ETL Pipeline")
    print("=" * 60)

    # 1. Load raw files
    print("\n[1/5] Loading raw CSV files …")
    movies_df = pd.read_csv(os.path.join(RAW_DIR, "movies.csv"))
    tags_df   = pd.read_csv(os.path.join(RAW_DIR, "tags.csv"))
    print(f"      movies : {len(movies_df):,} rows")
    print(f"      tags   : {len(tags_df):,} rows")

    # 2. Clean movies
    print("\n[2/5] Cleaning movie metadata …")
    tqdm.pandas(desc="      processing titles")
    movies_df["clean_title"]  = movies_df["title"].progress_apply(clean_title)
    movies_df["year"]         = movies_df["title"].apply(extract_year)
    movies_df["genres_clean"] = movies_df["genres"].apply(format_genres)

    # 3. Aggregate tags
    print("\n[3/5] Aggregating user tags …")
    tag_series = aggregate_tags(tags_df)
    movies_df  = movies_df.join(tag_series, on="movieId")
    movies_df["tags"] = movies_df["tags"].fillna("")

    # 4. Build the text soup
    print("\n[4/5] Building content soup …")
    tqdm.pandas(desc="      building soup")
    movies_df["soup"] = movies_df.progress_apply(build_soup, axis=1)

    # 5. Save
    print("\n[5/5] Saving processed data …")
    keep_cols = ["movieId", "title", "clean_title", "year",
                 "genres_clean", "tags", "soup"]
    movies_df[keep_cols].to_csv(OUTPUT_FILE, index=False)
    print(f"      ✓ Saved {len(movies_df):,} movies → {OUTPUT_FILE}")

    print("\n  Pipeline complete.\n")


if __name__ == "__main__":
    run_pipeline()
