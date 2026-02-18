"""
app.py  —  Segment 4: The Streamlit Dashboard
==============================================
The "wow-factor" frontend.

• Users type a free-text mood/vibe (e.g. "I'm sad and want to laugh")
• Results are shown as styled cards with title, year, genres, tags, and
  a similarity score bar.

Run:
    streamlit run app.py

Requirements:
    The FastAPI service must be running first:
        uvicorn api:app --reload --port 8000

    OR (standalone mode — talks directly to the engine):
        STANDALONE=1 streamlit run app.py
"""

import os
import time
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_URL    = os.getenv("API_URL", "http://127.0.0.1:8000")
STANDALONE = os.getenv("STANDALONE", "0") == "1"

# Genre → colour mapping for tags
GENRE_COLOURS = {
    "Action"    : "#e74c3c",
    "Comedy"    : "#f39c12",
    "Drama"     : "#3498db",
    "Thriller"  : "#8e44ad",
    "Romance"   : "#e91e8c",
    "Animation" : "#1abc9c",
    "Horror"    : "#c0392b",
    "Sci-Fi"    : "#2980b9",
    "Adventure" : "#27ae60",
    "Fantasy"   : "#9b59b6",
    "Crime"     : "#7f8c8d",
    "Documentary":"#16a085",
    "Children"  : "#f1c40f",
    "Musical"   : "#d35400",
    "Mystery"   : "#2c3e50",
    "War"       : "#95a5a6",
    "Western"   : "#a04000",
}

DEFAULT_COLOUR = "#555555"


# ── Helpers ───────────────────────────────────────────────────────────────────
def genre_pill(genre: str) -> str:
    colour = GENRE_COLOURS.get(genre.strip(), DEFAULT_COLOUR)
    return (
        f'<span style="background:{colour};color:#fff;'
        f'padding:3px 10px;border-radius:12px;font-size:0.75rem;'
        f'margin:2px;display:inline-block;">{genre.strip()}</span>'
    )


def score_bar(score: float) -> str:
    pct    = int(score * 100)
    colour = "#2ecc71" if score >= 0.5 else "#f39c12" if score >= 0.3 else "#e74c3c"
    return (
        f'<div style="background:#eee;border-radius:6px;height:8px;width:100%;margin:6px 0">'
        f'<div style="background:{colour};width:{pct}%;height:100%;border-radius:6px;'
        f'transition:width 0.4s ease"></div></div>'
        f'<small style="color:{colour};font-weight:600;">{pct}% match</small>'
    )


def movie_card(movie: dict, rank: int) -> str:
    genres_html = "".join(
        genre_pill(g) for g in movie["genres"].split() if g
    )
    tags_snippet = movie["tags"][:80] + "…" if len(movie["tags"]) > 80 else movie["tags"]
    tags_html    = (
        f'<p style="font-size:0.8rem;color:#888;margin:4px 0">🏷 {tags_snippet}</p>'
        if tags_snippet else ""
    )

    return f"""
    <div style="
        background: linear-gradient(135deg,#1e2130,#2a2f45);
        border-radius:14px;padding:18px 22px;margin:10px 0;
        border-left:4px solid #7c83fd;
        box-shadow:0 4px 15px rgba(0,0,0,0.3);
    ">
        <div style="display:flex;justify-content:space-between;align-items:flex-start">
            <div>
                <span style="font-size:1.3rem;font-weight:700;color:#e8eaf6">
                    #{rank} &nbsp;{movie['title']}
                </span>
                <span style="color:#888;font-size:0.9rem;margin-left:8px">
                    ({movie['year']})
                </span>
            </div>
        </div>
        <div style="margin:8px 0">{genres_html}</div>
        {tags_html}
        {score_bar(movie['score'])}
    </div>
    """


def fetch_recommendations_api(query: str, n: int) -> list[dict]:
    resp = requests.get(
        f"{API_URL}/recommend",
        params  = {"query": query, "n": n},
        timeout = 30,
    )
    resp.raise_for_status()
    return resp.json()["results"]


def fetch_recommendations_standalone(query: str, n: int) -> list[dict]:
    """
    Used when STANDALONE=1 — imports the engine directly so no
    separate API server is needed.
    """
    from engine import SemanticEngine
    if "engine" not in st.session_state:
        with st.spinner("Loading engine (first run — may take ~30 s) …"):
            st.session_state.engine = SemanticEngine()
            st.session_state.engine.index()
    return st.session_state.engine.recommend(query=query, n_results=n)


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Semantic Discovery Engine",
    page_icon  = "🎬",
    layout     = "wide",
)

# Global CSS
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"] {
        background: #13151f !important;
        color: #e8eaf6;
    }
    [data-testid="stSidebar"] { background: #1a1d2e !important; }
    div[data-testid="stMarkdownContainer"] p { color: #c5cae9; }
    .stTextInput > div > input {
        background:#1e2130 !important;color:#e8eaf6 !important;
        border:1px solid #7c83fd !important;border-radius:10px !important;
        font-size:1.05rem !important;
    }
    [data-testid="stButton"] button {
        background:linear-gradient(90deg,#7c83fd,#a78bfa) !important;
        color:#fff !important;border:none !important;
        border-radius:10px !important;font-weight:600 !important;
        padding:10px 28px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    n_results = st.slider("Number of results", 5, 30, 10)
    st.markdown("---")
    st.markdown("### 💡 Try these queries")
    example_queries = [
        "animated movie for kids with talking animals",
        "psychological thriller with a plot twist",
        "80s sci-fi adventure with a robot",
        "romantic comedy set in New York",
        "dark war drama based on true events",
        "feel-good comedy about unlikely friendship",
        "I want to cry watching a love story",
        "epic fantasy with magic and battles",
    ]
    for eq in example_queries:
        if st.button(eq, key=eq):
            st.session_state["query_input"] = eq

    st.markdown("---")
    mode = "🔌 Standalone" if STANDALONE else "🌐 API Mode"
    st.markdown(f"**Mode:** `{mode}`")
    if not STANDALONE:
        st.markdown(f"**API:** `{API_URL}`")


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center;font-size:2.8rem;
   background:linear-gradient(90deg,#7c83fd,#a78bfa);
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
   margin-bottom:4px'>
   🎬 Semantic Discovery Engine
</h1>
<p style='text-align:center;color:#888;font-size:1.1rem;margin-bottom:30px'>
   The Cold-Start Solver — describe a vibe, get perfect movie matches.
</p>
""", unsafe_allow_html=True)

# Query input
query = st.text_input(
    label       = "Describe what you want to watch",
    placeholder = "e.g.  'sad robot in space'  or  'I'm happy and want to laugh'",
    key         = "query_input",
)

col1, col2, col3 = st.columns([3, 1, 3])
with col2:
    search_clicked = st.button("🔍 Discover", use_container_width=True)

st.markdown("<hr style='border-color:#2a2f45'>", unsafe_allow_html=True)

# Results
if search_clicked and query.strip():
    with st.spinner("Searching the semantic space …"):
        t0 = time.time()
        try:
            if STANDALONE:
                results = fetch_recommendations_standalone(query.strip(), n_results)
            else:
                results = fetch_recommendations_api(query.strip(), n_results)
            elapsed = time.time() - t0
        except requests.exceptions.ConnectionError:
            st.error(
                f"Cannot connect to API at **{API_URL}**. "
                "Start with: `uvicorn api:app --reload --port 8000`  "
                "or run in standalone mode: `STANDALONE=1 streamlit run app.py`"
            )
            st.stop()

    if results:
        st.markdown(
            f"<p style='color:#888'>Found <b style='color:#7c83fd'>{len(results)}</b> "
            f"results for <i>\"{query}\"</i> in "
            f"<b style='color:#7c83fd'>{elapsed:.2f}s</b></p>",
            unsafe_allow_html=True,
        )

        # Two-column card layout
        left_col, right_col = st.columns(2)
        for i, movie in enumerate(results):
            card_html = movie_card(movie, i + 1)
            if i % 2 == 0:
                left_col.markdown(card_html, unsafe_allow_html=True)
            else:
                right_col.markdown(card_html, unsafe_allow_html=True)
    else:
        st.warning("No results found. Try a different query.")

elif not query.strip() and search_clicked:
    st.warning("Please enter a query before searching.")

else:
    st.markdown("""
    <div style='text-align:center;padding:60px 0;color:#555'>
        <div style='font-size:4rem'>🎬</div>
        <p style='font-size:1.1rem;margin-top:16px'>
            Type a mood, vibe, or description above to discover movies.
        </p>
        <p style='font-size:0.9rem'>
            Powered by <b style='color:#7c83fd'>all-MiniLM-L6-v2</b> embeddings
            + <b style='color:#7c83fd'>ChromaDB</b> vector search
        </p>
    </div>
    """, unsafe_allow_html=True)
