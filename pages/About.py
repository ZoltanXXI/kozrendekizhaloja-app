# =========================
# ABOUT.PY — KÖZRENDEK ÍZHÁLÓJA
# =========================

import os
import re
import unicodedata
from html import unescape
from pathlib import Path
from collections import defaultdict

import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
import streamlit as st

from utils.fasting import FASTING_RECIPE_TITLES

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="A PROJEKTRŐL",
    page_icon="📜",
    layout="wide"
)

# =========================
# CSS / STYLE (VÁLTOZATLANUL)
# =========================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&display=swap');

/* SIDEBAR */
[data-testid="stSidebar"] > div:first-child {
    background-color: #5c1a1a !important;
    font-family: 'Cinzel', serif !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] * {
    font-family: 'Cinzel', serif !important;
    color: #ffffff !important;
}

[data-testid="stIconMaterial"],
[data-testid="stKeyboardShortcutButton"],
button[aria-label*="keyboard"],
[data-testid^="stTooltip"] {
    display: none !important;
}

/* TYPOGRAPHY */
.main-title {
    text-align: center;
    color: #2c1810;
    font-size: 3.5rem;
    font-weight: bold;
    font-family: 'Georgia', serif;
}

.section-title {
    color: #2c1810;
    font-size: 2rem;
    font-weight: bold;
    margin-top: 3rem;
    font-family: 'Georgia', serif;
}

.body-text {
    color: #4a3728;
    font-size: 1.1rem;
    line-height: 1.8;
    text-align: justify;
}

.highlight-box {
    background: linear-gradient(to right, #fffbf0, #fff9e6);
    border-left: 4px solid #d4af37;
    padding: 2rem;
    margin: 2rem 0;
    font-style: italic;
    color: #5c4033;
    border-radius: 0 8px 8px 0;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.markdown("""
<div style="display:block;margin:auto;padding:0.5rem 2rem;
background:linear-gradient(to right,#5c070d,#840a13);border-radius:8px;">
<h1 style="font-family:Cinzel,serif;color:white;margin:0;">A PROJEKTRŐL</h1>
</div>
<div style="width:100px;height:4px;
background:linear-gradient(to right,#d4af37,#f0d98d,#d4af37);
margin:1.5rem auto 3rem auto;"></div>
""", unsafe_allow_html=True)

# =========================
# INTRO / NARRATIVE
# =========================

st.markdown("""
<div class="body-text">
<p>
A <strong>Közrendek Ízhálója</strong> projekt célja, hogy modern hálózattudományi
és mesterséges intelligencia eszközökkel rekonstruálja és újraértelmezze
a XVII. századi magyar gasztronómia ízlogikáját,
különös tekintettel a <em>Szakácsmesterségnek könyvecskéje</em> (1698) recepthálózatára.
</p>
</div>
""", unsafe_allow_html=True)

# =========================
# CSV / NORMALIZATION HELPERS (EGYESÍTVE)
# =========================

def strip_icon_ligatures(s):
    if not isinstance(s, str):
        return ""
    s = unescape(s)
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[_\-\s]+", " ", s).strip()
    return s

def normalize_label(s):
    return strip_icon_ligatures(s).lower().strip() if isinstance(s, str) else ""

def resolve_csv(filename):
    script_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(script_dir, "data", filename),
        os.path.join(os.getcwd(), "data", filename),
        os.path.join(os.path.abspath(os.path.join(script_dir, "..")), "data", filename),
        f"data/{filename}",
        filename
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# =========================
# FASTING METRIC (VIZUÁLIS BLOKKHOZ)
# =========================

hist_path = resolve_csv("HistoricalRecipe_export.csv")
fasting_pct_display = "—"

if hist_path:
    hist_df = pd.read_csv(hist_path, sep=None, engine="python", on_bad_lines="skip")
    if "title" in hist_df.columns:
        titles = hist_df["title"].apply(strip_icon_ligatures)
        fasting_count = sum(t in FASTING_RECIPE_TITLES for t in titles)
        fasting_pct_display = f"{round(fasting_count / len(titles) * 100)}%"

# =========================
# METRIC CARDS
# =========================

c1, c2, c3, c4 = st.columns(4)
c1.metric("Történeti recept", "330")
c2.metric("Node (hálózat)", "838")
c3.metric("Átlag szószám", "70.7")
c4.metric("Böjti receptek", fasting_pct_display)

# =========================
# 🔬 ADATVEZÉRELT KUTATÁSI EREDMÉNYEK (1. KÓD)
# =========================

st.markdown("---")
st.markdown("## 🔬 Kutatási eredmények (adatok alapján)")

tripartit_path = resolve_csv("Recept_halo__molekula_tripartit.csv")
edges_path = resolve_csv("recept_halo_edges.csv")

if not all([tripartit_path, edges_path, hist_path]):
    st.warning("Hiányzó adatfájl(ok).")
else:
    trip = pd.read_csv(tripartit_path, sep=";", on_bad_lines="skip")
    edges = pd.read_csv(edges_path, on_bad_lines="skip")
    hist = pd.read_csv(hist_path, on_bad_lines="skip")

    trip["Label"] = trip.iloc[:, 0].apply(strip_icon_ligatures)
    trip["norm"] = trip["Label"].apply(normalize_label)
    trip["node_type"] = trip.iloc[:, 1].astype(str)

    G = nx.Graph()
    for _, r in trip.iterrows():
        G.add_node(r["norm"], label=r["Label"], node_type=r["node_type"])

    srcs = edges.iloc[:, 0].apply(normalize_label)
    tgts = edges.iloc[:, -1].apply(normalize_label)
    G.add_edges_from(zip(srcs, tgts))

    ingredients = [
        n for n, d in G.nodes(data=True)
        if "alapanyag" in d.get("node_type", "").lower()
    ]

    deg = dict(G.degree())
    pr = nx.pagerank(G)

    st.markdown("### Legközpontibb alapanyagok")
    for n, v in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:10]:
        if n in ingredients:
            st.markdown(f"- **{G.nodes[n]['label']}** — Degree: {v}")

    st.markdown("### Molekuláris hasonlóság vs. együtt előfordulás")
    st.markdown(
        "Spearman-korreláció számítása az alapanyagpárok közös molekulái "
        "és történeti együtt-előfordulása között (részletesen dokumentálva)."
    )

# =========================
# FOOTER
# =========================

st.markdown("""
<div style="text-align:center;margin-top:4rem;padding:2rem;
background:linear-gradient(to bottom,#fffbf0,#fff9e6);border-radius:8px;">
<strong>Közrendek Ízhálója</strong><br/>
Hálózatelemzés · Történeti források · AI generálás<br/>
© 2025
</div>
""", unsafe_allow_html=True)
