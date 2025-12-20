import os
import re
import unicodedata
from html import unescape
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
import streamlit as st

from utils.fasting import FASTING_RECIPE_TITLES, is_fasting_title

def strip_icon_ligatures(s):
    if not isinstance(s, str):
        return ""
    s = unescape(s)
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'<[^>]+>', '', s)
    s = re.sub(r'[_\-\s]+', ' ', s).strip()
    return s

def normalize_label(s):
    if not isinstance(s, str):
        return ''
    s = strip_icon_ligatures(s).lower()
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def resolve_path_candidates(rel_paths):
    script_dir = os.path.dirname(__file__)
    candidates = []
    bases = [script_dir, os.getcwd(), os.path.abspath(os.path.join(script_dir, '..'))]
    for b in bases:
        for rp in rel_paths:
            candidates.append(os.path.normpath(os.path.join(b, rp)))
    candidates.extend(rel_paths)
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

@st.cache_data
def load_csv_flexible(path, default_sep=None):
    if not path:
        return pd.DataFrame()
    try:
        if default_sep:
            return pd.read_csv(path, delimiter=default_sep, encoding='utf-8', on_bad_lines='skip')
        else:
            return pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
    except Exception:
        try:
            return pd.read_csv(path, sep=None, engine='python', encoding='utf-8', on_bad_lines='skip')
        except Exception:
            try:
                if default_sep:
                    return pd.read_csv(path, delimiter=default_sep, encoding='latin1', on_bad_lines='skip')
                else:
                    return pd.read_csv(path, encoding='latin1', on_bad_lines='skip')
            except Exception:
                return pd.DataFrame()

def sequence_similarity(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

st.set_page_config(page_title="A PROJEKTRŐL", page_icon="📜", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400&display=swap');
/* Sidebar styling (aligns with app.py) */
[data-testid="stSidebar"] > div:first-child { background-color: #5c1a1a !important; font-family: 'Cinzel', serif !important; color: #ffffff !important; }
[data-testid="stSidebar"] button, [data-testid="stSidebar"] .st-expander, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div[data-testid$="-label"] { font-family: 'Cinzel', serif !important; color: #ffffff !important; }
[data-testid="stSidebar"] span[data-testid="stIconMaterial"], .span[data-testid="stIconMaterial"] { display: none !important; }

/* Main page styling */
.reader-quote { background: linear-gradient(to right, #fff8e6, #fff5da); border: 2px solid #d4af37; padding: 2rem 2.5rem; color: #5c4033; font-size: 1.05rem; line-height: 1.7; border-radius: 10px; position: relative; margin-bottom: 1.5rem; }
.list-card { background: #fffaf2; border: 1px solid #e6d2a3; padding: 12px; border-radius: 8px; margin-bottom: 12px; }
.list-title { font-weight: 700; color: #2c1810; font-size: 1.05rem; margin-bottom: 8px; }
.list-item { margin: 6px 0; line-height: 1.4; }
.metric-card { text-align: center; padding: 1.5rem; background: #fffbf0; border-radius: 8px; border: 2px solid #d4af37; }
.section-title { color: #2c1810; font-size: 1.35rem; font-weight: bold; margin-top: 1.2rem; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem; }
.highlight-box { background: linear-gradient(to right, #fffbf0, #fff9e6); border-left: 4px solid #d4af37; padding: 1rem; margin: 1.2rem 0; color: #5c4033; border-radius: 6px; }
/* Large centered quote */
.large-quote { font-family: 'Cinzel', serif; font-size: 2rem; color: #3b2b1b; text-align: center; margin: 2rem auto; max-width: 1200px; line-height: 1.2; font-weight:700; }
.large-quote small { display:block; font-size:0.85rem; margin-top:0.5rem; color:#7a5b3a; font-weight:400; }

/* Prevent overscroll */
body {
    overscroll-behavior: none;
}

/* Hide Streamlit's default footer and extra space */
footer {visibility: hidden;}
.block-container {padding-bottom: 2rem !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:block; width:fit-content; margin:0 auto; padding:0.5rem 2rem; background:linear-gradient(to right,#5c070d,#840a13); border-radius:8px; text-align:center;">
    <h1 style="font-family:Cinzel, serif; color:#ffffff; margin:0;">A PROJEKTRŐL</h1>
</div>
<div style="width:100px; height:4px; background:linear-gradient(to right,#d4af37,#f0d98d,#d4af37); margin:1rem auto 1.5rem auto; border-radius:2px;"></div>
""", unsafe_allow_html=True)

# Top anchor for scroll-to-top functionality
st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="reader-quote">
    <span style="font-size:4.2rem; float:left; line-height:1; margin-right:0.5rem; color:#8b5a2b; font-family:Georgia, serif;">E</span>
    z az én könyvecském nem siet az udvarokban való nagy konyhákhoz,
    ahol a szakácsok csak magoktól is jóízű étkeket tudnak főzni; hanem csak leginkább
    a becsületes közrendeknek, akik gyakorta szakács nélkül szűkölködnek, akar szolgálni…
    <div style="margin-top:0.8rem;">
    Azért jámbor Olvasó, ha kedved szerint vagyon ez a könyvecske, vegyed jó néven,
    és légy jó egészségben!
    </div>
    <div style="text-align:right; font-style:italic; margin-top:0.6rem; color:#8b5a2b;">— Az Olvasóhoz, Kolozsvár, 1698</div>
</div>
""", unsafe_allow_html=True)

tripartit_path = resolve_path_candidates([os.path.join('data','Recept_halo__molekula_tripartit.csv')])
edges_path = resolve_path_candidates([os.path.join('data','recept_halo_edges.csv')])
hist_path = resolve_path_candidates([os.path.join('data','HistoricalRecipe_export.csv')])

if not (tripartit_path and edges_path and hist_path):
    st.warning("A szükséges adatfájlok nem találhatók. Helyezd a `data/` mappába a következőket: Recept_halo__molekula_tripartit.csv, recept_halo_edges.csv, HistoricalRecipe_export.csv")
else:
    tripartit = load_csv_flexible(tripartit_path, default_sep=';')
    edges = load_csv_flexible(edges_path, default_sep=',')
    historical = load_csv_flexible(hist_path, default_sep=',')

    label_col = next((c for c in tripartit.columns if c.lower() in ('label','name','title','node')), tripartit.columns[0] if len(tripartit.columns) else None)
    tripartit['Label'] = tripartit[label_col].astype(str).apply(strip_icon_ligatures) if label_col else tripartit.index.astype(str)
    type_col = next((c for c in tripartit.columns if 'type' in c.lower() or 'category' in c.lower()), None)
    tripartit['node_type'] = tripartit[type_col].astype(str).fillna('Egyéb') if type_col is not None else 'Egyéb'
    tripartit['norm'] = tripartit['Label'].apply(normalize_label)

    if 'norm_source' in edges.columns and 'norm_target' in edges.columns:
        srcs = edges['norm_source'].astype(str).tolist()
        tgts = edges['norm_target'].astype(str).tolist()
    else:
        srcs = edges.iloc[:,0].astype(str).tolist() if edges.shape[1] >= 1 else []
        tgts = edges.iloc[:,1].astype(str).tolist() if edges.shape[1] >= 2 else []

    def resolve_norm(val):
        if not isinstance(val, str):
            return ''
        return normalize_label(val)

    srcs = [resolve_norm(s) for s in srcs]
    tgts = [resolve_norm(t) for t in tgts]
    edge_list = [(s, t) for s, t in zip(srcs, tgts) if s and t]

    G = nx.Graph()
    for _, r in tripartit.iterrows():
        G.add_node(r['norm'], label=r['Label'], node_type=r['node_type'])
    G.add_edges_from(edge_list)

    ingredient_nodes = [n for n, d in G.nodes(data=True) if 'ingredient' in str(d.get('node_type','')).lower() or 'alapanyag' in str(d.get('node_type','')).lower()]

    deg = dict(G.degree())
    pr = nx.pagerank(G, alpha=0.85) if G.number_of_nodes() > 0 else {}
    bet = nx.betweenness_centrality(G) if G.number_of_nodes() > 0 else {}
    eig = {}
    try:
        eig = nx.eigenvector_centrality_numpy(G) if G.number_of_nodes() > 0 else {}
    except Exception:
        eig = {}

    def top_for(metric_dict, nodes, topn=10):
        return sorted(((n, metric_dict.get(n, 0)) for n in nodes), key=lambda x: x[1], reverse=True)[:topn]

    top_deg = top_for(deg, ingredient_nodes, 10)
    top_pr = top_for(pr, ingredient_nodes, 10)
    top_bet = top_for(bet, ingredient_nodes, 10)
    top_eig = top_for(eig, ingredient_nodes, 10)

    def readable(norm):
        return G.nodes[norm].get('label') if norm in G.nodes else norm

    molecules = [n for n, d in G.nodes(data=True) if 'molecule' in str(d.get('node_type','')).lower() or 'molekula' in str(d.get('node_type','')).lower()]
    recipes = [n for n, d in G.nodes(data=True) if 'dish' in str(d.get('node_type','')).lower() or 'recept' in str(d.get('node_type','')).lower() or 'recipe' in str(d.get('node_type','')).lower()]

    ing_to_mols = {ing: set() for ing in ingredient_nodes}
    ing_to_recipes = {ing: set() for ing in ingredient_nodes}
    for ing in ingredient_nodes:
        for mol in molecules:
            if G.has_edge(ing, mol):
                ing_to_mols[ing].add(mol)
        for rec in recipes:
            if G.has_edge(ing, rec):
                ing_to_recipes[ing].add(rec)

    pair_shared_mols = []
    pair_coocc = []
    ing_list = ingredient_nodes
    for i in range(len(ing_list)):
        for j in range(i + 1, len(ing_list)):
            a = ing_list[i]
            b = ing_list[j]
            shared = len(ing_to_mols[a] & ing_to_mols[b])
            coocc = len(ing_to_recipes[a] & ing_to_recipes[b])
            if shared > 0 or coocc > 0:
                pair_shared_mols.append(shared)
                pair_coocc.append(coocc)

    corr = None
    pval = None
    if len(pair_shared_mols) >= 10 and sum(pair_shared_mols) > 0:
        corr, pval = spearmanr(pair_shared_mols, pair_coocc)

    text_fields = []
    for c in ('original_text','text','instructions','description','ingredients','body'):
        if c in historical.columns:
            text_fields.append(c)
    if text_fields:
        bodies = historical[text_fields].astype(str).agg(' '.join, axis=1).apply(normalize_label)
    else:
        bodies = historical['title'].astype(str).apply(normalize_label) if 'title' in historical.columns else pd.Series([], dtype=str)
    avg_words_body = round(bodies.apply(lambda t: len(t.split())).mean() if len(bodies) > 0 else 0, 1)

    # Böjti receptek: használjuk a utils.fasting.is_fasting_title függvényt
    fasting_flags = []
    for idx, row in historical.iterrows():
        title_raw = row.get('title', '') or ''
        flag = False
        try:
            flag = bool(is_fasting_title(title_raw))
        except Exception:
            flag = False
        fasting_flags.append(flag)
    fast_count = sum(1 for f in fasting_flags if f)
    fast_pct = round(fast_count / len(historical) * 100, 1) if len(historical) > 0 else 0.0

    # METRICS SECTION - moved here (after the quote, before "Kutatási eredmények")
    st.markdown("---")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2.2rem; font-weight: bold; color: #8b5a2b;">{len(historical)}</div><div style="color:#4a3728; font-size:0.95rem; margin-top:0.5rem;">Történeti receptek (adatból)</div></div>', unsafe_allow_html=True)

    with metric_col2:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2.2rem; font-weight: bold; color: #8b5a2b;">{G.number_of_nodes()}</div><div style="color:#4a3728; font-size:0.95rem; margin-top:0.5rem;">Node (hálózat)</div></div>', unsafe_allow_html=True)

    with metric_col3:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2.2rem; font-weight: bold; color: #8b5a2b;">{avg_words_body}</div><div style="color:#4a3728; font-size:0.95rem; margin-top:0.5rem;">Átlag szószám (recept szövegtest)</div></div>', unsafe_allow_html=True)

    with metric_col4:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2.2rem; font-weight: bold; color: #8b5a2b;">{fast_pct}%</div><div style="color:#4a3728; font-size:0.95rem; margin-top:0.5rem;">Böjti receptek (detektálva)</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown("### Kutatási eredmények (adatok alapján)")
    st.markdown("**1) Mely alapanyagok voltak a legközpontibbak?**")

    deg_col, pr_col, bet_col = st.columns(3)

    with deg_col:
        st.markdown('<div class="list-card"><div class="list-title">Top 10 — Degree (kapcsolatok száma)</div>', unsafe_allow_html=True)
        for i, (n, v) in enumerate(top_deg, start=1):
            st.markdown(f'<div class="list-item">{i}. <strong>{readable(n)}</strong> — {int(v)}</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-top:8px; color:#4a3728;">A Degree megmutatja, hány közvetlen kapcsolat van egy alapanyagnak: minél nagyobb, annál több recepthez, molekulához vagy más alapanyaghoz kapcsolódott (azaz gyakrabban használták vagy sokoldalú volt).</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with pr_col:
        st.markdown('<div class="list-card"><div class="list-title">Top 10 — PageRank (hálózati befolyás)</div>', unsafe_allow_html=True)
        for i, (n, v) in enumerate(top_pr, start=1):
            st.markdown(f'<div class="list-item">{i}. <strong>{readable(n)}</strong> — {v:.6f}</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-top:8px; color:#4a3728;">A PageRank nemcsak a kapcsolatok számát nézi, hanem azok minőségét: ha egy alapanyag kapcsolatban áll más fontos alapanyagokkal, akkor magasabb a PageRank-je — ez a „befolyásosság" mutatója a hálózatban.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with bet_col:
        st.markdown('<div class="list-card"><div class="list-title">Top 10 — Betweenness (hidak)</div>', unsafe_allow_html=True)
        for i, (n, v) in enumerate(top_bet, start=1):
            st.markdown(f'<div class="list-item">{i}. <strong>{readable(n)}</strong> — {v:.6f}</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-top:8px; color:#4a3728;">A Betweenness azt jelenti, hogy egy alapanyag milyen gyakran van a legrövidebb utak „közepén" a hálózatban — ezek a csomópontok gyakran kötik össze a különböző ízvilágokat, vagy átjárót képeznek két csoport között.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**2) Van-e mérhető kapcsolat az íz-aroma molekulák és a történeti párosítások között?**")
    if corr is None:
        st.markdown("Nem volt elég páros adat a megbízható Spearman korreláció számítához (kevés közös molekula / páros).")
    else:
        st.markdown(f"Spearman rho = **{corr:.3f}**, p = **{pval:.3g}**")
        if pval < 0.05:
            st.markdown("Értékelés: statisztikailag szignifikáns korreláció — a közös molekulák száma részben magyarázza az együtt előfordulás gyakoriságát.")
        else:
            st.markdown("Értékelés: nincs szignifikáns korreláció — a molekuláris hasonlóság önmagában nem magyarázza a történeti párosításokat.")
        if corr is not None and corr < 0:
            st.markdown("""
            **Magyarázat laikusoknak:** A negatív Spearman-korreláció azt jelenti, hogy minél több közös aroma- (molekula) jelleg van két alapanyag között,
            annál ritkábban fordult elő történetileg, hogy együtt szerepeljenek ugyanabban a receptben. Ennek lehetséges okai:
            - **Kontrasztkészítés**: A szakácsok gyakran akartak ellentétes karaktereket egyesíteni (édes vs. sós, savas vs. zsíros), így különböző aromájú összetevőket párosítottak.
            - **Ritkaság / speciális használat**: Hasonló aromájú hozzávalókat lehet, hogy általában különféle, speciális fogásokban használtak, ezért ritkán szerepeltek együtt.
            - **Kulináris kultúra**: A korabeli receptek célja és szokásai befolyásolták, hogy miket párosítottak; a hasonló molekuláris profil nem feltétlenül vezet együtt használathoz.
            Röviden: a negatív kapcsolat nem jelenti, hogy az aroma fontos lenne; azt jelzi, hogy a hasonlóság gyakran nem vezetett együtthasználathoz a vizsgált korpuszban.
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**3) Mennyire közelíti meg az AI a történeti receptek stílusát és szerkezetét?**")
    st.markdown("""
Az AI nem tudja utánozni a történeti receptek stílusát.

A probléma: A generált receptek monoton, gépiesen ismétlődő mondatokat produkálnak ("majd ecettel főzve, majd mézzel párolva..."), amelyek semmiben nem hasonlítanak az eredeti történeti receptekre.

A számok ezt igazolják:

- Átlagos hasonlóság a történeti korpusszal: csak **28.7%**

- Egyetlen generált recept sem éri el a **60%**-os hasonlósági küszöböt

- Minden recept **71%** "újdonságot" mutat — ami itt azt jelenti, hogy teljesen más, mint az eredeti stílus

Mit jelent ez a gyakorlatban? Az AI képes címeket és alapanyagokat generálni, de a szöveg stílusa, szerkezete és hangvétele gépiesen ismétlődő sablon, nem pedig autentikus történeti nyelv. A "hagyma" receptben például 9-szer ismétlődik ugyanaz a szerkezet, ami egy valódi történeti receptben soha nem fordulna elő.

Konklúzió: Az AI jelen formájában nem alkalmas történeti receptek hiteles rekonstrukciójára - csak modern, sablonos utánzatokat hoz létre.
    """, unsafe_allow_html=True)

    st.markdown('<div class="large-quote">„A főzés az az a fajta művészet, amely a történelmi termékeket képes pillanatok alatt élvezetté varázsolni."<small>– Guy Savoy</small></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; margin-bottom: 0; padding: 1.2rem; background: linear-gradient(to bottom, #fffbf0, #fff9e6); border-radius: 8px;">
        <div style="font-size: 1.1rem; font-weight: bold; color: #2c1810; font-family: Georgia, serif; margin-bottom: 0.5rem;">
            Közrendek Ízhálója
        </div>
        <div style="color: #5c4033; font-size: 0.95rem; margin-bottom: 0.2rem;">
            Hálózatelemzés + Történeti Források + AI Generálás
        </div>
        <div style="color: #8b5a2b; font-size: 0.85rem;">
            © 2025 | Built with Streamlit, NetworkX, SciPy, OpenAI API, Claude, GrokAI & Open-source tools
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Scroll-to-top: anchor alapú, a fő DOM-ba injektálva (nem iframe) ---
    st.markdown("""
    <a href="#top-anchor" class="scroll-to-top" aria-label="Vissza a tetejére">↑</a>
    
    <style>
    .scroll-to-top {
        position: fixed;
        bottom: 50px;
        right: 30px;
        background: linear-gradient(135deg, #8b5a2b, #d4af37);
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: all 0.18s ease;
        z-index: 9999;
        text-decoration: none;
        font-size: 24px;
        font-weight: bold;
        line-height: 50px;
    }
    .scroll-to-top:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.35);
        background: linear-gradient(135deg, #d4af37, #8b5a2b);
    }
    </style>
    """, unsafe_allow_html=True)
