import os
import re
import unicodedata
from html import unescape
from pathlib import Path

import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
import streamlit as st
from difflib import SequenceMatcher

try:
    from utils.fasting import FASTING_RECIPE_TITLES, FASTING_KEYWORDS, classify_fasting_text
except Exception:
    try:
        from utils.fasting import FASTING_RECIPE_TITLES
    except Exception:
        FASTING_RECIPE_TITLES = []
    FASTING_KEYWORDS = ['böjt', 'post', 'fast', 'fasta', 'luszt', 'lent']
    classify_fasting_text = None

def strip_icon_ligatures(s):
    if not isinstance(s, str):
        return ""
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

def resolve_historical_csv_path():
    script_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(script_dir, 'data', 'HistoricalRecipe_export.csv'),
        os.path.join(os.getcwd(), 'data', 'HistoricalRecipe_export.csv'),
        os.path.join(os.path.abspath(os.path.join(script_dir, '..')), 'data', 'HistoricalRecipe_export.csv'),
        'data/HistoricalRecipe_export.csv',
        'HistoricalRecipe_export.csv'
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def resolve_tripartit_path():
    script_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(script_dir, 'data', 'Recept_halo__molekula_tripartit.csv'),
        os.path.join(os.getcwd(), 'data', 'Recept_halo__molekula_tripartit.csv'),
        'data/Recept_halo__molekula_tripartit.csv',
        'Recept_halo__molekula_tripartit.csv'
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def resolve_edges_path():
    script_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(script_dir, 'data', 'recept_halo_edges.csv'),
        os.path.join(os.getcwd(), 'data', 'recept_halo_edges.csv'),
        'data/recept_halo_edges.csv',
        'recept_halo_edges.csv'
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def sequence_similarity(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

st.set_page_config(page_title="A PROJEKTRŐL", page_icon="📜", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&display=swap');
[data-testid="stSidebar"] > div:first-child {
    background-color: #5c1a1a !important;
    font-family: 'Cinzel', serif !important;
    color: #ffffff !important;
}
.list-card {
    background: #fffaf2;
    border: 1px solid #e6d2a3;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 12px;
}
.list-title {
    font-weight: 700;
    color: #2c1810;
    font-size: 1.05rem;
    margin-bottom: 8px;
}
.list-item {
    margin: 6px 0;
    line-height: 1.4;
}
.metric-card {
    text-align: center;
    padding: 1.5rem;
    background: #fffbf0;
    border-radius: 8px;
    border: 2px solid #d4af37;
}
.reader-quote {
    background: linear-gradient(to right, #fff8e6, #fff5da);
    border: 2px solid #d4af37;
    padding: 2rem 2.5rem;
    color: #5c4033;
    font-size: 1.05rem;
    line-height: 1.7;
    border-radius: 10px;
    position: relative;
    margin-bottom: 1.5rem;
}
.reader-quote .first-letter {
    float: left;
    font-size: 5.2rem;
    line-height: 1;
    font-weight: 700;
    margin-right: 0.4rem;
    color: #8b5a2b;
    font-family: 'Georgia', serif;
}
.reader-quote .signature {
    text-align: right;
    margin-top: 1rem;
    font-style: italic;
    color: #8b5a2b;
    font-size: 0.95rem;
    font-family: 'Georgia', serif;
}
.section-title {
    color: #2c1810;
    font-size: 1.35rem;
    font-weight: bold;
    margin-top: 1.2rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.highlight-box {
    background: linear-gradient(to right, #fffbf0, #fff9e6);
    border-left: 4px solid #d4af37;
    padding: 1rem;
    margin: 1.2rem 0;
    color: #5c4033;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    display: block;
    width: fit-content;
    margin: 0 auto;
    padding: 0.5rem 2rem;
    background: linear-gradient(to right, #5c070d, #840a13);
    border-radius: 8px;
    text-align: center;
">
    <h1 style="font-family: Cinzel, serif; color: #ffffff; margin: 0;">A PROJEKTRŐL</h1>
</div>
<div style="
    width: 100px;
    height: 4px;
    background: linear-gradient(to right, #d4af37, #f0d98d, #d4af37);
    margin: 1rem auto 2rem auto;
    border-radius: 2px;
"></div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="reader-quote">
    <span class="first-letter">E</span>z az én könyvecském nem siet az udvarokban való nagy konyhákhoz,
    ahol a szakácsok csak magoktól is jóízű étkeket tudnak főzni; hanem csak leginkább
    a becsületes közrendeknek, akik gyakorta szakács nélkül szűkölködnek, akar szolgálni…
    <div style="margin-top:0.8rem;">
    Azért jámbor Olvasó, ha kedved szerint vagyon ez a könyvecske, vegyed jó néven,
    és légy jó egészségben!
    </div>
    <div class="signature">— Az Olvasóhoz, Kolozsvár, 1698</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-title">📖 Rövid leírás</div>
<div style="color: #4a3728; font-size:1rem; line-height:1.7;">
    A Közrendek Ízhálója projekt célja, hogy modern technológiával elevenítse fel a XVII. századi magyar gasztronómia világát.
</div>
""", unsafe_allow_html=True)

tripartit_path = resolve_tripartit_path()
edges_path = resolve_edges_path()
hist_path = resolve_historical_csv_path()

if not (tripartit_path and edges_path and hist_path):
    st.warning("A szükséges CSV fájlok nem találhatók. Ellenőrizd, hogy a `data/` mappában vannak-e:\n- Recept_halo__molekula_tripartit.csv\n- recept_halo_edges.csv\n- HistoricalRecipe_export.csv")
else:
    tripartit = pd.read_csv(tripartit_path, delimiter=';', encoding='utf-8', on_bad_lines='skip')
    edges = pd.read_csv(edges_path, delimiter=',', encoding='utf-8', on_bad_lines='skip')
    historical = pd.read_csv(hist_path, encoding='utf-8', on_bad_lines='skip')

    label_col = next((c for c in tripartit.columns if c.lower() in ('label','name','title')), tripartit.columns[0])
    tripartit['Label'] = tripartit[label_col].astype(str).apply(strip_icon_ligatures)
    type_col = next((c for c in tripartit.columns if 'type' in c.lower() or 'category' in c.lower()), None)
    tripartit['node_type'] = tripartit[type_col].astype(str).fillna('Egyéb') if type_col is not None else 'Egyéb'
    tripartit['norm'] = tripartit['Label'].apply(normalize_label)

    if 'norm_source' in edges.columns and 'norm_target' in edges.columns:
        srcs = edges['norm_source'].astype(str).tolist()
        tgts = edges['norm_target'].astype(str).tolist()
    else:
        srcs = edges.iloc[:,0].astype(str).tolist()
        tgts = edges.iloc[:,1].astype(str).tolist()

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

    ingredient_nodes = [n for n, d in G.nodes(data=True) if 'ingredient' in str(d.get('node_type', '')).lower() or 'alapanyag' in str(d.get('node_type', '')).lower()]

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

    molecules = [n for n, d in G.nodes(data=True) if 'molecule' in str(d.get('node_type', '')).lower() or 'molekula' in str(d.get('node_type', '')).lower()]
    recipes = [n for n, d in G.nodes(data=True) if 'dish' in str(d.get('node_type', '')).lower() or 'recept' in str(d.get('node_type', '')).lower() or 'recipe' in str(d.get('node_type', '')).lower()]

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

    fasting_set = {normalize_label(t) for t in FASTING_RECIPE_TITLES}
    text_fields = []
    for c in ('text', 'instructions', 'description', 'ingredients', 'body'):
        if c in historical.columns:
            text_fields.append(c)
    fasting_flags = []
    for idx, row in historical.iterrows():
        title = normalize_label(str(row.get('title', '')))
        combined_text = title
        for c in text_fields:
            combined_text = combined_text + ' ' + normalize_label(str(row.get(c, '')))
        is_fasting = False
        if title in fasting_set:
            is_fasting = True
        else:
            for kw in (FASTING_KEYWORDS if FASTING_KEYWORDS else []):
                if kw in combined_text:
                    is_fasting = True
                    break
        if classify_fasting_text is not None:
            try:
                clf_res = classify_fasting_text(title + ' ' + combined_text)
                if isinstance(clf_res, bool):
                    is_fasting = is_fasting or clf_res
                elif isinstance(clf_res, (int, float)) and clf_res >= 0.5:
                    is_fasting = True
            except Exception:
                pass
        fasting_flags.append(is_fasting)
    fast_count = sum(1 for f in fasting_flags if f)
    fast_pct = round(fast_count / len(historical) * 100, 1) if len(historical) > 0 else 0.0

    if text_fields:
        bodies = historical[text_fields].astype(str).agg(' '.join, axis=1).apply(normalize_label)
    else:
        bodies = historical['title'].astype(str).apply(normalize_label)
    avg_words_body = round(bodies.apply(lambda t: len(t.split())).mean() if len(bodies) > 0 else 0, 1)

    st.markdown("### Kutatási eredmények (adatok alapján)")
    st.markdown("**1) Mely alapanyagok voltak a legközpontibbak?**")

    deg_col, pr_col, bet_col = st.columns(3)

    with deg_col:
        st.markdown('<div class="list-card"><div class="list-title">Top 10 — Degree (kapcsolatok száma)</div>', unsafe_allow_html=True)
        for i, (n, v) in enumerate(top_deg, start=1):
            st.markdown(f'<div class="list-item">{i}. <strong>{readable(n)}</strong> — {int(v)}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with pr_col:
        st.markdown('<div class="list-card"><div class="list-title">Top 10 — PageRank (hálózati befolyás)</div>', unsafe_allow_html=True)
        for i, (n, v) in enumerate(top_pr, start=1):
            st.markdown(f'<div class="list-item">{i}. <strong>{readable(n)}</strong> — {v:.6f}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with bet_col:
        st.markdown('<div class="list-card"><div class="list-title">Top 10 — Betweenness (hidak)</div>', unsafe_allow_html=True)
        for i, (n, v) in enumerate(top_bet, start=1):
            st.markdown(f'<div class="list-item">{i}. <strong>{readable(n)}</strong> — {v:.6f}</div>', unsafe_allow_html=True)
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
        if corr is not None:
            if corr < 0:
                st.markdown("""
                **Magyarázat laikusoknak:** A negatív Spearman-korreláció azt jelenti, hogy minél több közös aroma- (molekula) jelleg van két alapanyag között,
                annál ritkábban fordult elő történetileg, hogy együtt szerepeljenek ugyanabban a receptben. Ennek több magyarázata lehet:
                - **Komplementer ízek**: A szakácsok gyakran kombinálnak ellentétes karakterű alapanyagokat (például édes és sós, savas és zsíros), hogy kontrasztot hozzanak létre. Ha két alapanyag nagyon hasonló aromájú, kevésbé adnak hozzá új dimenziót.
                - **Ritkaság és státusz**: Különleges, hasonló aromájú hozzávalókat lehet, hogy általában különféle, ritkább ételekhez használtak, így kevésbé kerültek párba.
                - **Kulináris szokások**: A korabeli receptek ízlését, készítési módszereit és elérhető hozzávalókat befolyásolta a kultúra; a hasonló aromájú alapanyagokat lehet, hogy különböző fogásokban használták.
                Röviden: a negatív kapcsolat nem jelenti, hogy az aroma ne számítana; inkább azt mutatja, hogy a közös molekulák nem vezettek gyakori közös használathoz a vizsgált receptkorpuszban.
                """, unsafe_allow_html=True)

    st.markdown("---")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2.2rem; font-weight: bold; color: #8b5a2b;">{len(historical)}</div><div style="color:#4a3728; font-size:0.95rem; margin-top:0.5rem;">Történeti receptek</div></div>', unsafe_allow_html=True)

    with metric_col2:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2.2rem; font-weight: bold; color: #8b5a2b;">{G.number_of_nodes()}</div><div style="color:#4a3728; font-size:0.95rem; margin-top:0.5rem;">Node (hálózat)</div></div>', unsafe_allow_html=True)

    with metric_col3:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2.2rem; font-weight: bold; color: #8b5a2b;">{avg_words_body}</div><div style="color:#4a3728; font-size:0.95rem; margin-top:0.5rem;">Átlag szószám (recept szövegtest)</div></div>', unsafe_allow_html=True)

    with metric_col4:
        st.markdown(f'<div class="metric-card"><div style="font-size: 2.2rem; font-weight: bold; color: #8b5a2b;">{fast_pct}%</div><div style="color:#4a3728; font-size:0.95rem; margin-top:0.5rem;">Böjti receptek (detektálva)</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 4) Mennyire közelíti meg az AI a történeti receptek stílusát és szerkezetét?")
    st.markdown("Az alábbi eszközzel beilleszthetsz AI által generált recept(eke)t, és megmérjük a hasonlóságot a történeti korpusszal. Ha a similarity > 0.6, javasolt újragenerálni vagy erősebb groundingot alkalmazni.")

    gen_input = st.text_area("Illeszd be ide az AI által generált receptet(eke)t (különítsd el '---' vonallal több recept esetén):", height=220)
    uploaded = st.file_uploader("Vagy tölts fel txt fájlt (opcionális)", type=['txt'], accept_multiple_files=False)
    if uploaded is not None:
        try:
            content = uploaded.read().decode('utf-8')
            if gen_input.strip():
                gen_input = gen_input + "\n\n---\n\n" + content
            else:
                gen_input = content
        except Exception:
            pass

    corpus_texts = bodies.tolist() if len(bodies) > 0 else historical['title'].astype(str).apply(normalize_label).tolist()

    generated_list = [g.strip() for g in gen_input.split('---') if g.strip()]
    results = []
    for gen in generated_list:
        norm_gen = normalize_label(gen)
        sims = [sequence_similarity(norm_gen, c) for c in corpus_texts]
        max_sim = max(sims) if sims else 0.0
        mean_sim = sum(sims) / len(sims) if sims else 0.0
        novelty = 1.0 - max_sim
        results.append({'generated': gen, 'max_similarity': max_sim, 'mean_similarity': mean_sim, 'novelty': novelty})

    if generated_list:
        for i, r in enumerate(results, start=1):
            st.markdown(f"**Recept {i}**")
            st.markdown(f"- Legnagyobb similarity a korpusszal: **{r['max_similarity']:.3f}**")
            st.markdown(f"- Átlag similarity: **{r['mean_similarity']:.3f}**")
            st.markdown(f"- Novelty (1 - max_similarity): **{r['novelty']:.3f}**")
            if r['max_similarity'] > 0.6:
                st.warning("A similarity > 0.6. Javasolt az újragenerálás vagy a prompt grounding erősítése (több kontextus/azonosító példa a történeti stílusról).")
            else:
                st.success("A generált recept elég eltérőnek tűnik a korpuszhoz képest (novelty magas).")
    else:
        st.info("Nincsenek generált receptek bemeneti mezőben. Illessz be szöveget a fenti mezőbe vagy tölts fel txt fájlt.")

    st.markdown("---")
    st.markdown('<div class="highlight-box" style="text-align:center; font-size:1.1rem;">„A főzés az az a fajta művészet, amely a történelmi termékeket képes pillanatok alatt élvezetté varázsolni.” – Guy Savoy</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1.2rem; background: linear-gradient(to bottom, #fffbf0, #fff9e6); border-radius: 8px;">
        <div style="font-size: 1.1rem; font-weight: bold; color: #2c1810; font-family: Georgia, serif; margin-bottom: 0.5rem;">
            Közrendek Ízhálója
        </div>
        <div style="color: #5c4033; font-size: 0.95rem; margin-bottom: 0.2rem;">
            Hálózatelemzés + Történeti Források + AI Generálás
        </div>
        <div style="color: #8b5a2b; font-size: 0.85rem;">
            © 2025 | Built with Streamlit, NetworkX, SciPy & Open-source tools
        </div>
    </div>
    """, unsafe_allow_html=True)
