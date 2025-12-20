import os
from pathlib import Path
import re
from html import unescape
import unicodedata
import pandas as pd
import networkx as nx
from collections import defaultdict
from scipy.stats import spearmanr
import streamlit as st
from utils.fasting import FASTING_RECIPE_TITLES

st.set_page_config(page_title="A PROJEKTRŐL", page_icon="📜", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&display=swap');
[data-testid="stSidebar"] > div:first-child {
    background-color: #5c1a1a !important;
    font-family: 'Cinzel', serif !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] .st-expander,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div[data-testid$="-label"] {
    font-family: 'Cinzel', serif !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] span[data-testid="stIconMaterial"],
.span[data-testid="stIconMaterial"] {
    display: none !important;
}
[data-testid="stKeyboardShortcutButton"],
button[aria-label="Show keyboard shortcuts"],
button[aria-label="Show keyboard navigation"],
[data-testid^="stTooltip"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&display=swap');
[data-testid="stSidebar"] > div:first-child {
    background-color: #5c1a1a !important;
    font-family: 'Cinzel', serif !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] .st-expander,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div[data-testid$="-label"] {
    font-family: 'Cinzel', serif !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Főcím középre, középkori stílus */
    .main-title {
        text-align: center;
        color: #2c1810;
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        font-family: 'Georgia', serif;
    }
   
    .divider {
        width: 100px;
        height: 4px;
        background: linear-gradient(to right, #d4af37, #f0d98d, #d4af37);
        margin: 0 auto 3rem auto;
        border-radius: 2px;
    }
   
    /* Az Olvasóhoz idézet */
    .reader-quote {
        background: linear-gradient(to right, #fffbf0, #fff9e6);
        border-left: 8px solid #d4af37;
        padding: 3rem 2rem 3rem 4rem;
        font-style: italic;
        color: #5c4033;
        font-size: 1.2rem;
        line-height: 1.8;
        margin: 3rem 0;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.05);
        border-radius: 0 8px 8px 0;
    }
   
    .reader-quote .first-letter {
        float: left;
        font-size: 5rem;
        line-height: 1;
        font-weight: bold;
        margin-right: 0.5rem;
        color: #8b5a2b;
        font-family: 'Georgia', serif;
    }
   
    .signature {
        text-align: right;
        margin-top: 2rem;
        font-family: 'Georgia', serif;
        color: #8b5a2b;
        font-size: 0.95rem;
    }
   
    /* Fő szöveg stílus */
    .body-text {
        color: #4a3728;
        font-size: 1.1rem;
        line-height: 1.8;
        text-align: justify;
    }
   
    .body-text .first-letter-main {
        float: left;
        font-size: 4rem;
        line-height: 1;
        font-weight: bold;
        margin-right: 0.5rem;
        color: #8b5a2b;
        font-family: 'Georgia', serif;
    }
   
    /* Szekciócím */
    .section-title {
        color: #2c1810;
        font-size: 2rem;
        font-weight: bold;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        font-family: 'Georgia', serif;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
   
    /* Kiemelés doboz */
    .highlight-box {
        background: linear-gradient(to right, #fffbf0, #fff9e6);
        border-left: 4px solid #d4af37;
        padding: 2rem;
        margin: 2rem 0;
        font-style: italic;
        color: #5c4033;
        border-radius: 0 8px 8px 0;
    }
   
    /* Link stílus */
    a {
        color: #8b5a2b !important;
        text-decoration: underline;
    }
   
    a:hover {
        color: #d4af37 !important;
    }
   
    /* Scrollbar stílus */
    ::-webkit-scrollbar {
        width: 10px;
    }
   
    ::-webkit-scrollbar-track {
        background: #fffbf0;
    }
   
    ::-webkit-scrollbar-thumb {
        background: #d4af37;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    display: block;
    width: fit-content;
    margin: 0 auto; /* középre helyezés */
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
    margin: 1.5rem auto 3rem auto;
    border-radius: 2px;
"></div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="reader-quote">
    <span class="first-letter">E</span>z az én könyvecském nem siet az udvarokban való nagy konyhákhoz,
    ahol a szakácsok csak magoktól is jóízű étkeket tudnak főzni; hanem csak leginkább
    a becsületes közrendeknek, akik gyakorta szakács nélkül szűkölködnek, akar szolgálni…
    <br/><br/>
    Azért jámbor Olvasó, ha kedved szerint vagyon ez a könyvecske, vegyed jó néven,
    és légy jó egészségben!
    <div class="signature">— Az Olvasóhoz, Kolozsvár, 1698</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="body-text">
    <p>
        <span class="first-letter-main">A</span> Közrendek Ízhálója projekt célja,
        hogy modern technológia segítségével elevenítse fel a XVII. századi magyar gasztronómia
        elfeledett világát. A projekt alapját a híres "Szakácsmesterségnek könyvecskéje" képezi,
        amely 1698-ban jelent meg Kolozsváron, és az egyik legkorábbi ránk maradt magyar nyelvű
        nyomtatott szakácskönyv.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h3 class="section-title">
    📖 A Forrásmű
</h3>
""", unsafe_allow_html=True)

st.markdown("""
<div class="body-text">
    <p>
        A <a href="https://mek.oszk.hu/08300/08343/08343.htm" target="_blank" rel="noopener noreferrer">
        Szakácsmesterségnek könyvecskéje</a> receptjei nem pontos mennyiségeket, hanem arányokat és
        eljárásokat rögzítenek. A könyv kifejezetten a "becsületes közrendeknek" készült, akik gyakorta
        szakács nélkül szűkölködtek. Ez a <em>network science</em> (hálózatkutatás) szempontjából
        különösen izgalmas, hiszen az alapanyagok kapcsolódásai rajzolják ki a kor ízlésvilágának térképét.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h3 class="section-title">
    🕸️ Hálózatelemzés és Gasztronómia
</h3>
""", unsafe_allow_html=True)

st.markdown("""
<div class="body-text">
    <p>
        Barabási Albert-László <em>Network Science</em> című könyvében bemutatja a <strong>flavor network</strong>
        módszertant: egy háromrétegű hálózatot, amely recepteket, alapanyagokat és ízmolekulákat kapcsol össze.
        A modell szerint két alapanyag akkor kerül közel egymáshoz a hálózatban, ha jelentős számú közös
        ízkomponensük van.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="highlight-box">
    "Az ízek nem véletlenszerűen találkoznak, hanem rejtett hálózatok mentén szerveződnek harmóniába."
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="body-text">
    <p>
        A XVII. századi magyar konyha jellegzetes alapanyag-kombinációinak (sáfrány-gyömbér-bors-ecet-gyümölcs)
        flavor network szempontú elemzését tűztem ki célul Barabási Albert-László <em>Hálózatok Tudománya</em>
        című könyve nyomán, abból ihletődve. Ez a weboldal a hálózatelemzéses statisztikai számítások
        (<strong>Nodes, Edges, Eccentricity, Closeness Centrality, Harmonic Closeness Centrality,
        Betweenness Centrality, Degree, Eigen Centrality, PageRank</strong>, stb.) alapján igyekszik
        AI segítségével a meglévő receptek stílusa és összetevői, molekulái mellett és alapján is új,
        de stílusban illeszkedő recepteket generálni, összekötve ezzel is a múltat a jelennel.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<h3 class="section-title">
    🤖 Technikai Megvalósítás
</h3>
""", unsafe_allow_html=True)

st.markdown("""
<div class="body-text">
    <p>
        A projekt modern mesterséges intelligencia és hálózattudomány eszközeit használja:
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div style="background: #fffbf0; padding: 1.5rem; border-radius: 8px; border: 2px solid #d4af37;">
        <h4 style="color: #2c1810; font-family: Georgia, serif; margin-bottom: 1rem;">📊 Hálózatelemzés</h4>
        <ul style="color: #4a3728; line-height: 1.8;">
            <li><strong>Tripartit hálózat:</strong> Receptek ↔ Alapanyagok ↔ Molekulák</li>
            <li><strong>Degree Centrality:</strong> Központi alapanyagok azonosítása</li>
            <li><strong>Betweenness:</strong> "Híd" szerepű összetevők</li>
            <li><strong>PageRank:</strong> Kulcsfontosságú node-ok rangsorolása</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="background: #fffbf0; padding: 1.5rem; border-radius: 8px; border: 2px solid #d4af37;">
        <h4 style="color: #2c1810; font-family: Georgia, serif; margin-bottom: 1rem;">🧠 AI Receptgenerálás</h4>
        <ul style="color: #4a3728; line-height: 1.8;">
            <li><strong>GPT-5.1 Prompting:</strong> Strukturált, grounding-alapú</li>
            <li><strong>Adaptív hosszúság:</strong> Korpusz-vezérelt (40-160 szó)</li>
            <li><strong>Network-informed:</strong> Degree-súlyozott döntések</li>
            <li><strong>Confidence score:</strong> Transzparens megbízhatóság</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<h3 class="section-title">
    📚 Az Adatbázis
</h3>
""", unsafe_allow_html=True)

def strip_icon_ligatures_simple(s):
    if not isinstance(s, str):
        return ""
    s = unescape(s)
    s = re.sub(r"<[^>]+>", "", s)
    return s.strip()

def strip_icon_ligatures(s):
    if not isinstance(s, str): return ""
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'<[^>]+>', '', s)
    s = re.sub(r'[_\-\s]+', ' ', s).strip()
    return s

def normalize_label(s):
    if not isinstance(s, str): return ''
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

hist_path = resolve_historical_csv_path()
fasting_pct_display = "—"
if hist_path:
    try:
        hist_df = pd.read_csv(hist_path, sep=',', encoding='utf-8', on_bad_lines='skip')
    except Exception:
        try:
            hist_df = pd.read_csv(hist_path, sep=';', encoding='utf-8', on_bad_lines='skip')
        except Exception:
            hist_df = pd.read_csv(hist_path, sep=None, engine='python', encoding='latin1', on_bad_lines='skip')
    if 'title' in hist_df.columns:
        titles = hist_df['title'].apply(lambda x: strip_icon_ligatures_simple(x) if isinstance(x, str) else "")
        total = len(titles)
        if total > 0:
            fasting_count = sum(1 for t in titles if t in FASTING_RECIPE_TITLES)
            pct = round(fasting_count / total * 100)
            fasting_pct_display = f"{pct}%"
else:
    fasting_pct_display = "N/A"

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
with metric_col1:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: #fffbf0; border-radius: 8px; border: 2px solid #d4af37;">
        <div style="font-size: 2.5rem; font-weight: bold; color: #8b5a2b;">330</div>
        <div style="color: #4a3728; font-size: 1rem; margin-top: 0.5rem;">Történeti Recept</div>
    </div>
    """, unsafe_allow_html=True)
with metric_col2:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: #fffbf0; border-radius: 8px; border: 2px solid #d4af37;">
        <div style="font-size: 2.5rem; font-weight: bold; color: #8b5a2b;">838</div>
        <div style="color: #4a3728; font-size: 1rem; margin-top: 0.5rem;">Node (Hálózat)</div>
    </div>
    """, unsafe_allow_html=True)
with metric_col3:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: #fffbf0; border-radius: 8px; border: 2px solid #d4af37;">
        <div style="font-size: 2.5rem; font-weight: bold; color: #8b5a2b;">70.7</div>
        <div style="color: #4a3728; font-size: 1rem; margin-top: 0.5rem;">Átlag Szószám</div>
    </div>
    """, unsafe_allow_html=True)
with metric_col4:
    st.markdown(f"""
    <div style="text-align: center; padding: 1.5rem; background: #fffbf0; border-radius: 8px; border: 2px solid #d4af37;">
        <div style="font-size: 2.5rem; font-weight: bold; color: #8b5a2b;">{fasting_pct_display}</div>
        <div style="color: #4a3728; font-size: 1rem; margin-top: 0.5rem;">Böjti Receptek</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<h3 class="section-title">
    📖 Hivatkozások
</h3>
""", unsafe_allow_html=True)

st.markdown("""
<div class="body-text">
    <ul style="line-height: 2;">
        <li>
            <strong>Barabási Albert-László:</strong> <em>Network Science</em> (2016)
            - <a href="http://networksciencebook.com/" target="_blank">networksciencebook.com</a>
        </li>
        <li>
            <strong>Szakácsmesterségnek könyvecskéje</strong> (1698, Kolozsvár)
            - <a href="https://mek.oszk.hu/08300/08343/08343.htm#252" target="_blank">Magyar Elektronikus Könyvtár</a>
        </li>
        <li>
            <strong>Ahn, Y. Y., et al.:</strong> "Flavor network and the principles of food pairing"
            - <em>Scientific Reports</em> (2011)
        </li>
        <li>
            <strong>OpenAI:</strong> GPT-5.1 Prompting Guide
            - <a href="https://cookbook.openai.com/examples/gpt-5/gpt-5-1_prompting_guide" target="_blank">cookbook.openai.com</a>
        </li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<h3 class="section-title">
    🔬 Kutatási Kérdések
</h3>
""", unsafe_allow_html=True)

st.markdown("""
<div class="body-text">
    <p><strong>Jelenlegi fókusz:</strong></p>
    <ol style="line-height: 2;">
        <li>Mely alapanyagok voltak a legközpontibbak a XVII. századi magyar konyhában?</li>
        <li>Van-e mérhető kapcsolat az íz-aroma molekulák és a történeti párosítások között?</li>
        <li>Hogyan térképezhető fel a böjti konyha a hálózatban?</li>
        <li>Mennyire közelíti meg az AI a történeti receptek stílusát és szerkezetét?</li>
    </ol>
</div>
<div class="body-text">
    <p><strong>Jövőbeli irányok:</strong></p>
    <ol style="line-height: 2;">
        <li><strong>Temporal:</strong> Időbeli változások (XVI. vs. XVIII. század)</li>
        <li><strong>Regionális:</strong> Földrajzi különbségek (Erdély, Dunántúl, Felvidék)</li>
        <li><strong>Evaluation:</strong> AI minőségellenőrzés human evaluátorokkal</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Inserted analysis summary here
tripartit_path = resolve_tripartit_path()
edges_path = resolve_edges_path()
hist_path = resolve_historical_csv_path()
if not (tripartit_path and edges_path and hist_path):
    st.warning("A hálózati / történeti CSV fájlok nem találhatók. Ellenőrizd, hogy a projekt `data/` mappájában vannak-e:\n- Recept_halo__molekula_tripartit.csv\n- recept_halo_edges.csv\n- HistoricalRecipe_export.csv")
else:
    tripartit = pd.read_csv(tripartit_path, delimiter=';', encoding='utf-8', on_bad_lines='skip')
    edges = pd.read_csv(edges_path, delimiter=',', encoding='utf-8', on_bad_lines='skip')
    historical = pd.read_csv(hist_path, encoding='utf-8', on_bad_lines='skip')
    # standardise labels & types
    label_col = next((c for c in tripartit.columns if c.lower() in ('label','name','title')), tripartit.columns[0])
    tripartit['Label'] = tripartit[label_col].astype(str).apply(strip_icon_ligatures)
    type_col = next((c for c in tripartit.columns if 'type' in c.lower() or 'category' in c.lower()), None)
    tripartit['node_type'] = tripartit[type_col].astype(str).fillna('Egyéb') if type_col is not None else 'Egyéb'
    tripartit['norm'] = tripartit['Label'].apply(normalize_label)
    node_norm_map = {r['norm']: r for _, r in tripartit.iterrows()}
    # edges
    if 'norm_source' in edges.columns and 'norm_target' in edges.columns:
        srcs = edges['norm_source'].astype(str).tolist()
        tgts = edges['norm_target'].astype(str).tolist()
    else:
        srcs = edges.iloc[:,0].astype(str).tolist()
        tgts = edges.iloc[:,-1].astype(str).tolist()
    def resolve_norm(val):
        if not isinstance(val, str): return ''
        v = normalize_label(val)
        return v
    srcs = [resolve_norm(s) for s in srcs]
    tgts = [resolve_norm(t) for t in tgts]
    edge_list = [(s,t) for s,t in zip(srcs,tgts) if s and t]
    # build graph
    G = nx.Graph()
    for _, r in tripartit.iterrows():
        G.add_node(r['norm'], label=r['Label'], node_type=r['node_type'])
    G.add_edges_from(edge_list)
    # determine ingredient nodes
    ingredient_nodes = [n for n,d in G.nodes(data=True) if 'alapanyag' in str(d.get('node_type','')).lower() or 'ingredient' in str(d.get('node_type','')).lower()]
    if not ingredient_nodes:
        ingredient_nodes = [n for n,d in G.nodes(data=True) if ('molekula' not in str(d.get('node_type','')).lower()) and ('recept' not in str(d.get('node_type','')).lower())]
    # centralities
    deg = dict(G.degree())
    pr = nx.pagerank(G, alpha=0.85) if G.number_of_nodes()>0 else {}
    bet = nx.betweenness_centrality(G) if G.number_of_nodes()>0 else {}
    eig = {}
    try:
        eig = nx.eigenvector_centrality_numpy(G) if G.number_of_nodes()>0 else {}
    except Exception:
        eig = {}
    def top_for(metric_dict, nodes, topn=10):
        return sorted(((n, metric_dict.get(n,0)) for n in nodes), key=lambda x: x[1], reverse=True)[:topn]
    top_deg = top_for(deg, ingredient_nodes, 10)
    top_pr = top_for(pr, ingredient_nodes, 10)
    top_bet = top_for(bet, ingredient_nodes, 10)
    top_eig = top_for(eig, ingredient_nodes, 10)
    def readable(norm):
        return G.nodes[norm].get('label') if norm in G.nodes else norm
    # molecule vs pairing correlation
    molecules = [n for n,d in G.nodes(data=True) if 'molekula' in str(d.get('node_type','')).lower() or 'molecule' in str(d.get('node_type','')).lower()]
    recipes = [n for n,d in G.nodes(data=True) if 'recept' in str(d.get('node_type','')).lower() or 'dish' in str(d.get('node_type','')).lower()]
    ing_to_mols = {ing:set() for ing in ingredient_nodes}
    ing_to_recipes = {ing:set() for ing in ingredient_nodes}
    for ing in ingredient_nodes:
        for mol in molecules:
            if G.has_edge(ing,mol): ing_to_mols[ing].add(mol)
        for rec in recipes:
            if G.has_edge(ing,rec): ing_to_recipes[ing].add(rec)
    pair_shared_mols=[]
    pair_coocc=[]
    ing_list = ingredient_nodes
    for i in range(len(ing_list)):
        for j in range(i+1, len(ing_list)):
            a=ing_list[i]; b=ing_list[j]
            shared = len(ing_to_mols[a]&ing_to_mols[b])
            coocc = len(ing_to_recipes[a]&ing_to_recipes[b])
            if shared>0 or coocc>0:
                pair_shared_mols.append(shared); pair_coocc.append(coocc)
    corr=None; pval=None
    if len(pair_shared_mols)>=10 and sum(pair_shared_mols)>0:
        corr,pval = spearmanr(pair_shared_mols, pair_coocc)
    # fasting pct (keyword fallback)
    fast_kws = ['böjt','böjti','post','fast','lenten']
    titles = historical['title'].astype(str).apply(strip_icon_ligatures).str.lower()
    fast_count = titles.apply(lambda s: any(k in s for k in fast_kws)).sum()
    fast_pct = round(fast_count/len(titles)*100,1) if len(titles)>0 else None
    # render results
    st.markdown("### Kutatási eredmények (adatok alapján)")
    st.markdown("**1) Mely alapanyagok voltak a legközpontibbak?**")
    st.markdown("Top 10 — Degree (kapcsolatok száma):")
    for n,v in top_deg:
        st.markdown(f"- **{readable(n)}** — Degree: {int(v)}")
    st.markdown("Top 10 — PageRank (hálózati befolyás):")
    for n,v in top_pr:
        st.markdown(f"- **{readable(n)}** — PageRank: {v:.6f}")
    st.markdown("Top 10 — Betweenness (hidak):")
    for n,v in top_bet:
        st.markdown(f"- **{readable(n)}** — Betweenness: {v:.6f}")
    st.markdown("---")
    st.markdown("**2) Van-e mérhető kapcsolat az íz-aroma molekulák és a történeti párosítások között?**")
    if corr is None:
        st.markdown("Nem volt elég páros adat a megbízható Spearman korreláció számításhoz (kevés közös molekula / páros).")
    else:
        st.markdown(f"Spearman rho = **{corr:.3f}**, p = **{pval:.3g}**")
        if pval < 0.05:
            st.markdown("Értékelés: statisztikailag szignifikáns korreláció — a közös molekulák száma részben magyarázza az együtt előfordulás gyakoriságát.")
        else:
            st.markdown("Értékelés: nincs szignifikáns korreláció — a molekuláris hasonlóság önmagában nem magyarázza a történeti párosításokat.")
    st.markdown("---")
    st.markdown("**3) Hogyan térképezhető fel a böjti konyha a hálózatban?**")
    if fast_pct is None:
        st.markdown("A történeti recept-fájl nem tartalmazható/elérhető volt; böjti százalék: N/A")
    else:
        st.markdown(f"Böjti receptek (kulcsszó-fallback alapján): **{fast_pct}%** a teljes korpuszból.")
        st.markdown("Javaslat: szűrjük a `historical` fájlt a böjti címekre és nézzük meg a hozzájuk kapcsolódó alapanyagok előfordulását, központosságát (degree, PageRank), és klaszterezését — az About oldalra rövid toplistákat lehet kitenni.")
    st.markdown("---")
    st.markdown("**4) Mennyire közelíti meg az AI a történeti receptek stílusát és szerkezetét?**")
    st.markdown("- Az AI-alapú generálás `novelty` / `similarity` metrikával mérhető: javasolt módszer SequenceMatcher/levenshtein alapú hasonlóság a történeti corpus-szal, majd `novelty = 1 - max_similarity` minden generációra.")
    st.markdown("- Ajánlott küszöb: ha similarity > 0.6 -> új generálás vagy erősebb prompt grounding.")
    st.markdown("- Ha szeretnéd, beépítem ide a generálások példáit + a historical-hoz mért similarity statisztikát is (ha engedélyezed a generált receptek futtatását).")
    st.markdown("---")
    st.markdown("**Megjegyzés / következő lépések**")
    st.markdown("- Ha szeretnéd, exportálom a fenti toplistákat CSV-be és megjelenítem belőle a `About` oldalon táblázatosan.")
    st.markdown("- Ha szeretnéd, lefuttatom a teljes elemzést (`analysis_outputs/*.csv`) és beimportálom itt a konkrét toplistákat (ha a szerveren írási jogaid megvannak).")

st.markdown("---")

st.markdown("""
<div class="highlight-box" style="text-align: center; font-size: 1.3rem;">
    „A főzés az az a fajta művészet, amely a történelmi termékeket képes pillanatok alatt élvezetté varázsolni.”
                                                                                                    – Guy Savoy
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-top: 4rem; padding: 2rem; background: linear-gradient(to bottom, #fffbf0, #fff9e6); border-radius: 8px;">
    <div style="font-size: 1.5rem; font-weight: bold; color: #2c1810; font-family: Georgia, serif; margin-bottom: 1rem;">
        Közrendek Ízhálója
    </div>
    <div style="color: #5c4033; font-size: 1rem; margin-bottom: 0.5rem;">
        Hálózatelemzés + Történeti Források + AI Generálás
    </div>
    <div style="color: #8b5a2b; font-size: 0.9rem;">
        © 2025 | Built with Streamlit, NetworkX, Plotly, Anthropic's Claude, GrokAI & OpenAI GPTs
    </div>
</div>
""", unsafe_allow_html=True)
