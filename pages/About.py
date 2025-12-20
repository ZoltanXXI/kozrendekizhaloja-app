import os
import re
import json
import unicodedata
from html import unescape
from pathlib import Path
from difflib import SequenceMatcher
import random
import textwrap
import time

import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
import streamlit as st

try:
    import openai
except Exception:
    openai = None

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
[data-testid="stSidebar"] > div:first-child { background-color: #5c1a1a !important; font-family: 'Cinzel', serif !important; color: #ffffff !important; }
[data-testid="stSidebar"] button, [data-testid="stSidebar"] .st-expander, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div[data-testid$="-label"] { font-family: 'Cinzel', serif !important; color: #ffffff !important; }
[data-testid="stSidebar"] span[data-testid="stIconMaterial"], .span[data-testid="stIconMaterial"] { display: none !important; }
.reader-quote { background: linear-gradient(to right, #fff8e6, #fff5da); border: 2px solid #d4af37; padding: 2rem 2.5rem; color: #5c4033; font-size: 1.05rem; line-height: 1.7; border-radius: 10px; position: relative; margin-bottom: 1.5rem; }
.reader-quote .first-letter { float: left; font-size: 5.6rem; line-height: 1; font-weight: 700; margin-right: 0.5rem; color: #8b5a2b; font-family: 'Georgia', serif; }
.reader-quote .signature { text-align: right; margin-top: 1rem; font-style: italic; color: #8b5a2b; font-size: 0.95rem; font-family: 'Georgia', serif; }
.list-card { background: #fffaf2; border: 1px solid #e6d2a3; padding: 12px; border-radius: 8px; margin-bottom: 12px; }
.list-title { font-weight: 700; color: #2c1810; font-size: 1.05rem; margin-bottom: 8px; }
.list-item { margin: 6px 0; line-height: 1.4; }
.metric-card { text-align: center; padding: 1.5rem; background: #fffbf0; border-radius: 8px; border: 2px solid #d4af37; }
.section-title { color: #2c1810; font-size: 1.35rem; font-weight: bold; margin-top: 1.2rem; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem; }
.highlight-box { background: linear-gradient(to right, #fffbf0, #fff9e6); border-left: 4px solid #d4af37; padding: 1rem; margin: 1.2rem 0; color: #5c4033; border-radius: 6px; }
.sidebar-section { color: #ffffff; font-family: Cinzel, serif; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-section"><strong>Navigation</strong></div>', unsafe_allow_html=True)
    st.markdown("- `app.py`")
    st.markdown("- `about.py` (this page)")
    st.markdown("- `data/`")
    st.markdown("---")

st.markdown("""
<div style="display:block; width:fit-content; margin:0 auto; padding:0.5rem 2rem; background:linear-gradient(to right,#5c070d,#840a13); border-radius:8px; text-align:center;">
    <h1 style="font-family:Cinzel, serif; color:#ffffff; margin:0;">A PROJEKTRŐL</h1>
</div>
<div style="width:100px; height:4px; background:linear-gradient(to right,#d4af37,#f0d98d,#d4af37); margin:1rem auto 1.5rem auto; border-radius:2px;"></div>
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

    fasting_set = {normalize_label(t) for t in FASTING_RECIPE_TITLES}
    fasting_flags = []
    for idx, row in historical.iterrows():
        title = normalize_label(str(row.get('title','')))
        combined_text = title
        for c in text_fields:
            combined_text = combined_text + ' ' + normalize_label(str(row.get(c,'')))
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
                elif isinstance(clf_res, (int,float)) and clf_res >= 0.5:
                    is_fasting = True
            except Exception:
                pass
        fasting_flags.append(is_fasting)
    fast_count = sum(1 for f in fasting_flags if f)
    fast_pct = round(fast_count / len(historical) * 100, 1) if len(historical) > 0 else 0.0

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
        st.markdown('<div style="margin-top:8px; color:#4a3728;">A PageRank nemcsak a kapcsolatok számát nézi, hanem azok minőségét: ha egy alapanyag kapcsolatban áll más fontos alapanyagokkal, akkor magasabb a PageRank-je — ez a „befolyásosság” mutatója a hálózatban.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with bet_col:
        st.markdown('<div class="list-card"><div class="list-title">Top 10 — Betweenness (hidak)</div>', unsafe_allow_html=True)
        for i, (n, v) in enumerate(top_bet, start=1):
            st.markdown(f'<div class="list-item">{i}. <strong>{readable(n)}</strong> — {v:.6f}</div>', unsafe_allow_html=True)
        st.markdown('<div style="margin-top:8px; color:#4a3728;">A Betweenness azt jelenti, hogy egy alapanyag milyen gyakran van a legrövidebb utak „közepén” a hálózatban — ezek a csomópontok gyakran kötik össze a különböző ízvilágokat, vagy átjárót képeznek két csoport között.</div>', unsafe_allow_html=True)
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
    st.markdown("### 4) Mennyire közelíti meg az AI a történeti receptek stílusát és szerkezetét? (Valódi modell-lekéréssel — GPT-5.1)")

    def build_generation_pool(all_nodes_df, edges_df, historical_df):
        nodes = []
        for _, r in all_nodes_df.iterrows():
            nodes.append({
                "Label": strip_icon_ligatures(r.get("Label","")),
                "norm": normalize_label(r.get("Label","")),
                "node_type": r.get("node_type","Egyéb"),
                "Degree": int(r.get("Degree", 0) or 0) if "Degree" in r else 0
            })
        edge_map = {}
        for _, e in edges_df.iterrows():
            s = e.get("norm_source","")
            t = e.get("norm_target","")
            if s and t:
                edge_map.setdefault(s, set()).add(t)
                edge_map.setdefault(t, set()).add(s)
        hist_texts = []
        for _, hr in historical_df.iterrows():
            txt = ""
            for c in ('original_text','text','instructions','description','ingredients'):
                if c in historical_df.columns and isinstance(hr.get(c,''), str):
                    txt += " " + hr.get(c,'')
            txt = txt.strip() or hr.get("title","")
            hist_texts.append(strip_icon_ligatures(txt))
        return nodes, edge_map, hist_texts

    nodes_pool, edge_map, historical_texts = build_generation_pool(tripartit, edges, historical)

    def prepare_prompt_samples(hist_list, max_snippets=8, max_len=220):
        samples = [strip_icon_ligatures(h) for h in hist_list if h]
        rnd = random.Random(0)
        picked = rnd.sample(samples, min(len(samples), max_snippets)) if samples else []
        truncated = [s.replace('\n', ' ')[:max_len].strip() for s in picked]
        return truncated

    def call_openai_batch_generate(n_recipes, seed, sample_nodes, sample_snippets):
        if openai is None:
            return None, "OpenAI SDK not installed"
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        if not api_key:
            return None, "OPENAI_API_KEY not set in environment"
        openai.api_key = api_key
        prompt_snippets = "\n".join(f"- {s}" for s in sample_snippets)
        sample_nodes_text = ", ".join(sample_nodes[:20])
        system_msg = "You are a concise historical recipe generator that produces short archaic-style Hungarian recipe descriptions (not exact quotes). Output must be valid JSON: an array of objects with keys 'title' and 'archaic_recipe' and 'word_count'. Do not include any commentary outside the JSON."
        user_instructions = textwrap.dedent(f"""
        Generate {n_recipes} distinct recipes based on the provided node pool and example snippets.
        Constraints:
        1) Each recipe: 'title' (short), 'archaic_recipe' (Hungarian archaic style, 80-120 words), 'word_count' integer.
        2) Do not copy verbatim from the examples. Be inspired by style and vocabulary only.
        3) Prefer mixing node names/ingredients from the node list. Use simple, short archaic connectors.
        4) Return EXACTLY one JSON array, nothing else.
        Provided example snippets (do not copy verbatim):
        {prompt_snippets}
        Node examples (use for inspiration): {sample_nodes_text}
        Use deterministic seed: {seed}
        """)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_instructions}
        ]
        try:
            resp = openai.ChatCompletion.create(model="gpt-5.1", messages=messages, max_tokens=25000, temperature=0.8)
            text = resp['choices'][0]['message']['content']
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end != -1:
                json_text = text[start:end]
            else:
                json_text = text
            parsed = json.loads(json_text)
            return parsed, None
        except Exception as e:
            return None, str(e)

    n_generate = st.number_input("Generálandó receptek száma (valódi modell-lekérés):", min_value=10, max_value=200, value=100, step=10)
    seed_input = st.number_input("Random seed (deterministic):", min_value=0, max_value=999999, value=42, step=1)
    run_openai = st.button("Lekér GPT-5.1 és elemez (egyszer lefut, majd cache-el)")

    @st.cache_data(show_spinner=False)
    def run_openai_generation_cached(n_generate, seed_input, nodes_pool_serialized, edge_map_serialized, historical_texts_serialized):
        sample_nodes = [n.get("Label") for n in nodes_pool_serialized if n.get("Label")]
        sample_snippets = prepare_prompt_samples(historical_texts_serialized, max_snippets=8, max_len=220)
        generated, err = call_openai_batch_generate(int(n_generate), int(seed_input), sample_nodes, sample_snippets)
        return generated, err

    if run_openai:
        st.session_state.pop("ai_batch_generated", None)
        with st.spinner("Lekérés a GPT-5.1 modellhez... kérlek várj (egyszeri hívás, hosszabb ideig tarthat)"):
            nodes_serial = nodes_pool
            edge_map_serial = edge_map
            hist_serial = historical_texts
            gen, error = run_openai_generation_cached(n_generate, seed_input, nodes_serial, edge_map_serial, hist_serial)
            if gen is None:
                st.error(f"OpenAI hívás sikertelen: {error}. Helyi szimulációt futtatok helyette.")
                batch = []
                rnd = random.Random(seed_input)
                node_labels = [n["Label"] for n in nodes_pool if n.get("Label")]
                for i in range(int(n_generate)):
                    sel = rnd.choice(node_labels) if node_labels else f"alapanyag_{i}"
                    connected_norms = list(edge_map.get(normalize_label(sel), []))
                    connected = []
                    for cn in connected_norms:
                        node_record = next((x for x in nodes_pool if normalize_label(x.get("Label","")) == cn), None)
                        if node_record:
                            connected.append({"name": node_record.get("Label")})
                    title = sel
                    body = f"Vegyünk {sel}ot. " + " ".join([f"majd {rnd.choice(['hagymával','ecettel','mézzel','borssal'])} {rnd.choice(['pirítva','főzve','párolva'])}." for _ in range(8)])
                    wc = len(body.split())
                    sim = max(sequence_similarity(normalize_label(body), normalize_label(h)) for h in historical_texts) if historical_texts else 0.0
                    batch.append({"title": title, "archaic_recipe": body, "word_count": wc, "max_similarity": sim, "novelty": 1.0 - sim})
                st.session_state["ai_batch_generated"] = batch
            else:
                parsed = gen
                batch = []
                for item in parsed:
                    txt = item.get("archaic_recipe","") if isinstance(item, dict) else ""
                    wc = int(item.get("word_count", len(txt.split()))) if isinstance(item, dict) else len(txt.split())
                    sim = max(sequence_similarity(normalize_label(txt), normalize_label(h)) for h in historical_texts) if historical_texts else 0.0
                    batch.append({"title": item.get("title","(nincs cím)"), "archaic_recipe": txt, "word_count": wc, "max_similarity": sim, "novelty": 1.0 - sim})
                st.session_state["ai_batch_generated"] = batch
            st.success("Generálás és feldolgozás befejezve.")

    if "ai_batch_generated" in st.session_state:
        batch = st.session_state["ai_batch_generated"]
        sims = [b.get("max_similarity", 0.0) for b in batch]
        novelties = [b.get("novelty", 0.0) for b in batch]
        mean_max_sim = sum(sims)/len(sims) if sims else 0.0
        mean_novelty = sum(novelties)/len(novelties) if novelties else 0.0
        exceed60 = sum(1 for s in sims if s > 0.6)
        st.markdown(f"- Összes generált recept: **{len(batch)}**")
        st.markdown(f"- Átlag legnagyobb similarity a korpusszal: **{mean_max_sim:.3f}**")
        st.markdown(f"- Átlag novelty (1 - max_similarity): **{mean_novelty:.3f}**")
        st.markdown(f"- Hány recept haladja meg a similarity > 0.6 küszöböt: **{exceed60}** ({(exceed60/len(batch))*100:.1f}%)")
        top_similar = sorted(batch, key=lambda x: x.get("max_similarity",0), reverse=True)[:6]
        st.markdown("**Leginkább a korpusszal megegyező/hasonsző generált példák (top 6)**")
        for i, t in enumerate(top_similar, start=1):
            st.markdown(f"{i}. **{strip_icon_ligatures(t.get('title','(nincs cím)'))}** — max_similarity: **{t.get('max_similarity',0):.3f}**, novelty: **{t.get('novelty',0):.3f}**, szavak: **{t.get('word_count',0)}**")
            excerpt = t.get("archaic_recipe","")[:300]
            st.markdown(f"> {excerpt}...")
        st.markdown("---")
        st.markdown("**Módszertan röviden (ami történik a generálásnál):**")
        st.markdown(textwrap.dedent("""
        - A GPT-5.1-nek egyszer küldünk egy promptot, amely tartalmaz rövid történeti példákat és néhány node-címet; a modell JSON tömböt ad vissza `title` és `archaic_recipe` mezőkkel.
        - A visszaadott receptekhez kiszámoljuk a legnagyobb similarity értéket (SequenceMatcher) a történeti korpusz bármely teljes receptjével: ez a `max_similarity`.
        - Novelty = 1 - max_similarity. Ha `max_similarity` > 0.6, akkor a generált szöveg erősen hasonlít egy vagy több forráspéldához — ilyenkor javasolt újragenerálás vagy erősebb grounding.
        - A lekérést egyszer végezzük el; az eredmény a session-ben cache-elve és elmentve marad, így további elemzések tokenmentesek.
        """), unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Példa: 3 véletlenszerű generált recept (teljes szöveg)**")
        sample_three = random.Random(1234).sample(batch, min(3, len(batch)))
        for s in sample_three:
            st.markdown(f"#### {strip_icon_ligatures(s.get('title','(nincs cím)'))}")
            st.markdown(f"{s.get('archaic_recipe','')}")
            st.markdown(f"- Novelty: **{s.get('novelty',0):.3f}**, Max similarity: **{s.get('max_similarity',0):.3f}**, Szavak: **{s.get('word_count',0)}**")
            st.markdown("---")
    else:
        st.info("A GPT-5.1 lekéréshez nyomd meg a 'Lekér GPT-5.1 és elemez' gombot. Az eredmény egyszer generálódik és cache-elve lesz a session-ben.")
