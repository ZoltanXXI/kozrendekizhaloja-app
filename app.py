import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import plotly.graph_objects as go
import json
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import html as _html
import textwrap
import random
import unicodedata
import re
import base64
import difflib

st.set_page_config(
    page_title="Közrendek Ízhálója",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(""" 
<style>
div[data-baseweb="select"] > div {
    background-color: #7a0f0f !important;
    color: #f5f5f5 !important;
    border-radius: 10px;
    border: 1px solid #cfa34a;
}
div[data-baseweb="popover"] { background-color: #2a0c0c !important; border-radius: 12px; border: 1px solid #cfa34a; }
div[data-baseweb="menu"] { background-color: #2a0c0c !important; }
div[data-baseweb="option"] { color: #f0e6d2 !important; background-color: transparent !important; }
div[data-baseweb="option"]:hover { background-color: #7a0f0f !important; color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(""" 
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400&display=swap');
    .main { background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 50%, #1a1a1a 100%) !important; background-image: url("https://www.transparenttextures.com/patterns/dark-leather.png") !important; padding: 0 !important; }
    .block-container { padding: 2rem 3rem !important; max-width: 1400px !important; background: rgba(0, 0, 0, 0.3); }
    h1, h2, h3 { font-family: 'Cinzel', serif !important; color: white !important; font-weight: 700 !important; }
    h1 { font-size: 2.5rem !important; text-align: center !important; margin-bottom: 1rem !important; }
    .block-container p, .block-container div, .block-container span, .block-container li { font-family: 'Crimson Text', serif !important; color: white !important; font-size: 1.05rem; }
    .stButton > button { background: linear-gradient(135deg, #800000 0%, #5c1a1a 100%); color: white !important; border: none; border-radius: 8px; font-family: 'Cinzel', serif !important; font-size: 1rem !important; font-weight: 600 !important; padding: 0.6rem 1rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5); transition: all 0.3s ease; width: 100%; text-align: left; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(204, 170, 119, 0.3); background: linear-gradient(135deg, #a52a2a 0%, #722828 100%); }
    div[data-testid="stMetric"] { background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%); padding: 1.5rem; border-radius: 12px; border: 2px solid #ccaa77; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.5); }
    [data-testid="stMetricValue"] { font-family: 'Cinzel', serif !important; color: white !important; font-size: 2.5rem !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { font-family: 'Crimson Text', serif !important; color: white !important; font-size: 1rem !important; text-transform: uppercase; letter-spacing: 1px; }
    .stTextInput input, .stTextInput div[role="textbox"] input { background-color: #840A13 !important; color: #f5efe6 !important; }
    .stTextInput input::placeholder { color: #f5efe6 !important; opacity: 0.9 !important; font-style: italic; }
    div[data-testid="stSelectbox"] div[role="listbox"], div[data-baseweb="menu"] [role="listbox"], div[role="listbox"], div[role="presentation"] > div[role="listbox"], div[role="menu"], div[data-testid="stSelectbox"] .baseweb-popover-content, .rc-virtual-list, .baseweb-popover-content { background-color: #4a0d0d !important; color: #f5efe6 !important; border-radius: 10px !important; box-shadow: 0 10px 30px rgba(0,0,0,0.6) !important; max-height: 360px !important; overflow-y: auto !important; min-width: 260px !important; width: auto !important; padding: 0.2rem !important; z-index: 100001 !important; border: 1px solid rgba(255,36,0,0.12) !important; }
    div[data-testid="stSelectbox"] div[role="listbox"] *, div[data-testid="stSelectbox"] .baseweb-popover-content * { background-color: transparent !important; color: inherit !important; }
    div[data-testid="stSelectbox"] div[role="listbox"] [role="option"], .baseweb-popover-content [role="option"], div[role="option"] { background-color: transparent !important; color: #f5efe6 !important; padding: 0.6rem 0.9rem !important; }
    div[data-testid="stSelectbox"] div[role="listbox"]::-webkit-scrollbar, .baseweb-popover-content::-webkit-scrollbar { width: 10px !important; height: 10px !important; }
    div[data-testid="stSelectbox"] div[role="listbox"]::-webkit-scrollbar-thumb, .baseweb-popover-content::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.35) !important; border-radius: 8px !important; border: 2px solid rgba(255,255,255,0.02) !important; }
    div[data-testid="stSelectbox"] div[role="listbox"], div[data-baseweb="menu"] [role="listbox"], div[role="listbox"], div[role="presentation"] > div[role="listbox"], div[role="menu"] { background-color: #840A13 !important; color: #f5efe6 !important; border-radius: 10px !important; box-shadow: 0 10px 30px rgba(0,0,0,0.6) !important; max-height: 360px !important; overflow-y: auto !important; min-width: 260px !important; width: auto !important; padding: 0.2rem !important; z-index: 100001 !important; }
    div[role="option"] { background-color: transparent !important; color: #f5efe6 !important; padding: 0.6rem 0.9rem !important; font-family: 'Crimson Text', serif !important; font-size: 1rem !important; cursor: pointer !important; border-radius: 6px !important; margin: 0.12rem 0 !important; }
    div[role="option"]:hover, div[role="option"][data-highlighted="true"], div[role="option"][aria-selected="true"] { background-color: #FF2400 !important; color: #ffffff !important; font-weight: 600 !important; }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div { background-color: #840A13 !important; border: 2px solid #FF2400 !important; border-radius: 8px !important; color: #f5efe6 !important; z-index: 99999 !important; }
    div[data-testid="stSelectbox"] input, div[data-testid="stSelectbox"] [role="combobox"], div[data-testid="stSelectbox"] [role="button"] { color: #f5efe6 !important; background-color: #840A13 !important; }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > span { color: #f5efe6 !important; }
    @media (max-width: 800px) {
        div[data-testid="stSelectbox"] div[role="listbox"], div[role="listbox"] { left: 1rem !important; right: 1rem !important; width: auto !important; min-width: unset !important; }
    }
    [data-testid="stSidebar"] > div:first-child { background-color: #5c1a1a !important; font-family: 'Cinzel', serif !important; color: #ffffff !important; }
    [data-testid="stSidebar"] button, [data-testid="stSidebar"] .st-expander, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div[data-testid$="-label"] { font-family: 'Cinzel', serif !important; color: #ffffff !important; }
    [data-testid="stSidebar"] span[data-testid="stIconMaterial"], .span[data-testid="stIconMaterial"] { display: none !important; }
    [data-testid="stKeyboardShortcutButton"], button[aria-label="Show keyboard shortcuts"], button[aria-label="Show keyboard navigation"], [data-testid^="stTooltip"] { display: none !important; }
    .carousell-card { background: linear-gradient(135deg, #1a1a1a, #1f1f1f); border-radius: 18px; border: 1px solid #d4af37; box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4); padding: 24px; color: #f9f3e8; font-family: 'Playfair Display', serif; }
    .card-title { font-size: 26px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.08em; }
    .card-value { font-size: 42px; margin: 0; font-weight: 600; }
    .card-desc { font-size: 16px; line-height: 1.6; margin-top: 12px; color: #e7dac5; }
</style>
""", unsafe_allow_html=True)

load_dotenv()
api_key = None
try:
    api_key = st.secrets.get("OPENAI_API_KEY")
except Exception:
    api_key = None
if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ Hiányzik az OPENAI_API_KEY! Add it to `.streamlit/secrets.toml` or set the OPENAI_API_KEY environment variable.")
    st.stop()
client = OpenAI(api_key=api_key)
random.seed(42)

def strip_icon_ligatures(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = _html.unescape(s)
    s = re.sub(r"<[^>]+>", "", s)
    s = unicodedata.normalize('NFKC', s)
    filtered_chars = []
    for ch in s:
        cat = unicodedata.category(ch)
        o = ord(ch)
        if cat.startswith('C'):
            continue
        if 0xE000 <= o <= 0xF8FF:
            continue
        if 0xF0000 <= o <= 0xFFFFD:
            continue
        filtered_chars.append(ch)
    s = ''.join(filtered_chars)
    s = re.sub(r'[_\-\s]+', ' ', s).strip()
    icon_keywords = {'keyboard','keyb','arrow','check','radio','menu','close','settings','search','favorite','share','more','material','icon','icons','vert','horiz'}
    def token_clean(t: str) -> str:
        t_norm = re.sub(r'[^a-z0-9]+', '', t.lower())
        return t_norm
    tokens = [t for t in s.split() if not any(kw in token_clean(t) for kw in icon_keywords)]
    s = ' '.join(tokens)
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s

@st.cache_data
def load_data():
    script_dir = os.path.dirname(__file__)
    def _resolve(rel_path):
        candidates = []
        bases = [script_dir, os.getcwd(), os.path.abspath(os.path.join(script_dir, '..'))]
        for b in bases:
            candidates.append(os.path.normpath(os.path.join(b, rel_path)))
        candidates.append(os.path.normpath(rel_path))
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates

    tripartit_path = _resolve(os.path.join('data', 'Recept_halo__molekula_tripartit.csv'))
    edges_path = _resolve(os.path.join('data', 'recept_halo_edges.csv'))
    historical_path = _resolve(os.path.join('data', 'HistoricalRecipe_export.csv'))

    def _ensure_found(res, logical_name):
        if isinstance(res, list):
            st.error(f"❌ Hiányzik a fájl: {logical_name}. Próbált elérési utak:")
            for p in res:
                st.write(f"- {p}")
            st.stop()
        return res

    tripartit_path = _ensure_found(tripartit_path, 'data/Recept_halo__molekula_tripartit.csv')
    edges_path = _ensure_found(edges_path, 'data/recept_halo_edges.csv')
    historical_path = _ensure_found(historical_path, 'data/HistoricalRecipe_export.csv')

    def safe_read_csv(path, name, default_sep=';'):
        try:
            return pd.read_csv(path, delimiter=default_sep, encoding='utf-8', on_bad_lines='skip')
        except Exception:
            try:
                return pd.read_csv(path, sep=None, engine='python', encoding='utf-8', on_bad_lines='skip')
            except Exception:
                try:
                    return pd.read_csv(path, delimiter=default_sep, encoding='latin1', on_bad_lines='skip')
                except Exception:
                    try:
                        return pd.read_csv(path, sep=None, engine='python', encoding='latin1', on_bad_lines='skip')
                    except Exception:
                        try:
                            with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                                preview = fh.read(5000)
                        except Exception:
                            preview = f"(Could not read file contents for preview: {path})"
                        st.error(f"❌ Hiba a CSV beolvasásakor: {name}")
                        st.markdown("**Próbált beolvasási módszerek:** UTF-8 with ';', infer sep (python engine), Latin-1 variants.")
                        st.markdown("**Fájl előnézet (első 5000 karakter):**")
                        st.code(preview)
                        st.stop()

    tripartit_df = safe_read_csv(tripartit_path, 'data/Recept_halo__molekula_tripartit.csv', default_sep=';')
    edges_df = safe_read_csv(edges_path, 'data/recept_halo_edges.csv', default_sep=',')
    historical_df = safe_read_csv(historical_path, 'data/HistoricalRecipe_export.csv', default_sep=',')

    for col in ['title', 'original_text', 'ingredients']:
        if col in historical_df.columns:
            historical_df[col] = historical_df[col].apply(lambda x: strip_icon_ligatures(x) if isinstance(x, str) else x)

    perfect_ings = []
    try:
        perfect_candidate = _resolve(os.path.join('data', 'recept_alapanyagok_TÖKÉLETES.json'))
        if not isinstance(perfect_candidate, list) and os.path.exists(perfect_candidate):
            with open(perfect_candidate, encoding='utf-8') as f:
                raw = json.load(f)
                ingredients = set()
                if isinstance(raw, dict):
                    for v in raw.values():
                        if isinstance(v, list):
                            for item in v:
                                if isinstance(item, str):
                                    ingredients.add(item)
                        elif isinstance(v, str):
                            ingredients.add(v)
                elif isinstance(raw, list):
                    for entry in raw:
                        if isinstance(entry, str):
                            ingredients.add(entry)
                        elif isinstance(entry, dict):
                            for v in entry.values():
                                if isinstance(v, list):
                                    for item in v:
                                        if isinstance(item, str):
                                            ingredients.add(item)
                                elif isinstance(v, str):
                                    ingredients.add(v)
                perfect_ings = sorted(ingredients)
    except Exception:
        perfect_ings = []

    return tripartit_df, edges_df, historical_df, perfect_ings

tripartit_df, edges_df, historical_df, perfect_ings = load_data()

SYNONYM_MAP = {
    "rózsabors": ["rózsabors", "pink pepper", "schinus", "schinus molle", "rózsás bors"],
    "avokádó": ["avokádó", "avokado", "avokádós", "avocado"],
    "avokádós": ["avokádó", "avokado", "avokádós", "avocado"],
    "kaja": ["kaja", "étel", "fogás", "meal", "dish", "food"],
    "mandula": ["mandula", "almond"]
}
GENERIC_TOKENS = {"kaja", "étel", "fogás", "recept", "food", "dish", "meal"}

TOKEN_ROLE = {
  "ingredient": [],
  "flavour_descriptor": [],
  "preparation_style": [],
  "generic_food": [],
  "metaphorical": [],
}

ANACHRONISTIC_INGREDIENTS = {
  "avokádó", "paradicsom", "burgonya", "csili", "vanília", "kakaó", "paprika", "ananász"
}

HISTORICAL_ANALOGY_MAP = {
    "avokádó": ["mandula", "olaj-spék", "tört hüvelyes"],
    "avokádós": ["mandula", "olaj-spék", "tört hüvelyes"],
    "rózsabors": ["tiszta borssal", "rózsaszirom infúzió (illatosító)", "borsos szilva"],
    "pink pepper": ["tiszta borssal", "rózsaszirom infúzió (illatosító)"]
}

HISTORICAL_DISH_STRUCTURE_MAP = {
    "pite": {
        "interpreted_as": "töltött vagy rétegezett tésztás étel",
        "historical_equivalents": [
            "túrós tészta",
            "almás tészta",
            "mákos tészta",
            "káposztás tészta"
        ],
        "confidence": 0.75,
        "source": "MEK történeti receptgyűjtemény"
    }
}

def detect_label_col(df):
    candidates = [c for c in df.columns if c.lower() in ('label','name','title','node','node_name')]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if 'label' in c.lower() or 'name' in c.lower():
            return c
    return df.columns[0] if len(df.columns) else None

label_col = detect_label_col(tripartit_df)

def detect_id_col(df):
    candidates = [c for c in df.columns if c.lower() in ('id','node_id','label_id','idx','index')]
    if candidates:
        return candidates[0]
    for c in df.columns:
        if 'id'==c.lower() or c.lower().endswith('_id'):
            return c
    return None

id_col = detect_id_col(tripartit_df)

def detect_type_col(df):
    candidates = [c for c in df.columns if 'type' in c.lower() or 'category' in c.lower() or 'node_type'==c.lower() or 'class' in c.lower()]
    return candidates[0] if candidates else None

type_col = detect_type_col(tripartit_df)

type_mapping = {
    "dish": "Recept",
    "recipe": "Recept",
    "alapanyag": "Alapanyag",
    "ingredient": "Alapanyag",
    "molecule": "Molekula",
    "molekula": "Molekula",
    "ing": "Alapanyag",
    "food": "Alapanyag"
}

if label_col is None:
    tripartit_df['Label'] = tripartit_df.apply(lambda r: f"node_{r.name}", axis=1)
    label_col = 'Label'

if id_col is None:
    tripartit_df['node_id'] = tripartit_df.index.astype(str).apply(lambda x: f"node_{x}")
    id_col = 'node_id'
else:
    tripartit_df['node_id'] = tripartit_df[id_col].astype(str)

if type_col:
    tripartit_df['_type_raw'] = tripartit_df[type_col].astype(str).fillna("")
    tripartit_df['node_type'] = tripartit_df['_type_raw'].apply(lambda v: type_mapping.get(v.strip().lower(), None) or next((type_mapping.get(tok) for tok in re.split(r'[\s,;/]+', v.strip().lower()) if tok in type_mapping), None) or "Egyéb")
else:
    tripartit_df['node_type'] = "Egyéb"

tripartit_df['Label'] = tripartit_df[label_col].astype(str).apply(strip_icon_ligatures)

def normalize_label(s):
    if not isinstance(s, str):
        return ""
    cleaned = strip_icon_ligatures(s)
    cleaned = cleaned.lower()
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

tripartit_df['norm_label'] = tripartit_df['Label'].apply(normalize_label)
tripartit_df['norm_id'] = tripartit_df['node_id'].astype(str).apply(lambda x: normalize_label(str(x)))

node_norm_map = {}
node_id_map = {}
for _, row in tripartit_df.iterrows():
    norm = row['norm_label']
    nid = str(row['node_id'])
    rec = row.to_dict()
    node_norm_map[norm] = rec
    node_id_map[nid] = rec

def find_edge_candidate_cols(edges_df):
    cols = list(edges_df.columns)
    src_candidates = [c for c in cols if 'source' in c.lower() or 'from' in c.lower() or c.lower().startswith('src')]
    tgt_candidates = [c for c in cols if 'target' in c.lower() or 'to' in c.lower() or c.lower().startswith('dst') or c.lower().startswith('tgt')]
    if not src_candidates:
        src_candidates = [c for c in cols if 'label' in c.lower() or 'name' in c.lower()][:2]
    if not tgt_candidates:
        tgt_candidates = [c for c in cols if 'label' in c.lower() or 'name' in c.lower()][:2]
    if not src_candidates:
        src_candidates = cols[:1]
    if not tgt_candidates:
        tgt_candidates = cols[-1:]
    return src_candidates, tgt_candidates

src_candidates, tgt_candidates = find_edge_candidate_cols(edges_df)

def resolve_endpoint_value(val):
    if val is None:
        return ""
    sval = str(val).strip()
    if not sval:
        return ""
    s_norm = normalize_label(sval)
    if s_norm in node_norm_map:
        return s_norm
    if sval in node_id_map:
        return normalize_label(node_id_map[sval].get('Label',''))
    if s_norm in node_id_map:
        return s_norm
    if sval in node_id_map:
        return normalize_label(node_id_map[sval].get('Label',''))
    return s_norm

def compute_edge_norms(edges_df):
    norm_sources = []
    norm_targets = []
    for _, row in edges_df.iterrows():
        src_val = None
        for c in src_candidates:
            if c in row and str(row[c]).strip():
                src_val = row[c]
                break
        tgt_val = None
        for c in tgt_candidates:
            if c in row and str(row[c]).strip():
                tgt_val = row[c]
                break
        src_norm = resolve_endpoint_value(src_val)
        tgt_norm = resolve_endpoint_value(tgt_val)
        norm_sources.append(src_norm)
        norm_targets.append(tgt_norm)
    edges_df = edges_df.copy()
    edges_df['norm_source'] = norm_sources
    edges_df['norm_target'] = norm_targets
    return edges_df

edges_df = compute_edge_norms(edges_df)

all_nodes = tripartit_df.to_dict("records")
all_edges = edges_df.to_dict("records")
historical_recipes = historical_df.to_dict("records")

def load_full_recipe_corpus_from_hist(historical_recipes):
    recipes_full = []
    for recipe in historical_recipes:
        full_text = recipe.get('original_text', '') or ''
        title = strip_icon_ligatures(recipe.get('title', 'Névtelen'))
        ingredients = recipe.get('ingredients', '') or ''
        context = f"""
RECEPT CÍM: {title}

ALAPANYAGOK: {ingredients}

TELJES SZÖVEG:
{full_text}
        """.strip()
        recipes_full.append({
            'title': title,
            'ingredients': ingredients,
            'full_text': full_text,
            'context': context,
            'word_count': len(full_text.split())
        })
    return recipes_full

full_recipe_corpus = load_full_recipe_corpus_from_hist(historical_recipes)

FASTING_RECIPE_TITLES = {
    "Káposzta ikrával", "Alma-lév", "Mondola-perec", "Koldus-lév", "Ég-lév",
    "Zsákvászonnal", "Gutta-lév", "Szíjalt rák", "Lengyel cibre", "Körtvély főve",
    "Saláta", "Torzsa-saláta", "Ugorka-saláta", "Miskuláncia-saláta", "Mondola-lév",
    "Bot-lév", "Kendermag-cibre", "Ikrát főzni", "Nyers káposzta-saláta", "Borsóleves",
    "Párolt rák", "Korpa-cibre", "Borsót főzni", "Ugorkát télre sózni", "Fenyőgombát főzni",
    "Kínzott kása", "Lencseleves", "Hal rizskásával", "Olaj-spék", "Cicer",
    "Sült hal", "Lémonyával", "Törött lével hal", "Csukát csuka-lével", "Olajos domika",
    "Kozák-lével", "Zöld lével", "Borsos szilva", "Ecetes cibre", "Hal fekete lével",
    "Zuppon-lév", "Tiszta borssal", "Bors-porral", "Vizát viza-lével", "Szömörcsök-gomba",
    "Borított lév", "Kása olajjal", "Lencse olajjal", "Borsó laskával", "Káposztás béles",
    "Hagyma rántva", "Káposzta-lév cibre", "Lönye", "Lása", "Sós víz",
    "Seres kenyér", "Olajos lév", "Viza ikra", "Új káposzta"
}

def is_fasting_recipe(recipe):
    title = (recipe.get("title") or "").strip()
    return title in FASTING_RECIPE_TITLES

def create_network_graph(center_node, connected_nodes):
    if not center_node or not connected_nodes:
        return None
    G = nx.Graph()
    G.add_node(center_node, node_type='center')
    for n in connected_nodes:
        G.add_node(n["name"], degree=n.get("degree", 0), node_type=n.get("type", "unknown"))
        G.add_edge(center_node, n["name"], weight=n.get("degree", 1))
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode='lines',
                       line=dict(width=1.5, color='rgba(255,255,255,0.95)'), hoverinfo='none', showlegend=False)
        )
    node_colors = {'center': '#ccaa77', 'Alapanyag': '#8b5a2b', 'Molekula': '#808080', 'Recept': '#800000', 'unknown': '#999999'}
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    max_degree = max([n.get("degree", 1) for n in connected_nodes], default=1)
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if node == center_node:
            node_text.append(f"<b style='font-size: 14px'>{node}</b><br><i>(központi)</i>")
            node_size.append(44)
            node_color.append(node_colors['center'])
        else:
            degree = next((n["degree"] for n in connected_nodes if n["name"] == node), 1)
            node_type = next((n.get("type", "unknown") for n in connected_nodes if n["name"] == node), "unknown")
            node_text.append(f"<b>{node}</b><br>Degree: {degree}<br>Típus: {node_type}")
            node_size.append(16 + (degree / max_degree) * 34)
            node_color.append(node_colors.get(node_type, node_colors['unknown']))
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', hovertemplate='%{text}<extra></extra>',
        text=[n.split('<br>')[0].replace('<b style=\'font-size: 14px\'>', '').replace('</b>', '').replace('<b>', '') for n in node_text],
        textposition="top center", textfont=dict(size=10, family="Crimson Text", color='white'),
        marker=dict(size=node_size, color=node_color, line=dict(width=2, color='white')), customdata=node_text, showlegend=False
    )
    fig = go.Figure(data=edge_trace + [node_trace])
    fig.update_layout(showlegend=False, hovermode='closest', margin=dict(b=0, l=0, r=0, t=0),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', template='plotly_dark', height=800)
    return fig

def build_gpt_context(nodes, recipes, perfect_ings=None, user_query=None, max_nodes=120, max_recipes=40):
    grouped = {}
    for n in nodes:
        grouped.setdefault(n.get("node_type", "Egyéb"), []).append(n)
    sampled_nodes = []
    if grouped:
        for group in grouped.values():
            sampled_nodes.extend(random.sample(group, min(len(group), max_nodes // max(1, len(grouped)))))
    else:
        sampled_nodes = nodes[:max_nodes]
    normalized_query = None
    if user_query:
        def _normalize(s):
            if not isinstance(s, str):
                return ""
            s = s.lower()
            s = unicodedata.normalize('NFKD', s)
            s = ''.join(ch for ch in s if not unicodedata.combining(ch))
            s = re.sub(r"[^a-z0-9]+", ' ', s)
            s = ' '.join(s.split())
            return s
        normalized_query = _normalize(user_query)
    nodes_ctx = []
    for node in sampled_nodes:
        entry = dict(node)
        entry_name = strip_icon_ligatures(entry.get("Label")
                                          or entry.get("label")
                                          or entry.get("node_name")
                                          or entry.get("node_id")
                                          or entry.get("name")
                                          or "")
        entry["name"] = entry_name
        nodes_ctx.append(entry)

    simplified_nodes = [
        {
            "name": n["name"],
            "type": n.get("node_type") or n.get("type") or n.get("Type") or "Egyéb",
            "degree": int(n.get("Degree", n.get("degree", 0) or 0))
        }
        for n in nodes_ctx if n.get("name")
    ]

    if normalized_query and user_query:
        q_norm = normalized_query
        q_tokens = [t for t in q_norm.split() if len(t) > 1]
        if q_tokens:
            def _normalize(s):
                if not isinstance(s, str):
                    return ""
                s = s.lower()
                s = unicodedata.normalize('NFKD', s)
                s = ''.join(ch for ch in s if not unicodedata.combining(ch))
                s = re.sub(r"[^a-z0-9]+", ' ', s)
                s = ' '.join(s.split())
                return s
            matched = [n for n in nodes if any(tok in _normalize(n.get("Label", "")) for tok in q_tokens)]
            matched_perfect = []
            if perfect_ings:
                for p in (perfect_ings if isinstance(perfect_ings, list) else [perfect_ings]):
                    label = None
                    if isinstance(p, str):
                        label = p
                    elif isinstance(p, dict):
                        for key in ("label", "name", "ingredient", "term", "alapanyag"):
                            if key in p and isinstance(p[key], str):
                                label = p[key]
                                break
                        if not label:
                            for v in p.values():
                                if isinstance(v, str):
                                    label = v
                                    break
                    if label and any(tok in _normalize(label) for tok in q_tokens):
                        matched_perfect.append({"Label": label, "node_type": "Alapanyag", "Degree": 0})
            seen_labels = {_normalize(n.get("Label", "")) for n in sampled_nodes}
            for m in matched + matched_perfect:
                m_label = _normalize(m.get("Label", ""))
                if m_label and m_label not in seen_labels:
                    sampled_nodes.insert(0, m)
                    seen_labels.add(m_label)
    related_nodes = [n["name"] for n in nodes_ctx if n.get("name")]
    related_analogies = []
    for node in nodes_ctx:
        analogies = HISTORICAL_ANALOGY_MAP.get(node.get("name"))
        if analogies:
            related_analogies.extend(analogies)
    related_analogies = ", ".join(dict.fromkeys(related_analogies))
    system_prompt = f"""
    ...
    Kapcsolódó alapanyagok: {', '.join(related_nodes)}
    Kapcsolódó történeti analógiák: {related_analogies}
    """
    return nodes_ctx, simplified_nodes

def extract_json_from_text(text: str):
    if not isinstance(text, str):
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None

def fuzzy_suggest_nodes(query: str, max_suggestions: int = 5):
    if not query:
        return []
    q_norm = normalize_label(query)
    tokens = [t for t in re.split(r'[\s,;:()"\']+', q_norm) if t]
    full_labels = [n.get("Label","") for n in all_nodes if n.get("Label")]
    full_norms = [normalize_label(l) for l in full_labels]
    suggestions = []
    seen = set()
    for tok in tokens:
        if not tok:
            continue
        matches = difflib.get_close_matches(tok, full_norms, n=max_suggestions, cutoff=0.6)
        for m in matches:
            if m not in seen:
                seen.add(m)
                suggestions.append(node_norm_map.get(m, {}).get("Label", m))
    if len(suggestions) < max_suggestions:
        for tok in tokens:
            for i, n in enumerate(full_norms):
                if tok in n and full_labels[i] not in suggestions:
                    suggestions.append(full_labels[i])
                    if len(suggestions) >= max_suggestions:
                        break
            if len(suggestions) >= max_suggestions:
                break
    if len(suggestions) < max_suggestions:
        extra = [l for l in full_labels if l not in suggestions][:max_suggestions - len(suggestions)]
        suggestions.extend(extra)
    return suggestions[:max_suggestions]

def search_recipes_by_query(query: str, max_results: int = 3):
    if not query:
        return []
    q_norm = query.lower()
    q_tokens = [t for t in re.sub(r'[^a-z0-9\s]', ' ', q_norm).split() if len(t) > 1]
    matches = []
    for r in full_recipe_corpus:
        text = (r.get('full_text') or "").lower()
        title = (r.get('title') or "").lower()
        score = 0
        for tok in q_tokens:
            if tok in text:
                score += 2
            if tok in title:
                score += 3
        if score > 0:
            matches.append((score, r))
    matches.sort(key=lambda x: x[0], reverse=True)
    return [ {"title": m[1].get("title",""), "excerpt": (m[1].get("full_text","")[:400])} for m in matches[:max_results] ]

def analyze_query_tokens(user_query: str):
    tokens = [t for t in re.split(r'[\s,;:()"\']+', normalize_label(user_query)) if t]
    analysis = []
    for tok in tokens:
        item = {"token": tok, "base": tok, "role": None, "status": None, "strategy": None, "mapped_to": None, "confidence": 0.0}
        if tok in GENERIC_TOKENS:
            item["role"] = "generic_food"
            item["status"] = "generic"
            item["strategy"] = "ignore_for_node_selection"
            item["confidence"] = 0.2
            analysis.append(item)
            continue
        if tok in ANACHRONISTIC_INGREDIENTS:
            item["role"] = "ingredient"
            item["status"] = "anachronistic"
            mapped = HISTORICAL_ANALOGY_MAP.get(tok)
            if mapped:
                item["mapped_to"] = mapped
                item["strategy"] = "historical_analogy"
                item["confidence"] = 0.6
            else:
                item["mapped_to"] = None
                item["strategy"] = "analogy_required_manual"
                item["confidence"] = 0.3
            analysis.append(item)
            continue
        if tok.endswith('os') or tok.endswith('ós') or tok.endswith('es') or tok.endswith('és') or tok.endswith('i'):
            base = tok
            if tok.endswith('os') or tok.endswith('ós') or tok.endswith('es') or tok.endswith('és'):
                base = tok[:-2]
            elif tok.endswith('i') and len(tok) > 3:
                base = tok[:-1]
            item["base"] = base
            item["role"] = "flavour_descriptor"
            item["status"] = "descriptor"
            mapped_label = None
            norm_base = normalize_label(base)
            if norm_base in SYNONYM_MAP:
                for s in SYNONYM_MAP[norm_base]:
                    if normalize_label(s) in node_norm_map:
                        mapped_label = node_norm_map[normalize_label(s)].get("Label")
                        item["mapped_to"] = [mapped_label]
                        item["strategy"] = "synonym_map"
                        item["confidence"] = 0.8
                        break
            if not mapped_label and norm_base in node_norm_map:
                item["mapped_to"] = [node_norm_map[norm_base].get("Label")]
                item["strategy"] = "direct_node_match"
                item["confidence"] = 0.85
            if not item.get("mapped_to"):
                analogs = HISTORICAL_ANALOGY_MAP.get(norm_base) or HISTORICAL_ANALOGY_MAP.get(tok)
                if analogs:
                    item["mapped_to"] = analogs
                    item["strategy"] = "historical_analogy_for_descriptor"
                    item["confidence"] = 0.55
                else:
                    fuzzy = fuzzy_suggest_nodes(base, max_suggestions=1)
                    if fuzzy:
                        item["mapped_to"] = fuzzy
                        item["strategy"] = "fuzzy_fallback"
                        item["confidence"] = 0.4
                    else:
                        item["mapped_to"] = None
                        item["strategy"] = "no_mapping"
                        item["confidence"] = 0.25
            analysis.append(item)
            continue
        norm_tok = normalize_label(tok)
        if norm_tok in SYNONYM_MAP:
            for s in SYNONYM_MAP[norm_tok]:
                if normalize_label(s) in node_norm_map:
                    item["role"] = "ingredient"
                    item["status"] = "direct_synonym"
                    item["mapped_to"] = [node_norm_map[normalize_label(s)].get("Label")]
                    item["strategy"] = "synonym_map"
                    item["confidence"] = 0.9
                    break
            if item["mapped_to"]:
                analysis.append(item)
                continue
        if norm_tok in node_norm_map:
            item["role"] = "ingredient"
            item["status"] = "direct_node"
            item["mapped_to"] = [node_norm_map[norm_tok].get("Label")]
            item["strategy"] = "direct_node_match"
            item["confidence"] = 0.95
            analysis.append(item)
            continue
        close = difflib.get_close_matches(norm_tok, list(node_norm_map.keys()), n=1, cutoff=0.75)
        if close:
            item["role"] = "ingredient"
            item["status"] = "close_match"
            item["mapped_to"] = [node_norm_map[close[0]].get("Label")]
            item["strategy"] = "close_string_match"
            item["confidence"] = 0.75
            analysis.append(item)
            continue
        if 'bors' in norm_tok or 'pepper' in norm_tok or 'pink' in norm_tok:
            b_candidates = [k for k in node_norm_map.keys() if 'bors' in k or 'pepper' in k or 'tiszta borssal' in k or 'rózsabors' in k]
            if b_candidates:
                cand = difflib.get_close_matches(norm_tok, b_candidates, n=1, cutoff=0.35)
                if cand:
                    item["role"] = "ingredient"
                    item["status"] = "pepper_family"
                    item["mapped_to"] = [node_norm_map[cand[0]].get("Label")]
                    item["strategy"] = "special_pepper_rules"
                    item["confidence"] = 0.85
                    analysis.append(item)
                    continue
        fuzzy = fuzzy_suggest_nodes(tok, max_suggestions=1)
        if fuzzy:
            item["role"] = "ingredient"
            item["status"] = "fuzzy_suggest"
            item["mapped_to"] = fuzzy
            item["strategy"] = "fuzzy"
            item["confidence"] = 0.35
            analysis.append(item)
            continue
        item["role"] = "unknown"
        item["status"] = "no_mapping"
        item["strategy"] = "no_mapping"
        item["confidence"] = 0.0
        analysis.append(item)
        if tok in HISTORICAL_DISH_STRUCTURE_MAP:
            item["role"] = "dish_structure"
            item["status"] = "historically_interpretable"
            item["mapped_to"] = HISTORICAL_DISH_STRUCTURE_MAP[tok]["historical_equivalents"]
            item["strategy"] = "source_based_structural_mapping"
            item["confidence"] = HISTORICAL_DISH_STRUCTURE_MAP[tok]["confidence"]
            analysis.append(item)
            continue
    return analysis

def build_reasoning_paragraph(token_analysis: list) -> str:
    """
    A token-analízisből folyó szöveges, narratív reasoning-et készít.
    """
    sentences = []

    for item in token_analysis:
        tok = item["token"]
        role = item["role"]
        status = item["status"]
        strategy = item["strategy"]
        mapped = item.get("mapped_to")

        if role == "flavour_descriptor":
            s = f"A „{tok}” kifejezés ízleíróként jelenik meg, amely nem önálló alapanyagot, hanem érzékszervi irányt jelöl."
        elif status == "anachronistic":
            s = f"A „{tok}” modern alapanyagnak számít, ezért történeti analógiával került értelmezésre."
        elif strategy == "historical_analogy" and mapped:
            s = f"A „{tok}” esetében a történeti források alapján a következő analóg összetevők jöhetnek szóba: {', '.join(mapped)}."
        elif strategy == "direct_node_match":
            s = f"A „{tok}” egyértelműen azonosítható a történeti adatbázisban szereplő alapanyagként."
        elif strategy == "fuzzy_fallback":
            s = f"A „{tok}” pontos megfelelője nem szerepel az adatbázisban, ezért hangalaki hasonlóság alapján történt becslés."
        else:
            s = f"A „{tok}” értelmezése bizonytalan, ezért csak korlátozottan befolyásolta a keresést."

        sentences.append(s)

    return " ".join(sentences)

def gpt_search_recipes(user_query):
    query_lower = (user_query or "").strip()
    matched_recipes = []
    if query_lower:
        q_tokens = [t for t in re.sub(r'[^a-z0-9\s]', ' ', query_lower.lower()).split() if len(t) > 1]
    else:
        q_tokens = []
    for recipe in full_recipe_corpus:
        text = (recipe.get('full_text') or "").lower()
        if not text:
            continue
        if q_tokens and any(tok in text for tok in q_tokens):
            matched_recipes.append(recipe)
            if len(matched_recipes) >= 10:
                break
    nodes_ctx, simplified_nodes = build_gpt_context(all_nodes, historical_recipes, perfect_ings, user_query=query)
    system_prompt = f"""
    Te egy XVII. századi magyar szakácskönyv stílusában írsz AI Ajánlást.
    Feladat: a felhasználói kifejezéseket esszészerűen értelmezd, kulturális és érzéki szempontokat összekapcsolva.
    Ne listázz, hanem folyékony prózában indokold, miért és hogyan értelmezted a szavakat történeti gasztronómiai logika mentén.
    A cél: az ízélmény, textúra és jelentés történeti rekonstrukciója.
    Felhasználói query: {user_query}
    Kapcsolódó alapanyagok: {', '.join([n['name'] for n in simplified_nodes])}    node_analogies = []
    for node in simplified_nodes:
        node_analogies.extend(HISTORICAL_ANALOGY_MAP.get(node["name"], []))
    related_analogies = ", ".join(dict.fromkeys(node_analogies))
    """
    top_matched = matched_recipes[:5]
    matched_preview = [{"title": r.get("title", ""), "excerpt": (r.get("full_text") or "")[:400]} for r in top_matched]
    try:
        full_labels = sorted({n.get("Label", "") for n in all_nodes if n.get("Label")})
        full_labels_preview = json.dumps(full_labels[:300], ensure_ascii=False)
    except Exception:
        full_labels_preview = "[]"
    try:
        perfect_preview = (json.dumps(perfect_ings[:50], ensure_ascii=False) if isinstance(perfect_ings, list) else json.dumps(perfect_ings, ensure_ascii=False))
    except Exception:
        perfect_preview = "[]"
    user_prompt = f"""
Nyelv: magyar

Felhasználói lekérdezés: "{user_query}"

Elérhető csomópontok (rövid mintavétel):
{json.dumps(nodes_ctx[:40], ensure_ascii=False)}

Található történeti recept-részletek:
{json.dumps(matched_preview, ensure_ascii=False)}

Teljes node-címek (rövid előnézet):
{full_labels_preview}

Tökéletes alapanyagok (rövid):
{perfect_preview}

Utasítások: system_prompt = """
Először folyó, magyar (vagy a felhasználó által írt bármilyen nyelven) nyelvű magyarázó szövegben írd le, hogyan értelmezed a felhasználó kérdését történeti-gasztronómiai szempontból. Ezután – külön blokkban – add meg a strukturált adatokat JSON formátumban. A szöveg legyen élvezetes, értelmező jellegű, ne csak felsorolás. Ha a felhasználó olyan kifejezést említ, amely nincs a node-listában, térképezd a legközelebbi ismert node-ra és részletezd a mapping indoklását a "reasoning" mezőben. Javasolj legfeljebb 5 node-ot és legfeljebb 3 történeti receptcímeket.
"""
    try:
        response = client.responses.create(model="gpt-5.1", input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], max_output_tokens=900)
        raw = response.output_text if hasattr(response, "output_text") else (response.get("output_text") if isinstance(response, dict) else str(response))
        parsed = extract_json_from_text(raw)
        if parsed and isinstance(parsed, dict):
            if "suggested_nodes" in parsed and "suggested_recipes" in parsed:
                return parsed
        raise ValueError("Invalid JSON from model")
    except Exception:
        suggested_nodes = fuzzy_suggest_nodes(user_query, max_suggestions=5)
        suggested_recipes = [r["title"] for r in search_recipes_by_query(user_query, max_results=3)]
        analysis = analyze_query_tokens(user_query)
        reasoning_parts = []
        mapped_nodes = []
        for item in analysis:
            tok = item["token"]
            status = item.get("status", "unknown")
            strat = item.get("strategy", "none")
            conf = item.get("confidence", 0.0)
            mapped = item.get("mapped_to")
            if isinstance(mapped, list):
                mapped_display = ", ".join([str(m) for m in mapped if m])
            else:
                mapped_display = str(mapped) if mapped else "—"
            reasoning_parts.append(f'"{tok}" → státusz: {status}; stratégia: {strat}; leképezés: {mapped_display}; bizalom: {conf:.2f}')
            if item.get("mapped_to"):
                if isinstance(item["mapped_to"], list):
                    for m in item["mapped_to"]:
                        if isinstance(m, str) and normalize_label(m) in node_norm_map:
                            mapped_nodes.append(node_norm_map[normalize_label(m)].get("Label"))
                        elif isinstance(m, str):
                            mapped_nodes.append(m)
                else:
                    m = item["mapped_to"]
                    if isinstance(m, str) and normalize_label(m) in node_norm_map:
                        mapped_nodes.append(node_norm_map[normalize_label(m)].get("Label"))
                    elif isinstance(m, str):
                        mapped_nodes.append(m)
        mapped_nodes = [m for m in mapped_nodes if m]
        combined_suggestions = []
        seen = set()
        for n in mapped_nodes + suggested_nodes:
            if n and n not in seen:
                combined_suggestions.append(n)
                seen.add(n)
            if len(combined_suggestions) >= 5:
                break
        if not combined_suggestions:
            combined_suggestions = suggested_nodes[:5]
        analysis = analyze_query_tokens(user_query)
        reasoning_text = build_reasoning_paragraph(analysis)
        result = {
            "suggested_nodes": combined_suggestions,
            "suggested_recipes": suggested_recipes,
            "reasoning": reasoning,
            "mapping": analysis
        }
        return result

def max_similarity_to_historical(candidate: str, historical_list: list) -> float:
    if not candidate or not historical_list:
        return 0.0
    candidate_norm = re.sub(r'\s+', ' ', candidate.strip().lower())
    max_sim = 0.0
    for h in historical_list:
        text = ""
        if isinstance(h, dict):
            text = h.get("text", "") or h.get("original_text", "") or h.get("excerpt", "") or h.get("title", "")
        else:
            text = str(h)
        text_norm = re.sub(r'\s+', ' ', strip_icon_ligatures(text).strip().lower())
        if not text_norm:
            continue
        sim = difflib.SequenceMatcher(None, candidate_norm, text_norm).ratio()
        if sim > max_sim:
            max_sim = sim
    return float(max_sim)

def generate_ai_recipe(selected, connected, historical, user_query=None, samples=4, temperature=0.7):
    system_prompt = """
Írj egy XVII. századi magyar stílusú, választékos és beszédes receptet. Szabályok:
- 70–110 szó között
- archaikus, mégis érthető magyar stílus, összetett mondatokkal és gazdag szókinccsel
- használj lehetőleg csak a megadott összetevőket/kapcsolatokat; ha a felhasználói lekérdezés modern kifejezést tartalmaz, térképezd historikus megfelelőre és indokold röviden
- kerüld az adott történeti példák szó szerinti másolását; ha a generált szöveg >60% hasonlóságot mutat egy történeti példához, generálj újat
- a válasz CSAK ÉS KIZÁRÓLAG érvényes JSON legyen magyar mezőnevekkel: legalább 'title', 'archaic_recipe', 'confidence', 'novelty_score', 'word_count'
- legyél gondolkodó és okos: a 'reasoning' mezőben röviden írd le, hogyan képzeled el a mappingot, ha volt
"""
    user_prompt = f"""
Felhasználói keresés: {user_query}

Központi elem: {selected}

Kapcsolódó elemek (name,type,degree):
{json.dumps(connected, ensure_ascii=False)}

Történeti példák (rövid):
{json.dumps(historical, ensure_ascii=False)}

Ha valamelyik kapcsolt elem bizonytalan, térképezd a legplausibilisebb történeti alapanyagra. Adj vissza csak JSON-t.
"""
    candidates = []
    raw_texts = []
    for i in range(samples):
        try:
            response = client.responses.create(model="gpt-5.1", input=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=temperature, max_output_tokens=700)
            ai_text = response.output_text.strip() if hasattr(response, "output_text") else str(response)
            parsed = extract_json_from_text(ai_text)
            if parsed and isinstance(parsed, dict):
                candidates.append(parsed)
                raw_texts.append(parsed.get("archaic_recipe", "") or parsed.get("text", "") or "")
            else:
                if ai_text:
                    raw_texts.append(ai_text)
        except Exception:
            continue
    if not candidates and not raw_texts:
        return {"title": "Hiba történt", "archaic_recipe": "A recept generálása sikertelen volt: nincs érvényes válasz.", "confidence": "low", "word_count": 0, "novelty_score": 0.0}
    hist_texts = []
    for h in historical:
        if isinstance(h, dict):
            hist_texts.append(h.get("text", "") or h.get("original_text", "") or h.get("excerpt", "") or h.get("title", ""))
        else:
            hist_texts.append(str(h))
    best = None
    best_novelty = -1.0
    for cand in candidates:
        recipe_text = cand.get("archaic_recipe", "") or cand.get("text", "") or ""
        sim = max_similarity_to_historical(recipe_text, hist_texts)
        novelty = 1.0 - sim
        cand["novelty_score"] = round(novelty, 4)
        wc = len(recipe_text.split())
        cand["word_count"] = wc
        if 70 <= wc <= 110:
            cand["confidence"] = "high"
        elif 50 <= wc <= 130:
            cand["confidence"] = "medium"
        else:
            cand["confidence"] = "low"
        if novelty > best_novelty:
            best_novelty = novelty
            best = cand
    if not best:
        fallback_text = raw_texts[0] if raw_texts else ""
        wc = len(fallback_text.split())
        return {"title": selected, "archaic_recipe": fallback_text, "confidence": "low", "word_count": wc, "novelty_score": 0.0}
    return best

banner_path = "83076027-f357-4e82-8716-933911048498.png"
if os.path.exists(banner_path):
    with open(banner_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <div style="position: relative; text-align: center; margin-bottom: 3rem; border-radius: 16px; overflow: hidden; box-shadow: 0 8px 16px rgba(0, 0, 0, 0.7); height: 300px;">
        <img src="data:image/png;base64,{img_data}" style="width: 100%; height: 300px; object-fit: cover; display: block;">
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 100%; z-index: 10;">
            <h1 style="font-size: 3rem; color: white; text-shadow: 3px 3px 10px black, 0 0 20px rgba(0,0,0,0.8); margin: 0; font-family: 'Cinzel', serif;">
                Közrendek Ízhálója
            </h1>
            <p style="font-size: 1.3rem; font-style: italic; color: white; text-shadow: 2px 2px 8px black, 0 0 15px rgba(0,0,0,0.8); margin-top: 0.5rem; font-family: 'Crimson Text', serif;">
                Fedezd fel a XVII. századi magyar konyha ízhálózatát
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #800000 0%, #2d2d2d 50%, #1a1a1a 100%); 
                padding: 2.5rem 2rem; 
                border-radius: 16px; 
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.7);
                margin-bottom: 2rem;
                border: 3px solid #ccaa77;">
        <h1 style="font-size: 3rem; color: white; text-shadow: 3px 3px 8px black; margin: 0; text-align: center;">
            Közrendek Ízhálója
        </h1>
        <p style="font-size: 1.3rem; font-style: italic; color: #e8dcc8; text-shadow: 2px 2px 6px black; margin-top: 0.5rem; text-align: center;">
            Fedezd fel a XVII. századi magyar konyha ízhálózatát
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%); border: 3px solid #ccaa77; border-radius: 12px; padding: 2rem; margin: 2rem 0; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);">
    <h3 style="color: #ccaa77; font-family: 'Cinzel', serif; margin-bottom: 1rem; text-align: center;">
        🔍 Intelligens Keresés
    </h3>
    <p style="text-align: center; color: #e8dcc8; font-family: 'Crimson Text', serif; font-style: italic; margin-bottom: 1.5rem;">
        Írj le egy ételt vagy alapanyagot, és az AI megtalálja a kapcsolódó node-okat és történeti recepteket!
    </p>
</div>
""", unsafe_allow_html=True)

cols = st.columns(4)
data = [
    {"title": "Csomópontok / Nodes", "value": str(len(all_nodes)), "desc": "Minden egyes node egy alapanyagot, molekulát vagy receptet jelöl a hálózatban; ezek alkotják az összefüggő ízhálózat vázát."},
    {"title": "Élek / Edges", "value": str(len(all_edges)), "desc": "A kapcsolatok az összefüggéseket mutatják: ki milyen alapanyaggal, molekulával vagy recepttel van összekötve."},
    {"title": "Receptek", "value": str(len(historical_recipes)), "desc": "Történeti receptek száma; ezek adnak kulcsot a node-ok jelentéséhez a XVII. századi kontextusban."},
    {"title": "Átlag Fokszám / Degree", "value": f"{(sum([int(n.get('Degree', 0) or 0) for n in all_nodes]) / max(len(all_nodes),1)):.1f}", "desc": "Az átlag fokszám azt mutatja, mennyi kapcsolat jut egy csomópontra — a magasabb érték gazdagabb hálózati integrációt jelent."}
]
for col, info in zip(cols, data):
    with col:
        st.markdown(f"""
        <div class="carousell-card">
            <div class="card-title">{info["title"]}</div>
            <div class="card-value">{info["value"]}</div>
            <div class="card-desc">{info["desc"]}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height: 1cm;'></div>", unsafe_allow_html=True)

col_search, col_sort = st.columns([3, 1])
with col_search:
    query = st.text_input("Keresés", placeholder="🔍 pl. 'rózsabors', 'édes sütemény mandulával', 'boros leves'...", key="search_input", label_visibility="collapsed")

    if query and st.button("🤖 AI Keresés", key="gpt_search"):
        if "gpt_search_results" in st.session_state:
            del st.session_state["gpt_search_results"]
        if "selected" in st.session_state:
            del st.session_state["selected"]
        if "connected" in st.session_state:
            del st.session_state["connected"]
        if "historical_recipe" in st.session_state:
            del st.session_state["historical_recipe"]
        if "ai_recipe" in st.session_state:
            del st.session_state["ai_recipe"]
        with st.spinner("🔍 AI elemzi a kérést..."):
            search_results = gpt_search_recipes(query)
            st.session_state["gpt_search_results"] = search_results
            st.session_state["search_query"] = query
            try:
                suggested = search_results.get("suggested_nodes", []) or []
                if suggested:
                    top_name = str(suggested[0])
                    top_norm = normalize_label(top_name)
                    node_obj = node_norm_map.get(top_norm)
                    if not node_obj:
                        possible = fuzzy_suggest_nodes(top_name, max_suggestions=1)
                        node_label = possible[0] if possible else top_name
                        node_obj = node_norm_map.get(normalize_label(node_label))
                    if node_obj:
                        sel = node_obj.get("Label")
                        sel_norm = normalize_label(sel)
                        related_norms = []
                        for e in all_edges:
                            es = e.get("norm_source", "")
                            et = e.get("norm_target", "")
                            if sel_norm and es == sel_norm:
                                related_norms.append(et)
                            elif sel_norm and et == sel_norm:
                                related_norms.append(es)
                        related_norms = set([r for r in related_norms if r])
                        connected = []
                        for rn in related_norms:
                            node = node_norm_map.get(rn)
                            if node:
                                connected.append({"name": node.get("Label"), "degree": int(node.get("Degree", 0) or 0), "type": node.get("node_type", "unknown")})
                        historical_recipe = [{"title": strip_icon_ligatures(r.get("title", "Névtelen")), "text": strip_icon_ligatures(r.get("original_text", "")[:300])} for r in historical_recipes if sel.lower() in str(r).lower()][:5]
                        st.session_state["selected"] = sel
                        st.session_state["connected"] = connected
                        st.session_state["historical_recipe"] = historical_recipe
                        with st.spinner("⏳ AI receptgenerálás..."):
                            ai_recipe = generate_ai_recipe(sel, connected, historical_recipe, user_query=query)
                            st.session_state["ai_recipe"] = ai_recipe
            except Exception:
                pass

if "sort_option" not in st.session_state:
    st.session_state.sort_option = "📝 Név (A–Z)"

OPTIONS = ["📝 Név (A–Z)","🔁 Név (Z–A)","📊 Degree ↓","📈 Degree ↑"]
if "sort_mode" not in st.session_state:
    st.session_state.sort_mode = "name_asc"
with col_sort:
    st.markdown("#### Rendezés")
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    with c1:
        if st.button("📝 Név A–Z", use_container_width=True):
            st.session_state.sort_mode = "name_asc"
    with c2:
        if st.button("🔁 Név Z–A", use_container_width=True):
            st.session_state.sort_mode = "name_desc"
    with c3:
        if st.button("📊 Degree ↓", use_container_width=True):
            st.session_state.sort_mode = "deg_desc"
    with c4:
        if st.button("📈 Degree ↑", use_container_width=True):
            st.session_state.sort_mode = "deg_asc"

def _node_type(n):
    if not isinstance(n, dict):
        return "Egyéb"
    return n.get("node_type") or n.get("Type") or n.get("type") or "Egyéb"

def _node_label(n):
    if not isinstance(n, dict):
        return ""
    return strip_icon_ligatures(n.get("Label") or n.get("label") or "")

def _node_degree(n):
    try:
        return int(n.get("Degree", n.get("degree", 0) or 0))
    except Exception:
        return 0

node_types = sorted({ _node_type(n) for n in all_nodes if isinstance(n, dict) })
label_map = {t: f"🧱 {t}" if t=="Alapanyag" else ("🧪 "+t if t=="Molekula" else ("📖 "+t if t=="Recept" else t)) for t in node_types}
choices = [label_map[t] for t in node_types]
node_type_filter = st.multiselect("Kategória", options=node_types, default=node_types, key="node_type_filter", help="Szűrés csomópont-típus szerint")
node_type_filter_set = set(node_type_filter) if node_type_filter else set(node_types)
filtered_nodes = []

if "gpt_search_results" not in st.session_state or not query:
    candidates = (all_nodes or [])
else:
    suggested = st.session_state["gpt_search_results"].get("suggested_nodes", [])
    candidates = []
    for n in (all_nodes or []):
        if not isinstance(n, dict):
            continue
        if _node_type(n) not in node_type_filter_set:
            continue
        label = n.get("Label", "")
        if not query or query.lower() in str(label).lower() or label in suggested:
            candidates.append(n)

if "gpt_search_results" not in st.session_state or not query:
    for n in candidates:
        if not isinstance(n, dict):
            continue
        if _node_type(n) in node_type_filter_set:
            label = n.get("Label", "")
            if not query or query.lower() in str(label).lower():
                filtered_nodes.append(n)
else:
    filtered_nodes = candidates

mode = st.session_state.sort_mode
if mode == "name_asc":
    filtered_nodes.sort(key=lambda x: _node_label(x).lower())
elif mode == "name_desc":
    filtered_nodes.sort(key=lambda x: _node_label(x).lower(), reverse=True)
elif mode == "deg_desc":
    filtered_nodes.sort(key=lambda x: _node_degree(x), reverse=True)
elif mode == "deg_asc":
    filtered_nodes.sort(key=lambda x: _node_degree(x))

cols = st.columns(6)
for i, n in enumerate(filtered_nodes[:60]):
    type_emoji = {'Alapanyag': '🧱', 'Molekula': '🧪', 'Recept': '📖', 'Egyéb': '⚪'}.get(n.get('node_type'), '⚪')
    clean_label = strip_icon_ligatures(n.get('Label', ''))
    if cols[i % 6].button(f"{type_emoji} {clean_label}", key=f"node_{i}"):
        sel = n.get("Label", "")
        sel_norm = normalize_label(sel)
        related_norms = []
        for e in all_edges:
            es = e.get("norm_source", "")
            et = e.get("norm_target", "")
            if sel_norm and es == sel_norm:
                related_norms.append(et)
            elif sel_norm and et == sel_norm:
                related_norms.append(es)
        related_norms = set([r for r in related_norms if r])
        connected = []
        for rn in related_norms:
            node = node_norm_map.get(rn)
            if node:
                connected.append({"name": node.get("Label"), "degree": int(node.get("Degree", 0) or 0), "type": node.get("node_type", "unknown")})
        historical_recipe = [{"title": strip_icon_ligatures(r.get("title", "Névtelen")), "text": strip_icon_ligatures(r.get("original_text", "")[:300])} for r in historical_recipes if sel.lower() in str(r).lower()][:5]
        st.session_state["selected"] = sel
        st.session_state["connected"] = connected
        st.session_state["historical_recipe"] = historical_recipe
        with st.spinner("⏳ AI receptgenerálás..."):
            ai_recipe = generate_ai_recipe(sel, connected, historical_recipe, user_query=st.session_state.get("search_query"))
            st.session_state["ai_recipe"] = ai_recipe
        st.rerun()

if "gpt_search_results" in st.session_state:
    results = st.session_state["gpt_search_results"]
    reasoning = strip_icon_ligatures(results.get('reasoning', ''))
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #2d2d2d, #1a1a1a); border: 2px solid #ccaa77; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
        <h4 style="color: #ccaa77; font-family: 'Cinzel', serif; margin-bottom: 0.5rem;">
            💡 AI Ajánlás: "{strip_icon_ligatures(st.session_state.get('search_query', ''))}"
        </h4>
        <p style="color: #e8dcc8; font-family: 'Crimson Text', serif; font-style: italic;">
            {reasoning}
        </p>
    </div>
    """, unsafe_allow_html=True)
    if results.get("suggested_nodes"):
        st.markdown("**🎯 Ajánlott alapanyagok/csomópontok (nodes):**")
        cols_suggested = st.columns(min(len(results["suggested_nodes"]), 5))
        for i, node_name in enumerate(results["suggested_nodes"][:5]):
            clean_node_name = strip_icon_ligatures(str(node_name))
            node = node_norm_map.get(normalize_label(clean_node_name))
            if not node:
                poss = fuzzy_suggest_nodes(clean_node_name, max_suggestions=1)
                if poss:
                    node = node_norm_map.get(normalize_label(poss[0]))
            if node and i < len(cols_suggested):
                type_emoji = {'Alapanyag': '🧱', 'Molekula': '🧪', 'Recept': '📖', 'Egyéb': '⚪'}.get(node.get('node_type'), '⚪')
                clean_label = strip_icon_ligatures(node.get('Label', ''))
                if cols_suggested[i].button(f"{type_emoji} {clean_label}", key=f"suggested_{i}"):
                    sel = node.get("Label", "")
                    sel_norm = normalize_label(sel)
                    related_norms = []
                    for e in all_edges:
                        es = e.get("norm_source", "")
                        et = e.get("norm_target", "")
                        if sel_norm and es == sel_norm:
                            related_norms.append(et)
                        elif sel_norm and et == sel_norm:
                            related_norms.append(es)
                    related_norms = set([r for r in related_norms if r])
                    connected = []
                    for rn in related_norms:
                        nnode = node_norm_map.get(rn)
                        if nnode:
                            connected.append({"name": nnode.get("Label"), "degree": int(nnode.get("Degree", 0) or 0), "type": nnode.get("node_type", "unknown")})
                    historical_recipe = [{"title": strip_icon_ligatures(r.get("title", "Névtelen")), "text": strip_icon_ligatures(r.get("original_text", "")[:300])} for r in historical_recipes if sel.lower() in str(r).lower()][:5]
                    st.session_state["selected"] = sel
                    st.session_state["connected"] = connected
                    st.session_state["historical_recipe"] = historical_recipe
                    with st.spinner("⏳ AI receptgenerálás..."):
                        ai_recipe = generate_ai_recipe(sel, connected, historical_recipe, user_query=st.session_state.get("search_query"))
                        st.session_state["ai_recipe"] = ai_recipe
                    st.rerun()
    if results.get("suggested_recipes"):
        st.markdown("**📖 Releváns történeti receptek:**")
        for recipe_title in results["suggested_recipes"][:3]:
            clean_recipe_title = strip_icon_ligatures(str(recipe_title))
            recipe = next((r for r in historical_recipes if strip_icon_ligatures(r.get("title", "")).lower() == clean_recipe_title.lower()), None)
            if recipe:
                clean_title = strip_icon_ligatures(recipe.get('title', 'Névtelen'))
                clean_text = strip_icon_ligatures(recipe.get('original_text', '')[:400])
                with st.expander(f"📜 {clean_title}"):
                    st.markdown(clean_text + "...")

if "selected" in st.session_state:
    st.markdown("---")
    st.markdown(f"<h2 style='text-align: center;'>🎯 {strip_icon_ligatures(st.session_state['selected'])}</h2>", unsafe_allow_html=True)
    st.markdown("### 🗺️ Hálózati Térkép")
    fig = create_network_graph(st.session_state["selected"], st.session_state["connected"])
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📚 Történeti Példák")
        recipe = st.session_state.get("historical_recipe", [])
        if recipe:
            for ex in recipe[:3]:
                clean_title = strip_icon_ligatures(ex.get('title', 'Névtelen'))
                clean_text = strip_icon_ligatures(ex.get('text', ''))
                with st.expander(f"📖 {clean_title}"):
                    st.markdown(clean_text)
        else:
            st.info("Nincs történeti példa")
    with col2:
        st.markdown("### 🤖 AI Generált Recept")
        ai_recipe = st.session_state.get("ai_recipe")
        if ai_recipe:
            clean_ai_title = strip_icon_ligatures(ai_recipe.get('title', 'Cím nélkül'))
            clean_ai_text = strip_icon_ligatures(ai_recipe.get('archaic_recipe', ''))
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%); border: 3px solid #ccaa77; border-radius: 12px; padding: 2rem; box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5);">
                <h3 style="color: #ccaa77; font-family: 'Cinzel', serif; margin-bottom: 1rem;">{clean_ai_title}</h3>
                <p style="color: #e8dcc8; font-family: 'Crimson Text', serif; line-height: 1.8; font-size: 1.1rem;">{clean_ai_text}</p>
                <div style="display: flex; gap: 1rem; margin-top: 1.5rem;">
                    <span style="background: #800000; padding: 0.6rem 1rem; border-radius: 8px; color: #ccaa77; font-weight: 600;">✓ {ai_recipe.get('confidence', 'unknown')}</span>
                    <span style="background: #800000; padding: 0.6rem 1rem; border-radius: 8px; color: #ccaa77; font-weight: 600;">📝 {ai_recipe.get('word_count', 0)} szó</span>
                    <span style="background: #800000; padding: 0.6rem 1rem; border-radius: 8px; color: #ccaa77; font-weight: 600;">✨ {int(ai_recipe.get('novelty_score', 0.0)*100)}% új</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Hiba történt a generálás során")

st.markdown("---")
st.markdown("""
<div style="text-align: center; margin: 3rem 0 2rem 0;">
    <h3 style="color: #ccaa77; font-family: 'Cinzel', serif; margin-bottom: 1.5rem;">
        🧭 További oldalak
    </h3>
</div>
""", unsafe_allow_html=True)

nav_col1, nav_col2 = st.columns(2)
with nav_col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%); 
                border: 2px solid #ccaa77; 
                border-radius: 12px; 
                padding: 2rem; 
                text-align: center;
                margin-bottom: 1rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📖</div>
        <h4 style="color: #ccaa77; font-family: 'Cinzel', serif; margin-bottom: 0.5rem;">A Projektről</h4>
        <p style="color: #e8dcc8; font-size: 0.95rem; opacity: 0.8;">Történet, módszertan és források</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("📖 Tovább a Projektről oldalra", key="nav_about", use_container_width=True):
        try:
            st.experimental_set_query_params(page="About")
            st.experimental_rerun()
        except Exception:
            pass

with nav_col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%); 
                border: 2px solid #ccaa77; 
                border-radius: 12px; 
                padding: 2rem; 
                text-align: center;
                margin-bottom: 1rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
        <h4 style="color: #ccaa77; font-family: 'Cinzel', serif; margin-bottom: 0.5rem;">Analitika Dashboard</h4>
        <p style="color: #e8dcc8; font-size: 0.95rem; opacity: 0.8;">Részletes statisztikák és eloszlások</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("📖 Tovább az elemzői oldalra", key="nav_analytics", use_container_width=True):
        try:
            st.experimental_set_query_params(page="analytics")
            st.experimental_rerun()
        except Exception:
            pass

st.markdown("""
<p style="text-align: center; color: #888; font-size: 0.9rem; margin-top: 1.5rem;">
    💡 <em>Vagy használd a bal felső sarokban lévő menüt (>>) a navigáláshoz!</em>
</p>
""", unsafe_allow_html=True)

st.markdown(textwrap.dedent("""
<div style="text-align: center; padding: 3.5rem 2.5rem; background: linear-gradient(145deg, #1a0d0d 0%, #2b0f12 100%); color: #f5efe6; margin-top: 5rem; border-radius: 20px; border: 2px solid #ccaa77; box-shadow: 0 12px 40px rgba(0,0,0,0.6);">
    <p style="font-family: 'Cinzel', serif; font-size: 1.6rem; letter-spacing: 0.08em; margin-bottom: 0.3rem; color: #e8c896; text-shadow: 0 2px 6px rgba(0,0,0,0.8);">Közrendek Ízhálója</p>
    <div style="width: 120px; height: 2px; background: linear-gradient(90deg, transparent, #ccaa77, transparent); margin: 0.8rem auto 1.2rem auto;"></div>
    <p style="font-family: 'Crimson Text', serif; font-size: 1.05rem; opacity: 0.9; margin: 0.2rem 0 1.6rem 0; letter-spacing: 0.04em;">Hálózatelemzés • Történeti források • AI-alapú generálás</p>
    <p style="font-size: 0.95rem; line-height: 1.7; max-width: 820px; margin: 0 auto; opacity: 0.85; color: #efe6d8;">
        A projekt Barabási Albert-László hálózatkutatásaira és a
        <em>„Szakácsmesterségnek könyvecskéje"</em> (Tótfalusi Kis Miklós, 1698)
        című szakácskönyv digitális elemzésére épül.<br>
        Forrás: Magyar Elektronikus Könyvtár (MEK), Országos Széchényi Könyvtár
    </p>
    <p style="font-size: 0.9rem; margin-top: 1.4rem; opacity: 0.75; color: #d6b98c; letter-spacing: 0.06em;">
        Felhasznált technológiák: Streamlit • NetworkX • Plotly • SciPy • OpenAI GPT-5.1; 5-nano; 5-mini • Claude • Grok
    </p>
    <div style="width: 100%; height: 1px; background: linear-gradient(90deg, transparent, rgba(204,170,119,0.4), transparent); margin: 2rem 0 1.2rem 0;"></div>
    <p style="font-size: 0.85rem; opacity: 0.55; letter-spacing: 0.05em; color: #cbb58a;">
        © 2025 • Digitális bölcsészet-, társadalom- és hálózattudományi projekt
    </p>
</div>
"""), unsafe_allow_html=True)


