import streamlit as st
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

# ===============================
# STREAMLIT KONFIG
# ===============================
st.set_page_config(
    page_title="Közrendek Ízhálója",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== MODERN CSS - SÖTÉT TÉMA =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400&display=swap');
    
    /* Reset & Base */
    .main {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 50%, #1a1a1a 100%) !important;
        background-image: url("https://www.transparenttextures.com/patterns/dark-leather.png") !important;
        padding: 0 !important;
    }
    
    .block-container {
        padding: 2rem 3rem !important;
        max-width: 1400px !important;
        background: rgba(0, 0, 0, 0.3);
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Cinzel', serif !important;
        color: white !important;
        font-weight: 700 !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        text-align: center;
        margin-bottom: 1rem !important;
    }
    
    /* Scope the body serif font to the main content container so icon ligatures are preserved */
    .block-container p, .block-container div, .block-container span, .block-container li {
        font-family: 'Crimson Text', serif !important;
        color: white !important;
        font-size: 1.05rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #800000 0%, #5c1a1a 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        font-family: 'Cinzel', serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 0.6rem 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
        width: 100%;
        text-align: left;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(204, 170, 119, 0.3);
        background: linear-gradient(135deg, #a52a2a 0%, #722828 100%);
    }
    
    /* Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #ccaa77;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.5);
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'Cinzel', serif !important;
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Crimson Text', serif !important;
        color: white !important;
        font-size: 1rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Kereső input */
    .stTextInput input {
        background-color: #840A13 !important;
        color: #ffffff !important;
    }

    .stTextInput input::placeholder {
        color: #840A13 !important;
        opacity: 1 !important;
        font-style: italic;
    }

    /* ================================
       SELECTBOX – BASEWEB FIX
    ================================ */

    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #840A13 !important;
        border: 2px solid #FF2400 !important;
        border-radius: 8px !important;
    }

    .stSelectbox div[data-baseweb="select"] span {
        color: #f5efe6 !important;
        font-weight: 500;
    }

    div[data-baseweb="popover"] {
        background-color: #840A13 !important;
        border: 2px solid #FF2400 !important;
        border-radius: 12px !important;
        box-shadow: 0 12px 30px rgba(0,0,0,0.75) !important;
    }

    div[role="listbox"] {
        background-color: #840A13 !important;
        padding: 0.4rem 0 !important;
    }

    div[role="listbox"] ul,
    div[role="listbox"] li,
    div[role="listbox"] li > div {
        background-color: #840A13 !important;
    }

    div[role="option"] {
        background-color: #840A13 !important;
        color: #f5efe6 !important;
        font-family: 'Crimson Text', serif !important;
        font-size: 1rem !important;
        padding: 0.8rem 1.2rem !important;
        cursor: pointer !important;
    }

    div[role="option"] span {
        color: #f5efe6 !important;
    }

    div[role="option"]:hover,
    div[role="option"][data-highlighted="true"],
    div[role="listbox"] li > div:hover {
        background-color: #FF2400 !important;
        color: #ffffff !important;
    }

    div[role="option"][aria-selected="true"],
    div[role="listbox"] li > div[aria-selected="true"] {
        background-color: #FF2400 !important;
        font-weight: 600 !important;
        color: #ffffff !important;
    }

    /* Material Icons font kizárása */
    .stButton button,
    .stExpander,
    [data-testid="stMarkdownContainer"] {
        font-family: 'Crimson Text', serif !important;
    }
    
    /* Ligature rendering kikapcsolása */
    * {
        font-variant-ligatures: none !important;
        -webkit-font-variant-ligatures: none !important;
        text-rendering: optimizeLegibility !important;
    }

        /* Header megjelenítése a menüért */
header[data-testid="stHeader"] {
    visibility: visible !important;
    background-color: rgba(26, 26, 26, 0.95) !important;
}

/* MainMenu (Deploy gomb stb.) elrejtése */
#MainMenu {
    visibility: hidden !important;
}

footer[data-testid="stFooter"] {
    visibility: hidden !important;
}

    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(to bottom, #ccaa77, #800000);
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ===== FOOTER CSS =====
st.markdown("""
<style>
.custom-footer {
    max-width: 1100px;
    margin: 4rem auto 2rem auto;
    padding: 3rem 2.5rem;
    text-align: center;
    background: linear-gradient(135deg, #2b0f12 0%, #1a0d0d 100%);
    border: 2.5px solid #ccaa77;
    border-radius: 999px;
    box-shadow:
        0 12px 30px rgba(0,0,0,0.55),
        inset 0 0 0 1px rgba(204,170,119,0.15);
}
</style>
""", unsafe_allow_html=True)


# ===============================
# ENV + OPENAI
# ===============================
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

# ===============================
# ADATBETÖLTÉS
# ===============================
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

    alapanyag_path = _resolve(os.path.join('data', 'recept_alapanyagok_TÖKÉLETES.json'))
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
        except Exception as e1:
            try:
                return pd.read_csv(path, sep=None, engine='python', encoding='utf-8', on_bad_lines='skip')
            except Exception:
                try:
                    return pd.read_csv(path, delimiter=default_sep, encoding='latin1', on_bad_lines='skip')
                except Exception:
                    try:
                        return pd.read_csv(path, sep=None, engine='python', encoding='latin1', on_bad_lines='skip')
                    except Exception as final_e:
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
    
    perfect_ings = []
    try:
        perfect_candidate = _resolve(os.path.join('Data', 'recept_alapanyagok_TÖKÉLETES.json'))
        if isinstance(perfect_candidate, list):
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

# ===== FASTING RECIPES =====
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

# ===== HÁLÓZATI VIZUALIZÁCIÓ =====
def create_network_graph(center_node, connected_nodes):
    if not center_node or not connected_nodes:
        return None

    G = nx.Graph()
    G.add_node(center_node, node_type='center')
    
    for n in connected_nodes:
        G.add_node(n["name"], degree=n["degree"], node_type=n.get("type", "unknown"))
        G.add_edge(center_node, n["name"], weight=n["degree"])

    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
    
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='rgba(204, 170, 119, 0.3)'),
                hoverinfo='none',
                showlegend=False
            )
        )

    node_colors = {
        'center': '#ccaa77',
        'Alapanyag': '#8b5a2b',
        'Molekula': '#808080',
        'Recept': '#800000',
        'unknown': '#999999'
    }

    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    max_degree = max([n["degree"] for n in connected_nodes], default=1)
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        if node == center_node:
            node_text.append(f"<b style='font-size: 14px'>{node}</b><br><i>(központi)</i>")
            node_size.append(40)
            node_color.append(node_colors['center'])
        else:
            degree = next((n["degree"] for n in connected_nodes if n["name"] == node), 1)
            node_type = next((n.get("type", "unknown") for n in connected_nodes if n["name"] == node), "unknown")
            node_text.append(f"<b>{node}</b><br>Degree: {degree}<br>Típus: {node_type}")
            node_size.append(15 + (degree / max_degree) * 30)
            node_color.append(node_colors.get(node_type, node_colors['unknown']))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hovertemplate='%{text}<extra></extra>',
        text=[n.split('<br>')[0].replace('<b style=\'font-size: 14px\'>', '').replace('</b>', '').replace('<b>', '') for n in node_text],
        textposition="top center",
        textfont=dict(size=10, family="Crimson Text", color='white'),
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white')
        ),
        customdata=node_text,
        showlegend=False
    )

    fig = go.Figure(data=edge_trace + [node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        template='plotly_dark',
        height=800
    )
    
    return fig

# ===============================
# NODE TÍPUS NORMALIZÁLÁS
# ===============================
type_mapping = {
    "dish": "Recept",
    "ingredient": "Alapanyag",
    "molecule": "Molekula"
}

type_column = next(
    (c for c in ["type", "Type", "Intervaltype"] if c in tripartit_df.columns),
    None
)

if type_column:
    tripartit_df["node_type"] = (
        tripartit_df[type_column]
        .map(type_mapping)
        .fillna("Egyéb")
    )
else:
    tripartit_df["node_type"] = "Egyéb"

all_nodes = tripartit_df.to_dict("records")
all_edges = edges_df.to_dict("records")
historical_recipes = historical_df.to_dict("records")

fasting_recipes = [r for r in historical_recipes if is_fasting_recipe(r)]
fasting_ratio = len(fasting_recipes) / max(len(historical_recipes), 1)

# ===============================
# UTILITY FUNCTIONS
# ===============================
def strip_icon_ligatures(s: str) -> str:
    """Eltávolítja a Material Icons ligature-öket és HTML entitásokat"""
    if not isinstance(s, str):
        return s
    
    # HTML entitások dekódolása
    s = _html.unescape(s)
    
    # HTML tagek eltávolítása
    s = re.sub(r"<[^>]+>", '', s)
    
    # Material Icons specifikus pattern-ek eltávolítása
    icon_patterns = [
        r'keyboard_arrow_right', r'keyboard_arrow_left', r'keyboard_arrow_up', 
        r'keyboard_arrow_down', r'arrow_right', r'arrow_left', r'arrow_forward',
        r'arrow_back', r'check_circle', r'check_box', r'radio_button',
        r'menu', r'close', r'settings', r'search', r'favorite', r'share',
        r'more_vert', r'more_horiz',
    ]
    
    for pattern in icon_patterns:
        s = re.sub(pattern, '', s, flags=re.IGNORECASE)
    
    # Általános pattern: bármilyen szó_szó vagy szó-szó ami icon lehet
    s = re.sub(r'\b[a-z]+_[a-z]+(_[a-z]+)?\b', '', s, flags=re.IGNORECASE)
    
    # Zero-width és control karakterek eltávolítása
    s = re.sub(r"[\u200B-\u200F\uFEFF\u0000-\u001F]", '', s)
    
    # Whitespace normalizálás
    s = re.sub(r"\s{2,}", ' ', s).strip()
    
    return s

def build_gpt_context(nodes, recipes, perfect_ings=None, user_query=None, max_nodes=120, max_recipes=40):
    grouped = {}
    for n in nodes:
        grouped.setdefault(n["node_type"], []).append(n)

    sampled_nodes = []
    if grouped:
        for group in grouped.values():
            sampled_nodes.extend(random.sample(
                group,
                min(len(group), max_nodes // len(grouped))
            ))
    else:
        sampled_nodes = nodes[:max_nodes]

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

        q_norm = _normalize(user_query)
        q_tokens = [t for t in q_norm.split() if len(t) > 1]
        if q_tokens:
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

    simplified_nodes = [
        {
            "name": n["Label"],
            "type": n["node_type"],
            "degree": int(n.get("Degree", 0))
        }
        for n in sampled_nodes
    ]

    sampled_recipes = random.sample(
        recipes,
        min(len(recipes), max_recipes)
    )

    simplified_recipes = [
        {
            "title": r.get("title", ""),
            "excerpt": r.get("original_text", "")[:150]
        }
        for r in sampled_recipes
    ]

    return simplified_nodes, simplified_recipes

def gpt_search_recipes(user_query):
    nodes_ctx, recipes_ctx = build_gpt_context(
        all_nodes,
        historical_recipes,
        perfect_ings,
        user_query=user_query
    )

    system_prompt = """
Te egy XVII. századi magyar gasztronómia szakértő asszisztens vagy.

Feladat:
- a felhasználó leírása alapján válaszd ki
  - max 5 releváns node-ot
  - max 3 releváns történeti receptet

Fontosabb alapanyagok és fűszerek a XVII. századi magyar konyhaművészetben:
szerecsendió, szerecsendió-virág, sáfrány, fahéj, gyömbér, szegfűszeg,
bors, só, cukor, méz, ecet, olaj, vaj, tejföl, hús, hal, baromfi,
gabonanemű, lencse, borsó, bab, káposzta, répa, hagyma, fokhagyma.

Ha a felhasználó keresésében ilyen alapanyag vagy fűszer szerepel,
azokat részesítsd előnyben a node-választásnál.

Válasz KIZÁRÓLAG JSON:
{
  "suggested_nodes": [string],
  "suggested_recipes": [string],
  "reasoning": string
}
"""

    user_prompt = f"""
Felhasználó keresése: "{user_query}"

Elérhető node-ok:
{json.dumps(nodes_ctx, ensure_ascii=False)}

Történeti receptek:
{json.dumps(recipes_ctx, ensure_ascii=False)}
"""

    try:
        full_labels = sorted({n.get("Label", "") for n in all_nodes})
        full_labels_preview = json.dumps(full_labels, ensure_ascii=False)
    except Exception:
        full_labels_preview = "[]"

    user_prompt += f"\nTeljes csomópontlista (labels):\n{full_labels_preview}\n"

    try:
        perfect_preview = (
            json.dumps(perfect_ings[:50], ensure_ascii=False)
            if isinstance(perfect_ings, list)
            else json.dumps(perfect_ings, ensure_ascii=False)
        )
    except Exception:
        perfect_preview = "[]"

    user_prompt += f"\nTökéletes alapanyaglista (rövid):\n{perfect_preview}\n"

    # 🔑 HELYES GPT-5-NANO HÍVÁS (Responses API)
    response = client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_output_tokens=900
    )

    try:
        result = json.loads(response.output_text)
    except Exception:
        return {
            "suggested_nodes": [],
            "suggested_recipes": [],
            "reasoning": "Az AI válasza nem volt értelmezhető JSON formátumban."
        }

    return result

# ===============================
# AI RECIPE GENERATOR
# ===============================
def generate_ai_recipe(selected, connected, historical):
    system_prompt = """
Te egy XVII. századi magyar szakácskönyv stílusában írsz receptet.

SZABÁLYOK:
- 70–110 szó
- archaikus, régies magyar nyelvezet
- CSAK a kapott kapcsolatokból dolgozz
- ne használj modern alapanyagokat, csak azokat, amiket az adatbázisban találsz
- **Válasz KIZÁRÓLAG JSON formátumban**, tartalmazza pontosan ezt a szerkezetet:

{
  "title": "",
  "archaic_recipe": "",
  "confidence": "low|medium|high"
}
Csak a JSON‑választ add vissza, semmi egyebet!
"""

    user_prompt = f"""
Központi alapanyag:
{selected}

Kapcsolódó alapanyagok:
{json.dumps(connected, ensure_ascii=False)}

Történeti példák:
{json.dumps(historical, ensure_ascii=False)}
"""

    try:
        response = client.responses.create(
            model="gpt-5.1",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_output_tokens=700
        )
        ai_text = response.output_text.strip()

        # Remove markdown code blocks if present
        if ai_text.startswith("```json"):
            ai_text = ai_text[7:]
        if ai_text.startswith("```"):
            ai_text = ai_text[3:]
        if ai_text.endswith("```"):
            ai_text = ai_text[:-3]
        ai_text = ai_text.strip()

        # Parse JSON
        result = json.loads(ai_text)

        # Calculate word count and confidence
        wc = len(result.get("archaic_recipe", "").split())
        result["word_count"] = wc

        if 70 <= wc <= 110:
            result["confidence"] = "high"
        elif 50 <= wc <= 130:
            result["confidence"] = "medium"
        else:
            result["confidence"] = "low"

        return result

    except Exception as e:
        return {
            "title": "Hiba történt",
            "archaic_recipe": f"A recept generálása sikertelen volt: {str(e)}",
            "confidence": "low",
            "word_count": 0,
            "raw_text": ai_text if 'ai_text' in locals() else ""
        }


# ===== HERO SECTION =====
import base64
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

# ===== METRIKÁK =====
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Csomópontok (Nodes)", f"{len(all_nodes)}")
with col2:
    st.metric("Kapcsolatok", f"{len(all_edges)}")
with col3:
    st.metric("Receptek", f"{len(historical_recipes)}")
with col4:
    st.metric("Átlag Degree", f"{tripartit_df['Degree'].mean():.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# ===== KATEGÓRIA VÁLASZTÓ =====
category = st.radio(
    "Kategória",
    ["🌐 Összes", "⚗️ Molekulák", "🥘 Alapanyagok", "📖 Receptek"],
    horizontal=True,
    label_visibility="collapsed"
)

node_type_filter = {
    "🌐 Összes": ['Alapanyag', 'Molekula', 'Recept'],
    "⚗️ Molekulák": ['Molekula'],
    "🥘 Alapanyagok": ['Alapanyag'],
    "📖 Receptek": ['Recept']
}[category]

# ===== KERESÉS ÉS SZŰRÉS =====
st.markdown("""
<div style="background: linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%); border: 3px solid #ccaa77; border-radius: 12px; padding: 2rem; margin: 2rem 0; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);">
    <h3 style="color: #ccaa77; font-family: 'Cinzel', serif; margin-bottom: 1rem; text-align: center;">
        🔍 Intelligens Keresés
    </h3>
    <p style="text-align: center; color: #e8dcc8; font-family: 'Crimson Text', serif; margin-bottom: 1.5rem;">
        Írj le egy ételt vagy alapanyagot, és az AI megtalálja a kapcsolódó node-okat és történeti recepteket!
    </p>
</div>
""", unsafe_allow_html=True)

col_search, col_sort = st.columns([3, 1])
with col_search:
    query = st.text_input("Keresés", placeholder="🔍 pl. 'valami fűszeres hal', 'édes sütemény mandulával', 'boros leves'...", key="search_input", label_visibility="collapsed")
    
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
                    top_name = strip_icon_ligatures(str(suggested[0]))
                    node_obj = next((n for n in all_nodes if strip_icon_ligatures(n.get("Label", "")).lower() == top_name.lower()), None)
                    if node_obj:
                        sel = node_obj["Label"]
                        related = [e["Target"] if e["Source"] == sel else e["Source"] for e in all_edges if sel in [e["Source"], e["Target"]]]
                        connected = [{"name": x["Label"], "degree": x.get("Degree", 1), "type": x.get("node_type", "unknown")} for x in all_nodes if x["Label"] in related]
                        historical_recipe = [{"title": strip_icon_ligatures(r.get("title", "Névtelen")), "text": strip_icon_ligatures(r.get("original_text", "")[:300])} for r in historical_recipes if sel.lower() in str(r).lower()][:5]

                        st.session_state["selected"] = sel
                        st.session_state["connected"] = connected
                        st.session_state["historical_recipe"] = historical_recipe

                        with st.spinner("⏳ AI receptgenerálás..."):
                            ai_recipe = generate_ai_recipe(sel, connected, historical_recipe)
                            st.session_state["ai_recipe"] = ai_recipe
            except Exception:
                pass
        
with col_sort:
    sort_by = st.selectbox(
        "Rendezés",
        [
            "📝 Név (A–Z)",
            "🔁 Név (Z–A)",
            "📊 Degree ↓",
            "📈 Degree ↑"
        ],
        key="sort_select",
        label_visibility="collapsed"
    )

# GPT keresési eredmények megjelenítése
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
            node = next((n for n in all_nodes if strip_icon_ligatures(n.get("Label", "")).lower() == clean_node_name.lower()), None)
            if node and i < len(cols_suggested):
                type_emoji = {'Alapanyag': '🥘', 'Molekula': '⚗️', 'Recept': '📖', 'Egyéb': '⚪'}.get(node.get('node_type'), '⚪')
                clean_label = strip_icon_ligatures(node['Label'])
                if cols_suggested[i].button(f"{type_emoji} {clean_label}", key=f"suggested_{i}"):
                    sel = node["Label"]
                    related = [e["Target"] if e["Source"] == sel else e["Source"] for e in all_edges if sel in [e["Source"], e["Target"]]]
                    connected = [{"name": x["Label"], "degree": x.get("Degree", 1), "type": x.get("node_type", "unknown")} for x in all_nodes if x["Label"] in related]
                    historical_recipe = [{"title": strip_icon_ligatures(r.get("title", "Névtelen")), "text": strip_icon_ligatures(r.get("original_text", "")[:300])} for r in historical_recipes if sel.lower() in str(r).lower()][:5]

                    st.session_state["selected"] = sel
                    st.session_state["connected"] = connected
                    st.session_state["historical_recipe"] = historical_recipe

                    with st.spinner("⏳ AI receptgenerálás..."):
                        ai_recipe = generate_ai_recipe(sel, connected, historical_recipe)
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

# Node szűrés
if "gpt_search_results" not in st.session_state or not query:
    filtered_nodes = [
        n for n in all_nodes
        if (not query or query.lower() in n["Label"].lower())
        and n.get("node_type") in node_type_filter
    ]
else:
    suggested = st.session_state["gpt_search_results"].get("suggested_nodes", [])
    filtered_nodes = [
        n for n in all_nodes
        if n.get("node_type") in node_type_filter and (
            not query or 
            query.lower() in n["Label"].lower() or 
            n["Label"] in suggested
        )
    ]

if sort_by == "📊 Degree ↓":
    filtered_nodes = sorted(filtered_nodes, key=lambda x: x.get("Degree", 0), reverse=True)
elif sort_by == "📈 Degree ↑":
    filtered_nodes = sorted(filtered_nodes, key=lambda x: x.get("Degree", 0))
elif sort_by == "🔁 Név (Z–A)":
    filtered_nodes = sorted(filtered_nodes, key=lambda x: x["Label"], reverse=True)
else:
    filtered_nodes = sorted(filtered_nodes, key=lambda x: x["Label"])

st.markdown(f"<h3 style='text-align: center; color: white; font-family: Cinzel, serif; font-weight: 700; margin: 2rem 0 1.5rem 0;'>Elérhető csomópontok (nodes) ({len(filtered_nodes)} db)</h3>", unsafe_allow_html=True)

# ===== NODE GOMBOK =====
cols = st.columns(6)
for i, n in enumerate(filtered_nodes[:60]):
    type_emoji = {'Alapanyag': '🥘', 'Molekula': '⚗️', 'Recept': '📖', 'Egyéb': '⚪'}.get(n.get('node_type'), '⚪')
    clean_label = strip_icon_ligatures(n['Label'])
    if cols[i % 6].button(f"{type_emoji} {clean_label}", key=f"node_{i}"):
        sel = n["Label"]
        
        related = [
            e["Target"] if e["Source"] == sel else e["Source"]
            for e in all_edges if sel in [e["Source"], e["Target"]]
        ]
        
        connected = [
            {"name": x["Label"], "degree": x.get("Degree", 1), "type": x.get("node_type", "unknown")}
            for x in all_nodes if x["Label"] in related
        ]
        
        historical_recipe = [
            {"title": strip_icon_ligatures(r.get("title", "Névtelen")), "text": strip_icon_ligatures(r.get("original_text", "")[:300])}
            for r in historical_recipes if sel.lower() in str(r).lower()
        ][:5]
        
        st.session_state["selected"] = sel
        st.session_state["connected"] = connected
        st.session_state["historical_recipe"] = historical_recipe
        
        with st.spinner("⏳ AI receptgenerálás..."):
            ai_recipe = generate_ai_recipe(sel, connected, historical_recipe)
            st.session_state["ai_recipe"] = ai_recipe
        
        st.rerun()

# ===== EREDMÉNYEK =====
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
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("❌ Hiba történt a generálás során")

# ===== NAVIGÁCIÓS TÁJÉKOZTATÓ =====
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
        st.switch_page("Pages/About.py")

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
    
    if st.button("📊 Tovább az Analitika oldalra", key="nav_analytics", use_container_width=True):
        st.switch_page("Pages/analytics.py")

st.markdown("""
<p style="text-align: center; color: #888; font-size: 0.9rem; margin-top: 1.5rem;">
    💡 <em>Vagy használd a bal felső sarokban lévő menüt (☰) a navigáláshoz!</em>
</p>
""", unsafe_allow_html=True)

# ===== FOOTER =====
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
        © 2025 • Digitális bölcsészeti-, társadalom- és hálózattudományi projekt
    </p>
</div>
"""), unsafe_allow_html=True)







