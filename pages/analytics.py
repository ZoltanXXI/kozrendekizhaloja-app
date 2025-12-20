import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import numpy as np
from scipy import stats
import os

st.set_page_config(page_title="Statisztika", page_icon="📊", layout="wide")

# Top anchor for scroll-to-top functionality
st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)

# ===== CUSTOM CSS - TÖRTÉNELMI STÍLUS (konzolidált) =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700;900&family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap');

/* === ÁLTALÁNOS BEÁLLÍTÁSOK === */
body, .block-container {
    font-family: 'Cormorant Garamond', serif !important;
}

/* === CÍMSOROK === */
h1, h2, h3, h4 {
    font-family: 'Cinzel', serif !important;
    color: #2c1810 !important;
    margin: 0.2rem 0 !important;
}

h1 { font-size: 2.5rem !important; font-weight: 900 !important; }
h2 { font-size: 2rem !important; font-weight: 700 !important; }
h3 { font-size: 1.5rem !important; font-weight: 700 !important; }
h4 { font-size: 1.25rem !important; font-weight: 600 !important; }

/* === BEKEZDÉSEK ÉS SZÖVEGEK === */
p, div, span, li {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.05rem !important;
    line-height: 1.6 !important;
}

/* === KIEMELT SZÖVEGEK === */
strong, b {
    font-family: 'Playfair Display', serif !important;
    font-weight: 700 !important;
}

em, i {
    font-family: 'Playfair Display', serif !important;
    font-style: italic !important;
}

/* === SIDEBAR === */
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
.span[data-testid="stIconMaterial"] { display: none !important; }

/* === GOMBOK === */
button, .stButton > button { font-family: 'Cinzel', serif !important; font-weight: 600 !important; }

/* === METRIKÁK === */
[data-testid="stMetricValue"] { font-family: 'Playfair Display', serif !important; font-weight: 700 !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { font-family: 'Cinzel', serif !important; font-size: 0.9rem !important; }

/* === TAB-ok === */
.stTabs [data-baseweb="tab-list"] { gap: 1rem; }
.stTabs [data-baseweb="tab"] {
    background: #fffbf0;
    border: 2px solid #d4af37;
    border-radius: 8px 8px 0 0;
    color: #2c1810;
    font-family: 'Cinzel', serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
}
.stTabs [aria-selected="true"] { background: linear-gradient(to bottom, #d4af37, #b8941f); color: white; }

/* === OLDAL HÁTTEREK === */
.main, .block-container {
    background: linear-gradient(to bottom, #5c1a1a, #7b1f1f);
    color: #2c1810;
}

/* Large centered quote */
.large-quote { font-family: 'Cinzel', serif; font-size: 1.8rem; color: #f7efe1; text-align: center; margin: 2rem auto; max-width: 1200px; line-height: 1.2; font-weight:700; }

/* Prevent overscroll */
body { overscroll-behavior: none; }

/* Hide Streamlit's default footer */
footer { visibility: hidden; }

/* Scroll-to-top button style (anchor) */
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

# ===== ADATOK BETÖLTÉSE =====
@st.cache_data
def load_analytics_data():
    base = os.getcwd()
    tripartit_path = os.path.join(base, "data", "Recept_halo__molekula_tripartit.csv")
    edges_path = os.path.join(base, "data", "recept_halo_edges.csv")
    hist_path = os.path.join(base, "data", "HistoricalRecipe_export.csv")
    tripartit_df = pd.read_csv(tripartit_path, delimiter=";", encoding="utf-8", on_bad_lines='skip')
    edges_df = pd.read_csv(edges_path, encoding="utf-8", on_bad_lines='skip')
    historical_df = pd.read_csv(hist_path, encoding="utf-8", on_bad_lines='skip')
    return tripartit_df, edges_df, historical_df

tripartit_df, edges_df, historical_df = load_analytics_data()

# Biztonsági fallback oszlopok
if 'original_text' not in historical_df.columns and 'text' in historical_df.columns:
    historical_df['original_text'] = historical_df['text'].fillna('')
if 'original_text' not in historical_df.columns:
    historical_df['original_text'] = historical_df.get('description', '').fillna('')

historical_df['original_text'] = historical_df['original_text'].fillna('')
historical_df['word_count'] = historical_df['original_text'].apply(lambda x: len(str(x).split()))

# Degree fallback: ha nincs 'Degree' oszlop, próbáljuk számolni az edges alapján (gyenge közelítés)
if 'Degree' not in tripartit_df.columns:
    if {'source', 'target'}.issubset(edges_df.columns):
        nodes = pd.concat([edges_df['source'].astype(str), edges_df['target'].astype(str)])
        deg_counts = nodes.value_counts().to_dict()
        tripartit_df['Degree'] = tripartit_df['Label'].map(lambda l: deg_counts.get(str(l), 0))
    else:
        tripartit_df['Degree'] = 0

# ===== NODE TÍPUS MAPPING =====
type_mapping = {'dish': 'Recept', 'molecule': 'Molekula', 'ingredient': 'Alapanyag'}
type_column = None
for col in ['type', 'Type', 'Intervaltype', 'intervaltype', 'node_type', 'category']:
    if col in tripartit_df.columns:
        type_column = col
        break

if type_column:
    tripartit_df['node_type'] = tripartit_df[type_column].map(type_mapping).fillna(tripartit_df[type_column])
else:
    tripartit_df['node_type'] = 'Egyéb'

# ===== STATISZTIKAI ELOSZLÁS ELEMZÉS =====
def analyze_distribution(data):
    mean = float(np.mean(data)) if len(data) > 0 else 0.0
    median = float(np.median(data)) if len(data) > 0 else 0.0
    std = float(np.std(data)) if len(data) > 0 else 0.0
    skewness = float(stats.skew(data)) if len(data) > 2 else 0.0
    kurtosis = float(stats.kurtosis(data)) if len(data) > 2 else 0.0

    shapiro_p = None
    if len(data) >= 3 and len(data) < 5000:
        try:
            shapiro_stat, shapiro_p = stats.shapiro(data)
        except Exception:
            shapiro_p = None
    else:
        shapiro_p = None

    distributions = {'norm': 'Normális', 'lognorm': 'Lognormális', 'expon': 'Exponenciális', 'gamma': 'Gamma', 'weibull_min': 'Weibull'}
    best_fit = None
    best_ks_stat = float('inf')
    for dist_name, dist_label in distributions.items():
        try:
            params = getattr(stats, dist_name).fit(data)
            ks_stat, ks_p = stats.kstest(data, dist_name, args=params)
            if ks_stat < best_ks_stat:
                best_ks_stat = ks_stat
                best_fit = dist_label
        except Exception:
            continue

    if shapiro_p and shapiro_p > 0.05:
        dist_type = "✅ **Normális eloszlás**"
        explanation = "Az adatok normális eloszlást követnek (Shapiro-Wilk p > 0.05)"
    elif skewness > 1:
        dist_type = "📈 **Jobbra ferde eloszlás**"
        explanation = f"Erősen jobbra ferde (ferdeség: {skewness:.2f})."
    elif skewness > 0.5:
        dist_type = "📊 **Mérsékelten jobbra ferde**"
        explanation = f"Jobbra ferde (ferdeség: {skewness:.2f})."
    elif skewness < -1:
        dist_type = "📉 **Balra ferde eloszlás**"
        explanation = f"Erősen balra ferde (ferdeség: {skewness:.2f})."
    elif abs(skewness) < 0.5:
        dist_type = "⚖️ **Szimmetrikus eloszlás**"
        explanation = f"Közel szimmetrikus (ferdeség: {skewness:.2f})"
    else:
        dist_type = "📊 **Aszimmetrikus eloszlás**"
        explanation = f"Aszimmetrikus eloszlás (ferdeség: {skewness:.2f})"

    if kurtosis > 3:
        kurtosis_type = "🔺 Leptokurtikus (csúcsos)"
    elif kurtosis < -3:
        kurtosis_type = "🔻 Platykurtikus (lapos)"
    else:
        kurtosis_type = "⚫ Mezokurtikus (normális csúcsosság)"

    return {
        'type': dist_type,
        'explanation': explanation,
        'best_fit': best_fit,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'kurtosis_type': kurtosis_type,
        'shapiro_p': shapiro_p,
        'mean': mean,
        'median': median,
        'std': std
    }

# ===== HEADER/INTRO =====
st.markdown("""
<div style="
    background: linear-gradient(135deg, #5c070d, #840A13);
    padding: 20px 40px;
    border-radius: 50px;
    text-align: center;
    color: #fff8e0;
    font-family: 'Cinzel', serif;
    font-size: 1.5rem;
    font-weight: bold;
    box-shadow: 2px 4px 12px rgba(0,0,0,0.4);
    margin-bottom: 1.5rem;
">
📊 Receptadatok Mélyelemzése<br>
<span style='font-size:1rem; font-style:italic; color:#f5e1d1;'>Hálózati statisztikák, recept hosszúság eloszlás és AI generálási stratégiák</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="width: 150px; height: 3px; background: linear-gradient(to right, #d4af37, #f0d98d, #d4af37); 
            margin: 0 auto 2rem auto; border-radius: 2px;"></div>
""", unsafe_allow_html=True)

# ===== TAB-ok =====
tab1, tab2, tab3, tab4 = st.tabs([
    "🕸️ Hálózati Elemzés",
    "📏 Recept Hosszúság",
    "🤖 AI Stratégiák",
    "📚 Történeti Korpusz"
])

# ===== TAB 1: HÁLÓZATI ELEMZÉS =====
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔢 Alapstatisztikák")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        metrics_col1.metric("Node-ok száma", len(tripartit_df))
        metrics_col2.metric("Kapcsolatok száma", len(edges_df))
        metrics_col3.metric("Átlagos degree", round(tripartit_df['Degree'].mean(), 2) if 'Degree' in tripartit_df.columns else 0)

        st.markdown("### 📊 Degree Eloszlás")
        fig_degree = go.Figure()
        fig_degree.add_trace(go.Histogram(
            x=tripartit_df['Degree'],
            nbinsx=30,
            marker_color='#8b5a2b',
            name='Degree',
            opacity=0.7,
            histnorm='probability density'
        ))
        degree_data = tripartit_df['Degree'].values
        if len(degree_data) > 0:
            x_range = np.linspace(degree_data.min(), degree_data.max(), 100)
            mu, sigma = degree_data.mean(), degree_data.std()
            normal_curve = stats.norm.pdf(x_range, mu, sigma)
            fig_degree.add_trace(go.Scatter(x=x_range, y=normal_curve, mode='lines', name='Normális illesztés', line=dict(color='red', width=2, dash='dash')))
            try:
                shape, loc, scale = stats.lognorm.fit(degree_data, floc=0)
                lognorm_curve = stats.lognorm.pdf(x_range, shape, loc, scale)
                fig_degree.add_trace(go.Scatter(x=x_range, y=lognorm_curve, mode='lines', name='Lognormális illesztés', line=dict(color='green', width=3)))
            except Exception:
                pass

        fig_degree.update_layout(xaxis_title="Degree", yaxis_title="Sűrűség", paper_bgcolor='#fcf5e5', plot_bgcolor='#fcf5e5', height=400, showlegend=True)
        st.plotly_chart(fig_degree, use_container_width=True)

        st.markdown("### 📈 Degree Eloszlás Elemzése")
        degree_analysis = analyze_distribution(tripartit_df['Degree'].values if 'Degree' in tripartit_df.columns else np.array([]))
        st.info(f"""
        **{degree_analysis['type']}**
        
        {degree_analysis['explanation']}
        
        - **Legjobb illeszkedés:** {degree_analysis['best_fit']}
        - **Ferdeség (skewness):** {degree_analysis['skewness']:.3f}
        - **Csúcsosság (kurtosis):** {degree_analysis['kurtosis']:.3f} — {degree_analysis['kurtosis_type']}
        """)

    with col2:
        st.markdown("### 🎨 Node Típusok")
        type_counts = tripartit_df['node_type'].value_counts()
        st.markdown("#### 📊 Típus Eloszlás")
        for node_type, count in type_counts.items():
            percent = (count / len(tripartit_df)) * 100 if len(tripartit_df) > 0 else 0
            emoji = {'Alapanyag': '🥘', 'Molekula': '⚗️', 'Recept': '📖', 'Egyéb': '⚪'}.get(node_type, '⚪')
            st.markdown(f"{emoji} **{node_type}:** {count} db ({percent:.1f}%)")

        fig_types = go.Figure(data=[go.Pie(labels=type_counts.index, values=type_counts.values, marker=dict(colors=['#8b5a2b', '#4a7c59', '#b85450', '#cccccc']), hole=0.4)])
        fig_types.update_layout(paper_bgcolor='#fcf5e5', height=350)
        st.plotly_chart(fig_types, use_container_width=True)

        st.markdown("### 🏆 Top 10 Node (degree szerint)")
        if 'Degree' in tripartit_df.columns:
            top_nodes = tripartit_df.nlargest(10, 'Degree')[['Label', 'Degree', 'node_type']]
            for idx, row in top_nodes.iterrows():
                emoji = {'Alapanyag': '🥘', 'Molekula': '⚗️', 'Recept': '📖', 'Egyéb': '⚪'}.get(row['node_type'], '⚪')
                st.markdown(f"{emoji} **{row['Label']}** - Degree: {row['Degree']}")
        else:
            st.markdown("Nincs Degree adat a fájlban.")

# ===== TAB 2: RECEPT HOSSZÚSÁG =====
with tab2:
    st.markdown("### 📏 Recept Hosszúság Statisztikák")
    stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
    stats_col1.metric("Receptek száma", len(historical_df))
    stats_col2.metric("Átlag szó", round(historical_df['word_count'].mean(), 1) if len(historical_df) > 0 else 0)
    stats_col3.metric("Medián szó", int(historical_df['word_count'].median()) if len(historical_df) > 0 else 0)
    stats_col4.metric("Min szó", int(historical_df['word_count'].min()) if len(historical_df) > 0 else 0)
    stats_col5.metric("Max szó", int(historical_df['word_count'].max()) if len(historical_df) > 0 else 0)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Szószám Eloszlás Histogram")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=historical_df['word_count'], nbinsx=40, marker_color='#8b5a2b', name='Szószám', opacity=0.7, histnorm='probability density'))
        word_data = historical_df['word_count'].values if len(historical_df) > 0 else np.array([0])
        x_range = np.linspace(word_data.min(), word_data.max(), 200) if word_data.max() > word_data.min() else np.linspace(0, max(1, word_data.max()), 2)
        mu, sigma = word_data.mean(), word_data.std()
        normal_curve = stats.norm.pdf(x_range, mu, sigma)
        fig_hist.add_trace(go.Scatter(x=x_range, y=normal_curve, mode='lines', name='Normális (elméleti)', line=dict(color='red', width=2, dash='dash')))
        try:
            shape, loc, scale = stats.lognorm.fit(word_data, floc=0)
            lognorm_curve = stats.lognorm.pdf(x_range, shape, loc, scale)
            fig_hist.add_trace(go.Scatter(x=x_range, y=lognorm_curve, mode='lines', name='Lognormális (illesztett)', line=dict(color='green', width=3)))
        except Exception:
            pass
        fig_hist.add_vline(x=historical_df['word_count'].mean(), line_dash="dash", line_color="darkred", annotation_text=f"Átlag: {historical_df['word_count'].mean():.1f}", annotation_position="top")
        fig_hist.add_vline(x=historical_df['word_count'].median(), line_dash="dash", line_color="darkblue", annotation_text=f"Medián: {historical_df['word_count'].median():.0f}", annotation_position="top")
        fig_hist.update_layout(xaxis_title="Szószám", yaxis_title="Sűrűség", paper_bgcolor='#fcf5e5', plot_bgcolor='#fcf5e5', height=450, showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("### 📈 Eloszlás Elemzése")
        word_analysis = analyze_distribution(historical_df['word_count'].values)
        st.success(f"""
        **{word_analysis['type']}**
        
        {word_analysis['explanation']}
        
        **Statisztikai jellemzők:**
        - **Legjobb illeszkedés:** {word_analysis['best_fit']}
        - **Ferdeség (skewness):** {word_analysis['skewness']:.3f}
        - **Csúcsosság (kurtosis):** {word_analysis['kurtosis']:.3f} — {word_analysis['kurtosis_type']}
        - **Szórás:** {word_analysis['std']:.1f} szó
        """)

        if word_analysis['skewness'] > 0.5:
            st.warning("""
            ⚠️ **Jobbra ferde eloszlás magyarázat:**
            - A legtöbb recept rövid (30-70 szó)
            - Van néhány extrém hosszú recept (200+ szó)
            - Átlag > Medián (outlierek hatása)
            """)

        st.markdown("#### 🎨 Eloszlás Vizualizáció")
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            st.markdown("""
            **🔴 Piros vonal (Normális):** Szimmetrikus haranggörbe
            - Átlag = Medián
            """)
        with col_viz2:
            st.markdown("""
            **🟢 Zöld vonal (Lognormális):** Jobbra ferde
            - Átlag > Medián
            """)

    with col2:
        st.markdown("### 🥧 Hosszúság Kategóriák")
        def categorize_length(word_count):
            if word_count <= 30:
                return 'Nagyon rövid (≤30)'
            elif word_count <= 60:
                return 'Rövid (31-60)'
            elif word_count <= 100:
                return 'Közepes (61-100)'
            elif word_count <= 200:
                return 'Hosszú (101-200)'
            else:
                return 'Nagyon hosszú (>200)'
        historical_df['length_category'] = historical_df['word_count'].apply(categorize_length)
        category_counts = historical_df['length_category'].value_counts()
        colors = {'Nagyon rövid (≤30)': '#8b5a2b', 'Rövid (31-60)': '#a67c52', 'Közepes (61-100)': '#c9a877', 'Hosszú (101-200)': '#dcc5a0', 'Nagyon hosszú (>200)': '#f0e5d3'}
        fig_pie = go.Figure(data=[go.Pie(labels=category_counts.index, values=category_counts.values, marker=dict(colors=[colors[cat] for cat in category_counts.index]), textinfo='label+percent', textposition='outside')])
        fig_pie.update_layout(paper_bgcolor='#fcf5e5', height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("#### 📋 Részletes Eloszlás")
        for cat, count in category_counts.items():
            percent = (count / len(historical_df)) * 100 if len(historical_df) > 0 else 0
            st.markdown(f"**{cat}:** {count} db ({percent:.1f}%)")

# ===== TAB 3: AI STRATÉGIÁK =====
with tab3:
    st.markdown("### 🤖 AI Generálási Stratégiák (GPT-5.1 & GPT-5-mini Best Practices)")
    strategies = [
        {"mode": "minimal", "trigger": "0 példa VAGY átlag degree < 3", "word_target": "max 40 szó", "style": "Emlékeztető stílus, minimális kontextus", "use_case": "Gyenge hálózati alap, nincs történeti példa", "prompt_key": "Ultra-concise, grounding warnings", "color": "#b85450"},
        {"mode": "concise", "trigger": "1-2 példa", "word_target": "40-70 szó", "style": "Lakonikus, tapasztalt szakács stílus", "use_case": "Kevés példa, közepes kapcsolódás", "prompt_key": "Terse instructions, assumed knowledge", "color": "#8b5a2b"},
        {"mode": "standard", "trigger": "3-5 példa", "word_target": "70-110 szó", "style": "Klasszikus 17. századi recept forma", "use_case": "Közepes példatár, erős hálózat", "prompt_key": "Complete but compact, contextual", "color": "#4a7c59"},
        {"mode": "detailed", "trigger": "6+ példa", "word_target": "110-160 szó", "style": "Részletes technológiai leírás kontextussal", "use_case": "Gazdag forrás, kiváló kapcsolódás", "prompt_key": "Step-by-step, timing, cultural context", "color": "#d4af37"}
    ]
    for strategy in strategies:
        with st.expander(f"**{strategy['mode'].upper()}** - {strategy['word_target']}", expanded=False):
            col1, col2 = st.columns([3,1])
            with col1:
                st.markdown(f"**Trigger feltétel:** {strategy['trigger']}")
                st.markdown(f"**Stílus:** {strategy['style']}")
                st.markdown(f"**Használat:** {strategy['use_case']}")
                st.markdown(f"**Prompt kulcselem:** `{strategy['prompt_key']}`")
            with col2:
                st.markdown(f"<div style='background-color: {strategy['color']}; color: white; padding: 14px; border-radius: 10px; text-align: center;'><h3 style='margin:0;'> {strategy['word_target']} </h3></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📖 GPT-5-mini Prompt Engineering Principles")
    principles = [
        ("🎯 Grounding & Accuracy", "SOHA ne találj ki adatokat - csak hálózati kapcsolatok alapján"),
        ("📏 Verbosity Control", "Adaptív hosszúság: példaszám + hálózati erősség alapján"),
        ("🔍 Network-Informed", "Degree-súlyozott döntések (magas degree = erős párosítás)"),
        ("⚠️ Uncertainty Handling", "Explicit confidence score (low/medium/high)"),
        ("✅ Self-Check", "Generálás előtti validáció: kapcsolatok, források, hossz"),
        ("📊 Structured Output", "JSON schema strict enforcement")
    ]
    for title, desc in principles:
        st.markdown(f"**{title}:** {desc}")

# ===== TAB 4: TÖRTÉNETI KORPUSZ =====
with tab4:
    st.markdown("### 📚 Történeti Receptek Böngészése")
    search = st.text_input("🔍 Keresés a receptekben", placeholder="Pl. hal, bors, leves...")
    col1, col2 = st.columns([1, 3])
    with col1:
        length_filter = st.selectbox("Hosszúság szűrő", ["Összes", 'Nagyon rövid (≤30)', 'Rövid (31-60)', 'Közepes (61-100)', 'Hosszú (101-200)', 'Nagyon hosszú (>200)'])
    filtered_df = historical_df.copy()
    if search:
        filtered_df = filtered_df[filtered_df['title'].fillna('').str.contains(search, case=False) | filtered_df['original_text'].fillna('').str.contains(search, case=False)]
    if length_filter != "Összes":
        filtered_df = filtered_df[filtered_df['length_category'] == length_filter]
    st.markdown(f"**Találatok:** {len(filtered_df)} recept")
    for idx, row in filtered_df.head(30).iterrows():
        title = row.get('title', 'Névtelen')
        source = row.get('source', 'Ismeretlen forrás')
        wc = row.get('word_count', 0)
        with st.expander(f"📖 {title} ({wc} szó) - {source}"):
            st.markdown(f"**Kategória:** {row.get('length_category', '—')}")
            st.markdown("---")
            text = str(row.get('original_text', ''))
            st.markdown(text if len(text) <= 1000 else text[:1000] + "...")

# ===== FOOTER / METRIKA BLOKK FELETT =====
st.markdown("""
<div style="display:flex; gap:1rem; justify-content:center; flex-wrap:wrap; margin:1.25rem 0;">
    <div style="min-width:160px; text-align:center; background:#fffbf0; border:2px solid #d4af37; padding:1rem; border-radius:10px;">
        <div style="font-size:2rem; font-weight:700; color:#8b5a2b;">{n_hist}</div>
        <div style="color:#4a3728; margin-top:0.3rem;">Történeti receptek</div>
    </div>
    <div style="min-width:160px; text-align:center; background:#fffbf0; border:2px solid #d4af37; padding:1rem; border-radius:10px;">
        <div style="font-size:2rem; font-weight:700; color:#8b5a2b;">{n_nodes}</div>
        <div style="color:#4a3728; margin-top:0.3rem;">Node (hálózat)</div>
    </div>
    <div style="min-width:160px; text-align:center; background:#fffbf0; border:2px solid #d4af37; padding:1rem; border-radius:10px;">
        <div style="font-size:2rem; font-weight:700; color:#8b5a2b;">{avg_words}</div>
        <div style="color:#4a3728; margin-top:0.3rem;">Átlag szószám (recept szövegtest)</div>
    </div>
    <div style="min-width:160px; text-align:center; background:#fffbf0; border:2px solid #d4af37; padding:1rem; border-radius:10px;">
        <div style="font-size:2rem; font-weight:700; color:#8b5a2b;">{fast_pct}</div>
        <div style="color:#4a3728; margin-top:0.3rem;">Böjti receptek (detektálva)</div>
    </div>
</div>
""".format(
    n_hist=len(historical_df),
    n_nodes=len(tripartit_df),
    avg_words=round(historical_df['word_count'].mean(), 1) if len(historical_df) > 0 else 0.0,
    fast_pct="--"
), unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #f3e8dd;'>"
    f"Analytics Dashboard © 2025 | Korpusz: {len(historical_df)} recept, átlag {historical_df['word_count'].mean():.1f} szó"
    "</div>",
    unsafe_allow_html=True
)

# --- Scroll-to-top: anchor alapú (nincs iframe) ---
st.markdown("""
<a href="#top-anchor" class="scroll-to-top" aria-label="Vissza a tetejére">↑</a>
""", unsafe_allow_html=True)
