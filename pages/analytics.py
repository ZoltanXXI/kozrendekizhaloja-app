import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import numpy as np
from scipy import stats

st.set_page_config(page_title="Statisztika", page_icon="📊", layout="wide")
st.set_page_config(page_title="Statisztika")


# ===== CUSTOM CSS - TÖRTÉNELMI STÍLUS =====
st.markdown("""
<style>
    .main {
        background: linear-gradient(to bottom, #fffbf0, #fff9e6);
    }
    
    h1, h2, h3 {
        color: #2c1810 !important;
        font-family: 'Georgia', serif !important;
    }
    
    .subtitle {
        text-align: center;
        color: #5c4033;
        font-style: italic;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #fffbf0;
        border: 2px solid #d4af37;
        border-radius: 8px 8px 0 0;
        color: #2c1810;
        font-weight: bold;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(to bottom, #d4af37, #b8941f);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ===== ADATOK BETÖLTÉSE =====
@st.cache_data
def load_analytics_data():
    tripartit_df = pd.read_csv("data/Recept_halo__molekula_tripartit.csv", delimiter=";", encoding="utf-8")
    edges_df = pd.read_csv("data/recept_halo_edges.csv", encoding="utf-8")
    historical_df = pd.read_csv("data/HistoricalRecipe_export.csv", encoding="utf-8")
    return tripartit_df, edges_df, historical_df

tripartit_df, edges_df, historical_df = load_analytics_data()

# Szószám számítás
historical_df['word_count'] = historical_df['original_text'].fillna('').apply(lambda x: len(str(x).split()))

# ===== NODE TÍPUS MAPPING =====
# A CSV "type" oszlopát használjuk (nem Intervaltype!), magyar nevekkel
type_mapping = {
    'dish': 'Recept',
    'molecule': 'Molekula',
    'ingredient': 'Alapanyag'
}

# Próbáljuk meg megtalálni a típus oszlopot (különböző nevekkel)
type_column = None
for col in ['type', 'Type', 'Intervaltype', 'intervaltype']:
    if col in tripartit_df.columns:
        type_column = col
        break

if type_column:
    tripartit_df['node_type'] = tripartit_df[type_column].map(type_mapping)
    # Ha valami nem illeszkedik, "Egyéb" kategória
    tripartit_df['node_type'] = tripartit_df['node_type'].fillna('Egyéb')
else:
    # Fallback: ha nincs típus oszlop, mindenki "Egyéb"
    tripartit_df['node_type'] = 'Egyéb'
    st.warning("⚠️ Nem található típus oszlop a CSV-ben. Elérhető oszlopok: " + ", ".join(tripartit_df.columns.tolist()))

# ===== STATISZTIKAI ELOSZLÁS ELEMZÉS =====
def analyze_distribution(data):
    """
    Elemzi az adatok eloszlását és visszaadja a legjobban illeszkedő típust
    """
    # Alapstatisztikák
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    # Normalitás teszt (Shapiro-Wilk)
    # Ha p > 0.05, akkor normális eloszlás
    if len(data) < 5000:  # Shapiro-Wilk max 5000 mintára működik jól
        shapiro_stat, shapiro_p = stats.shapiro(data)
    else:
        # Nagy mintákra Anderson-Darling teszt
        anderson_result = stats.anderson(data, dist='norm')
        shapiro_p = 0.05 if anderson_result.statistic > anderson_result.critical_values[2] else 0.1
    
    # Különböző eloszlásokhoz illesztés
    distributions = {
        'norm': 'Normális',
        'lognorm': 'Lognormális',
        'expon': 'Exponenciális',
        'gamma': 'Gamma',
        'weibull_min': 'Weibull'
    }
    
    best_fit = None
    best_ks_stat = float('inf')
    best_dist_name = None
    
    for dist_name, dist_label in distributions.items():
        try:
            # Paraméter illesztés
            params = getattr(stats, dist_name).fit(data)
            # Kolmogorov-Smirnov teszt
            ks_stat, ks_p = stats.kstest(data, dist_name, args=params)
            
            if ks_stat < best_ks_stat:
                best_ks_stat = ks_stat
                best_fit = dist_label
                best_dist_name = dist_name
        except:
            continue
    
    # Eloszlás típus meghatározása a tulajdonságok alapján
    if shapiro_p > 0.05:
        dist_type = "✅ **Normális eloszlás**"
        explanation = "Az adatok normális eloszlást követnek (Shapiro-Wilk p > 0.05)"
    elif skewness > 1:
        dist_type = "📈 **Jobbra ferde eloszlás**"
        explanation = f"Erősen jobbra ferde (ferdeség: {skewness:.2f}). Van néhány extrém nagy érték."
    elif skewness > 0.5:
        dist_type = "📊 **Mérsékelten jobbra ferde**"
        explanation = f"Jobbra ferde (ferdeség: {skewness:.2f}). Az átlag > medián."
    elif skewness < -1:
        dist_type = "📉 **Balra ferde eloszlás**"
        explanation = f"Erősen balra ferde (ferdeség: {skewness:.2f})"
    elif abs(skewness) < 0.5:
        dist_type = "⚖️ **Szimmetrikus eloszlás**"
        explanation = f"Közel szimmetrikus (ferdeség: {skewness:.2f})"
    else:
        dist_type = "📊 **Aszimmetrikus eloszlás**"
        explanation = f"Balra ferde (ferdeség: {skewness:.2f})"
    
    # Csúcsosság értelmezése
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
        'shapiro_p': shapiro_p if 'shapiro_p' in locals() else None,
        'mean': mean,
        'median': median,
        'std': std
    }

# ===== HEADER =====
st.title("📊 Korpusz Analitika Dashboard")
st.markdown('<div class="subtitle">Hálózati statisztikák, recept hosszúság eloszlás és AI generálási stratégiák</div>', unsafe_allow_html=True)

# Dekoratív elválasztó
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
        metrics_col3.metric("Átlagos degree", round(tripartit_df['Degree'].mean(), 2))
        
        # Degree eloszlás histogram
        st.markdown("### 📊 Degree Eloszlás")
        fig_degree = go.Figure()
        
        # Histogram
        fig_degree.add_trace(go.Histogram(
            x=tripartit_df['Degree'],
            nbinsx=30,
            marker_color='#8b5a2b',
            name='Degree',
            opacity=0.7,
            histnorm='probability density'
        ))
        
        # Illesztett eloszlás görbe
        degree_data = tripartit_df['Degree'].values
        x_range = np.linspace(degree_data.min(), degree_data.max(), 100)
        
        # Normális eloszlás görbe
        mu, sigma = degree_data.mean(), degree_data.std()
        normal_curve = stats.norm.pdf(x_range, mu, sigma)
        fig_degree.add_trace(go.Scatter(
            x=x_range,
            y=normal_curve,
            mode='lines',
            name='Normális illesztés',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Lognormális illesztés (ha ez a legjobb)
        try:
            shape, loc, scale = stats.lognorm.fit(degree_data, floc=0)
            lognorm_curve = stats.lognorm.pdf(x_range, shape, loc, scale)
            fig_degree.add_trace(go.Scatter(
                x=x_range,
                y=lognorm_curve,
                mode='lines',
                name='Lognormális illesztés',
                line=dict(color='green', width=3)
            ))
        except:
            pass
        
        fig_degree.update_layout(
            xaxis_title="Degree",
            yaxis_title="Sűrűség",
            paper_bgcolor='#fcf5e5',
            plot_bgcolor='#fcf5e5',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig_degree, use_container_width=True)
        
        # Eloszlás analízis a Degree-re
        st.markdown("### 📈 Degree Eloszlás Elemzése")
        degree_analysis = analyze_distribution(tripartit_df['Degree'].values)
        
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
        
        # Részletes statisztika
        st.markdown("#### 📊 Típus Eloszlás")
        for node_type, count in type_counts.items():
            percent = (count / len(tripartit_df)) * 100
            emoji = {'Alapanyag': '🥘', 'Molekula': '⚗️', 'Recept': '📖', 'Egyéb': '⚪'}.get(node_type, '⚪')
            st.markdown(f"{emoji} **{node_type}:** {count} db ({percent:.1f}%)")
        
        # Pie chart
        fig_types = go.Figure(data=[go.Pie(
            labels=type_counts.index,
            values=type_counts.values,
            marker=dict(colors=['#8b5a2b', '#4a7c59', '#b85450', '#cccccc']),
            hole=0.4
        )])
        fig_types.update_layout(
            paper_bgcolor='#fcf5e5',
            height=350
        )
        st.plotly_chart(fig_types, use_container_width=True)
        
        # Top 10 legnagyobb degree
        st.markdown("### 🏆 Top 10 Node (degree szerint)")
        top_nodes = tripartit_df.nlargest(10, 'Degree')[['Label', 'Degree', 'node_type']]
        
        for idx, row in top_nodes.iterrows():
            emoji = {'Alapanyag': '🥘', 'Molekula': '⚗️', 'Recept': '📖', 'Egyéb': '⚪'}.get(row['node_type'], '⚪')
            st.markdown(f"{emoji} **{row['Label']}** - Degree: {row['Degree']}")

# ===== TAB 2: RECEPT HOSSZÚSÁG =====
with tab2:
    st.markdown("### 📏 Recept Hosszúság Statisztikák")
    
    # Leíró stat
    stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
    stats_col1.metric("Receptek száma", len(historical_df))
    stats_col2.metric("Átlag szó", round(historical_df['word_count'].mean(), 1))
    stats_col3.metric("Medián szó", int(historical_df['word_count'].median()))
    stats_col4.metric("Min szó", int(historical_df['word_count'].min()))
    stats_col5.metric("Max szó", int(historical_df['word_count'].max()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Szószám Eloszlás Histogram")
        
        fig_hist = go.Figure()
        
        # Histogram
        fig_hist.add_trace(go.Histogram(
            x=historical_df['word_count'],
            nbinsx=40,
            marker_color='#8b5a2b',
            name='Szószám',
            opacity=0.7,
            histnorm='probability density'
        ))
        
        # Illesztett eloszlás görbék
        word_data = historical_df['word_count'].values
        x_range = np.linspace(word_data.min(), word_data.max(), 200)
        
        # Normális eloszlás görbe (összehasonlításhoz)
        mu, sigma = word_data.mean(), word_data.std()
        normal_curve = stats.norm.pdf(x_range, mu, sigma)
        fig_hist.add_trace(go.Scatter(
            x=x_range,
            y=normal_curve,
            mode='lines',
            name='Normális (elméleti)',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Lognormális illesztés (valós eloszlás)
        try:
            shape, loc, scale = stats.lognorm.fit(word_data, floc=0)
            lognorm_curve = stats.lognorm.pdf(x_range, shape, loc, scale)
            fig_hist.add_trace(go.Scatter(
                x=x_range,
                y=lognorm_curve,
                mode='lines',
                name='Lognormális (illesztett)',
                line=dict(color='green', width=3)
            ))
        except:
            pass
        
        # Átlag és medián vonalak
        fig_hist.add_vline(
            x=historical_df['word_count'].mean(),
            line_dash="dash",
            line_color="darkred",
            annotation_text=f"Átlag: {historical_df['word_count'].mean():.1f}",
            annotation_position="top"
        )
        fig_hist.add_vline(
            x=historical_df['word_count'].median(),
            line_dash="dash",
            line_color="darkblue",
            annotation_text=f"Medián: {historical_df['word_count'].median():.0f}",
            annotation_position="top"
        )
        
        fig_hist.update_layout(
            xaxis_title="Szószám",
            yaxis_title="Sűrűség",
            paper_bgcolor='#fcf5e5',
            plot_bgcolor='#fcf5e5',
            height=450,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Eloszlás analízis a szószámra
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
            
            **"Jobbra ferde" = a hosszú farok jobbra (nagy értékek felé) nyúlik**
            
            Ez azt jelenti:
            - 📊 A legtöbb recept **rövid** (30-70 szó)
            - 📈 Van néhány **extrém hosszú** recept (200+ szó) → ezek "húzzák jobbra" a farok végét
            - ⚖️ **Átlag > Medián** (az outlier-ek felhúzzák az átlagot)
            - 🎯 A **medián megbízhatóbb** mint az átlag
            
            **Példa:** Ha 90% rövid (50 szó), de van 10 db 300+ szavas recept → 
            az átlag magasabb lesz, mint a tipikus recept hossza.
            """)
        
        # Vizuális magyarázat diagram
        st.markdown("#### 🎨 Eloszlás Vizualizáció")
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            st.markdown("""
            **🔴 Piros vonal (Normális):** Szimmetrikus haranggörbe
            - Átlag = Medián = Módusz
            - Nincs hosszú farok
            """)
        
        with col_viz2:
            st.markdown("""
            **🟢 Zöld vonal (Lognormális):** Jobbra ferde
            - Átlag > Medián
            - Hosszú jobb oldali farok
            - Ez illeszkedik az adatainkra!
            """)

    
    with col2:
        st.markdown("### 🥧 Hosszúság Kategóriák")
        
        # Kategorizálás
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
        
        colors = {
            'Nagyon rövid (≤30)': '#8b5a2b',
            'Rövid (31-60)': '#a67c52',
            'Közepes (61-100)': '#c9a877',
            'Hosszú (101-200)': '#dcc5a0',
            'Nagyon hosszú (>200)': '#f0e5d3'
        }
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=category_counts.index,
            values=category_counts.values,
            marker=dict(colors=[colors[cat] for cat in category_counts.index]),
            textinfo='label+percent',
            textposition='outside'
        )])
        fig_pie.update_layout(
            paper_bgcolor='#fcf5e5',
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Részletes lebontás
        st.markdown("#### 📋 Részletes Eloszlás")
        for cat, count in category_counts.items():
            percent = (count / len(historical_df)) * 100
            st.markdown(f"**{cat}:** {count} db ({percent:.1f}%)")

# ===== TAB 3: AI STRATÉGIÁK =====
with tab3:
    st.markdown("### 🤖 AI Generálási Stratégiák (GPT-5.1 & GPT-5-mini Best Practices)")
    
    strategies = [
        {
            "mode": "minimal",
            "trigger": "0 példa VAGY átlag degree < 3",
            "word_target": "max 40 szó",
            "style": "Emlékeztető stílus, minimális kontextus",
            "use_case": "Gyenge hálózati alap, nincs történeti példa",
            "prompt_key": "Ultra-concise, grounding warnings",
            "color": "#b85450"
        },
        {
            "mode": "concise",
            "trigger": "1-2 példa",
            "word_target": "40-70 szó",
            "style": "Lakonikus, tapasztalt szakács stílus",
            "use_case": "Kevés példa, közepes kapcsolódás",
            "prompt_key": "Terse instructions, assumed knowledge",
            "color": "#8b5a2b"
        },
        {
            "mode": "standard",
            "trigger": "3-5 példa",
            "word_target": "70-110 szó",
            "style": "Klasszikus 17. századi recept forma",
            "use_case": "Közepes példatár, erős hálózat",
            "prompt_key": "Complete but compact, contextual",
            "color": "#4a7c59"
        },
        {
            "mode": "detailed",
            "trigger": "6+ példa",
            "word_target": "110-160 szó",
            "style": "Részletes technológiai leírás kontextussal",
            "use_case": "Gazdag forrás, kiváló kapcsolódás",
            "prompt_key": "Step-by-step, timing, cultural context",
            "color": "#d4af37"
        }
    ]
    
    for strategy in strategies:
        with st.expander(f"**{strategy['mode'].upper()}** - {strategy['word_target']}", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Trigger feltétel:** {strategy['trigger']}")
                st.markdown(f"**Stílus:** {strategy['style']}")
                st.markdown(f"**Használat:** {strategy['use_case']}")
                st.markdown(f"**Prompt kulcselem:** `{strategy['prompt_key']}`")
            
            with col2:
                st.markdown(
                    f"<div style='background-color: {strategy['color']}; "
                    f"color: white; padding: 20px; border-radius: 10px; text-align: center;'>"
                    f"<h3 style='margin: 0;'>{strategy['word_target']}</h3>"
                    f"</div>",
                    unsafe_allow_html=True
                )
    
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
    
    # Keresés
    search = st.text_input("🔍 Keresés a receptekben", placeholder="Pl. hal, bors, leves...")
    
    # Szűrés hossz szerint
    col1, col2 = st.columns([1, 3])
    with col1:
        length_filter = st.selectbox(
            "Hosszúság szűrő",
            ["Összes", "Nagyon rövid (≤30)", "Rövid (31-60)", "Közepes (61-100)", "Hosszú (101-200)", "Nagyon hosszú (>200)"]
        )
    
    # Szűrt adatok
    filtered_df = historical_df.copy()
    
    if search:
        filtered_df = filtered_df[
            filtered_df['title'].fillna('').str.contains(search, case=False) |
            filtered_df['original_text'].fillna('').str.contains(search, case=False)
        ]
    
    if length_filter != "Összes":
        filtered_df = filtered_df[filtered_df['length_category'] == length_filter]
    
    st.markdown(f"**Találatok:** {len(filtered_df)} recept")
    
    # Megjelenítés
    for idx, row in filtered_df.head(20).iterrows():
        with st.expander(f"📖 {row['title']} ({row['word_count']} szó) - {row.get('source', 'Ismeretlen forrás')}"):
            st.markdown(f"**Kategória:** {row['length_category']}")
            st.markdown("---")
            st.markdown(row['original_text'][:500] + ("..." if len(str(row['original_text'])) > 500 else ""))

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #8b5a2b;'>"
    f"Analytics Dashboard © 2025 | Korpusz: {len(historical_df)} recept, {historical_df['word_count'].mean():.1f} szó átlag"
    "</div>",
    unsafe_allow_html=True

)


