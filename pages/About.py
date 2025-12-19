import streamlit as st

st.set_page_config(page_title="A Projektről", page_icon="📜", layout="wide")

# Custom CSS - Történelmi stílus
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

# ===== HEADER =====
st.markdown('<h1 class="main-title">A Projektről</h1>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ===== AZ OLVASÓHOZ IDÉZET =====
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

# ===== FŐ SZÖVEG =====
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

# ===== A FORRÁSMŰ =====
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

# ===== HÁLÓZATELEMZÉS ÉS GASZTRONÓMIA =====
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

# ===== TECHNIKAI RÉSZLETEK (új szekció) =====
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
            <li><strong>GPT-5.2 Prompting:</strong> Strukturált, grounding-alapú</li>
            <li><strong>Adaptív hosszúság:</strong> Korpusz-vezérelt (40-160 szó)</li>
            <li><strong>Network-informed:</strong> Degree-súlyozott döntések</li>
            <li><strong>Confidence score:</strong> Transzparens megbízhatóság</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ===== ADATOK =====
st.markdown("---")
st.markdown("""
<h3 class="section-title">
    📚 Az Adatbázis
</h3>
""", unsafe_allow_html=True)

# Metrikák
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
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: #fffbf0; border-radius: 8px; border: 2px solid #d4af37;">
        <div style="font-size: 2.5rem; font-weight: bold; color: #8b5a2b;">32%</div>
        <div style="color: #4a3728; font-size: 1rem; margin-top: 0.5rem;">Böjti Receptek</div>
    </div>
    """, unsafe_allow_html=True)

# ===== HIVATKOZÁSOK =====
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
            <strong>OpenAI:</strong> GPT-5.2 Prompting Guide 
            - <a href="https://cookbook.openai.com/examples/gpt-5/gpt-5-2_prompting_guide" target="_blank">cookbook.openai.com</a>
        </li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ===== KUTATÁSI KÉRDÉSEK =====
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
        <li><strong>Multimodal:</strong> Kódex-képek feldolgozása (illusztrációk, margójelek)</li>
        <li><strong>Temporal:</strong> Időbeli változások (XVI. vs. XVIII. század)</li>
        <li><strong>Regionális:</strong> Földrajzi különbségek (Erdély, Dunántúl, Felvidék)</li>
        <li><strong>Evaluation:</strong> AI minőségellenőrzés human evaluátorokkal</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# ===== ZÁRÓ IDÉZET =====
st.markdown("---")
st.markdown("""
<div class="highlight-box" style="text-align: center; font-size: 1.3rem;">
    "A múlt ízeit megérteni egyet jelent azzal, hogy a jelen számára új utakat nyitunk 
    a gasztronómia művészetében."
</div>
""", unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("""
<div style="text-align: center; margin-top: 4rem; padding: 2rem; background: linear-gradient(to bottom, #fffbf0, #fff9e6); border-radius: 8px;">
    <div style="font-size: 1.5rem; font-weight: bold; color: #2c1810; font-family: Georgia, serif; margin-bottom: 1rem;">
        Közrendek Ízhálója
    </div>
    <div style="color: #5c4033; font-size: 1rem; margin-bottom: 0.5rem;">
        Hálózatelemzés + Történeti Források + AI Generálás
    </div>
    <div style="color: #8b5a2b; font-size: 0.9rem;">
        © 2025 | Built with Streamlit, NetworkX, Plotly & OpenAI GPT-5.2
    </div>
</div>

""", unsafe_allow_html=True)

