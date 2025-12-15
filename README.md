# 🍲 Közrendek Ízhálója

> **Középkori magyar ízkapcsolatok hálózatelemzése és AI-alapú receptgenerálás**

## 📋 Tartalomjegyzék

- [Áttekintés](#áttekintés)
- [Jellemzők](#jellemzők)
- [Telepítés](#telepítés)
- [Használat](#használat)
- [Projekt Struktúra](#projekt-struktúra)
- [AI Logika](#ai-logika)
- [Adatok](#adatok)
- [Technológiák](#technológiák)

---

## 🎯 Áttekintés

A **Közrendek Ízhálója** a XVII. századi magyar gasztronómia modern hálózattudományi megközelítése. A projekt a híres **"Szakácsmesterségnek könyvecskéje"** (Kolozsvár, 1698) receptjeinek alapanyag-kapcsolatait elemzi, és AI segítségével új, stílusban illeszkedő recepteket generál.

### 📚 A Forrásmű

A [Szakácsmesterségnek könyvecskéje](https://mek.oszk.hu/08300/08343/08343.htm#252) az egyik legkorábbi ránk maradt magyar nyelvű nyomtatott szakácskönyv. Receptjei nem pontos mennyiségeket, hanem arányokat és eljárásokat rögzítenek — a "becsületes közrendeknek" készült, akik tapasztalatból főztek.

### 🕸️ Barabási-féle Flavor Network

Barabási Albert-László *Network Science* módszertanát követve tripartit hálózatot építettünk:

1. **Hálózatelemzéssel** feltérképezi a középkori magyar közrendi konyha alapanyag-kapcsolatait
2. **Történeti forrásokat** (330 receptet) dolgoz fel statisztikai módszerekkel
3. **AI-alapú receptgenerálást** végez a hálózati struktúra és történeti példák kombinálásával
4. **Molekuláris gasztronómia** kapcsolatokat integrál (íz-aroma profilok)

### 🔬 Módszertan

- **Tripartit hálózat:** Alapanyagok ↔ Molekulák ↔ Receptek
- **Degree-súlyozott kapcsolatok:** Erős párosítások azonosítása
- **Korpusz analitika:** 330 történeti recept szövegbányászata (átlag 70.7 szó)
- **GPT-5.2 best practices:** Strukturált prompt engineering

---

## ✨ Jellemzők

### 🏠 Home Oldal
- **Interaktív hálózati térkép** (Plotly)
  - Degree-alapú node méretezés
  - Típus-specifikus színkódolás (alapanyag/molekula/recept)
- **Keresés & szűrés** (típus, degree)
- **Történeti példák** megjelenítése
- **AI receptgenerálás** confidence score-ral

### 📊 Analytics Oldal
- Hálózati statisztikák (degree eloszlás, top node-ok)
- Recept hosszúság elemzés (histogram, kategóriák)
- AI generálási stratégiák (4 mód)
- Korpusz böngésző

### ℹ️ About Oldal
- Projekt módszertan
- AI logika részletesen
- Használati útmutató

---

## 🚀 Telepítés

### 1️⃣ Követelmények

- Python 3.10+
- pip
- OpenAI API kulcs

### 2️⃣ Repository klónozása

```bash
git clone https://github.com/your-username/kozrendek-izhaloja.git
cd kozrendek-izhaloja
```

### 3️⃣ Virtuális környezet

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4️⃣ Függőségek telepítése

```bash
pip install -r requirements.txt
```

**requirements.txt tartalma:**
```
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.17.0
networkx>=3.1
python-dotenv>=1.0.0
openai>=1.0.0
```

### 5️⃣ .env fájl létrehozása

Hozz létre egy `.env` fájlt a projekt gyökerében:

```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

⚠️ **FONTOS:** A `.env` fájl NE kerüljön verziókezelésbe! (lásd `.gitignore`)

### 6️⃣ Adatok elhelyezése

Győződj meg róla, hogy a `data/` mappában a következő fájlok vannak:
- `Recept_halo__molekula_tripartit.csv`
- `recept_halo_edges.csv`
- `HistoricalRecipe_export.csv`

---

## 🎮 Használat

### Indítás

```bash
streamlit run app.py
```

Az app megnyílik a böngészőben: `http://localhost:8501`

### Alapvető Workflow

1. **Keresés:** Írj be egy alapanyagot (pl. "bors", "hal")
2. **Szűrés:** Sidebar → Típus + Min degree
3. **Node kiválasztás:** Kattints egy gombra
4. **Eredmények:**
   - **Bal:** Hálózati térkép
   - **Közép:** Történeti példák
   - **Jobb:** AI generált recept

---

## 📁 Projekt Struktúra

```
kozrendek-izhaloja/
│
├── app.py                          # Főoldal (Home)
├── pages/
│   ├── 1_📊_Analytics.py          # Analitika dashboard
│   └── 2_ℹ️_About.py              # Információk
│
├── data/
│   ├── Recept_halo__molekula_tripartit.csv
│   ├── recept_halo_edges.csv
│   └── HistoricalRecipe_export.csv
│
├── .env                            # API kulcs (NE töltsd fel!)
├── .env.example                    # Példa .env fájl
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🤖 AI Logika

### GPT-5.2 Prompt Engineering

Az AI generálás az [OpenAI GPT-5.2 Prompting Guide](https://cookbook.openai.com/examples/gpt-5/gpt-5-2_prompting_guide) alapján épül fel.

#### 🧩 Strukturált Prompt

```xml
<role> - Történeti gasztronómia kutatóasszisztens
<task> - Hálózat-alapú recept generálás
<constraints>
  <grounding_and_accuracy> - Anti-hallucináció
  <output_verbosity_spec> - Adaptív hosszúság
  <network_informed_reasoning> - Degree-súlyozott döntések
  <high_risk_self_check> - Validáció generálás előtt
</constraints>
<structured_output> - JSON schema
<reasoning_strategy> - Lépésről-lépésre
```

#### 📏 Adaptív Verbosity Control

| Mód | Trigger | Szószám | Stílus |
|-----|---------|---------|--------|
| **Minimal** | 0 példa VAGY degree < 3 | max 40 | Emlékeztető |
| **Concise** | 1-2 példa | 40-70 | Lakonikus |
| **Standard** | 3-5 példa | 70-110 | Klasszikus 18. sz. |
| **Detailed** | 6+ példa | 110-160 | Technológiai |

**Indoklás:** Korpusz 57%-a ≤60 szó → Default rövid stílus

#### 🎯 Grounding & Accuracy

- ⛔ **SZIGORÚ TILTÁS:** Kitalált alapanyagok, források
- ✅ **KÖTELEZŐ:** Minden alapanyag a hálózati kapcsolatokban
- ⚠️ **Confidence:** low/medium/high
- 📚 **Source note:** Transzparens forrásjelölés

---

## 📊 Adatok

### Hálózati Adatok

- **Node-ok:** ~450 (alapanyagok, molekulák, receptek)
- **Kapcsolatok:** ~800 él
- **Átlagos degree:** 3.5

### Történeti Korpusz

- **Receptek száma:** 330
- **Átlagos hossz:** 70.7 szó
- **Medián:** 61 szó
- **Eloszlás:** Jobbra ferde (átlag > medián)
- **Böjti receptek:** ~32%

### Adatforrások

- Magyar Nemzeti Múzeum Könyvtára
- 18. századi szakácskönyvek
- Molekuláris gasztronómia adatbázisok

---

## 🛠️ Technológiák

### Frontend
- **Streamlit** - Multi-page web app
- **Plotly** - Interaktív vizualizációk
- **NetworkX** - Gráfstruktúrák

### Backend
- **Python 3.10+**
- **Pandas** - Adatelemzés
- **OpenAI GPT-4o** - AI receptgenerálás
- **python-dotenv** - Környezeti változók

---

## 🔬 Kutatási Kérdések

### Jelenlegi fókusz
1. Mely alapanyagok a legközpontibbak a középkori magyar konyhában?
2. Van-e kapcsolat az íz-aroma molekulák és a történeti párosítások között?
3. Hogyan térképezhető fel a böjti konyha a hálózatban?
4. Mennyire közelíti meg az AI a történeti stílust?

### Jövőbeli irányok
- Multimodal (képfeldolgozás)
- Temporal (időbeli változások)
- Regionális (földrajzi különbségek)
- Evaluation (AI minőségellenőrzés)

---

## 📚 Hivatkozások

- [OpenAI GPT-5.2 Prompting Guide](https://cookbook.openai.com/examples/gpt-5/gpt-5-2_prompting_guide)
- NetworkX Documentation
- Ahn, Y. Y., et al. (2011). "Flavor network and the principles of food pairing." *Scientific Reports*.

---

## 📄 Licenc

MIT License - lásd a `LICENSE` fájlt

---

## 👥 Közreműködés

Közreműködés várható! Issues és pull requestek szívesen fogadottak.

---

## 📞 Kapcsolat

- **Email:** your.email@example.com
- **GitHub:** [@your-username](https://github.com/your-username)

---

<div align="center">
  <p><strong>Közrendek Ízhálója © 2025</strong></p>
  <p>Hálózatelemzés + Történeti Források + AI Generálás</p>
  <p>Built with ❤️ using Streamlit, NetworkX, Plotly & OpenAI GPT-4o</p>
</div>