import streamlit as st
import pandas as pd

# Root cause of the segfaults found via faulthandler: pandas 3.0's new default
# Arrow-backed string dtype (pandas/core/arrays/string_arrow.py) is unstable in
# this environment and crashes the process after enough repeated operations on
# string columns (read_csv, dataframe display, cache hashing all touch it).
# Forcing the classic numpy-object string handling avoids that code path
# entirely. This must run before any pd.read_csv call.
pd.set_option('future.infer_string', False)

import numpy as np
import joblib

from src.visuals import (
    plot_histogram, plot_box, plot_count, plot_correlation_heatmap,
    plot_scatter, plot_bar_by_category, plot_class_distribution,
    plot_confidence_gauge, CATEGORICAL,
)

DATA_PATH = 'data/processed/cleaned_semi_processed.csv'
MODEL_READY_PATH = 'data/processed/clinvar_model_ready.csv'
MODEL_PATH = 'data/processed/rf_model.pkl'
FEATURES_PATH = 'data/processed/model_features.pkl'

st.set_page_config(page_title='ClinVar Variant Explorer', page_icon='🧬', layout='wide')

# ---------------------------------------------------------------------------
# Custom styling: animated gradient hero, glass KPI cards, hover motion
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

#MainMenu, footer, header { visibility: hidden; }

.block-container { padding-top: 1.5rem; max-width: 1300px; }

/* -------------------- Dramatic dark theme -------------------- */
.stApp {
  background:
    radial-gradient(ellipse 70% 45% at 15% 0%, rgba(13,148,136,0.16) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 100% 20%, rgba(219,39,119,0.10) 0%, transparent 60%),
    linear-gradient(180deg, #060a14 0%, #080d1a 40%, #05070d 100%);
}
.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
.stMarkdown, .stMarkdown p, div[data-testid="stMarkdownContainer"] {
  color: #dbe4f0;
}
h1, h2, h3, h4, .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: #f4f7fb !important; }
div[data-testid="stVerticalBlock"] h3 { position: relative; padding-bottom: 0.35rem; }
div[data-testid="stVerticalBlock"] h3::after {
  content: ''; position: absolute; left: 0; bottom: 0; width: 46px; height: 3px; border-radius: 3px;
  background: linear-gradient(90deg, #2dd4bf, #db2777);
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a1120 0%, #070b16 100%);
  border-right: 1px solid rgba(45,212,191,0.14);
}
section[data-testid="stSidebar"] * { color: #cbd5e1; }
section[data-testid="stSidebar"] h2 {
  color: #5eead4 !important; font-size: 1.05rem;
  text-shadow: 0 0 16px rgba(45,212,191,0.35);
}
section[data-testid="stSidebar"] a { color: #5eead4; }
section[data-testid="stSidebar"] hr { border-color: rgba(148,163,184,0.15); }

/* Sidebar form widgets: darken the BaseWeb chrome so it sits inside the dark rail */
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="base-input"],
section[data-testid="stSidebar"] input {
  background-color: #101a2e !important; border-color: rgba(45,212,191,0.25) !important; color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] [data-baseweb="tag"] { background-color: #0f766e !important; }
section[data-testid="stSidebar"] div[data-testid="stForm"] {
  background: rgba(15,23,42,0.55); border: 1px solid rgba(45,212,191,0.15);
  border-radius: 14px; padding: 1.1rem 1rem 0.6rem;
}
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div { background: #1e293b !important; }
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
  background: #2dd4bf !important; box-shadow: 0 0 10px rgba(45,212,191,0.7);
}

button[kind="primary"], button[kind="formSubmit"] {
  background: linear-gradient(135deg, #0d9488, #db2777) !important; border: none !important;
  box-shadow: 0 4px 18px rgba(219,39,119,0.35); transition: box-shadow 0.25s ease, transform 0.2s ease;
}
button[kind="primary"]:hover, button[kind="formSubmit"]:hover {
  box-shadow: 0 6px 26px rgba(219,39,119,0.55); transform: translateY(-2px);
}

@keyframes floatIn {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes glowDrift {
  0%, 100% { transform: translate(0, 0) scale(1); }
  33%      { transform: translate(3%, -4%) scale(1.08); }
  66%      { transform: translate(-2%, 3%) scale(0.95); }
}
@keyframes particleFloat {
  0%, 100% { transform: translateY(0) scale(1); opacity: 0.55; }
  50%      { transform: translateY(-16px) scale(1.3); opacity: 1; }
}
@keyframes helixSpin {
  0%   { stroke-dashoffset: 0; }
  100% { stroke-dashoffset: -200; }
}

.hero {
  position: relative;
  overflow: hidden;
  background: radial-gradient(ellipse 120% 100% at 20% -10%, #133a3a 0%, #0a1224 45%, #050810 100%);
  border-radius: 20px;
  padding: 2.4rem 2.6rem;
  margin-bottom: 1.4rem;
  box-shadow: 0 20px 50px rgba(3, 10, 25, 0.45);
  animation: floatIn 0.7s ease-out;
}
.hero::before, .hero::after {
  content: ''; position: absolute; border-radius: 50%; filter: blur(45px);
  animation: glowDrift 14s ease-in-out infinite;
}
.hero::before {
  width: 340px; height: 340px; background: radial-gradient(circle, rgba(20,184,166,0.55), transparent 70%);
  top: -100px; right: -60px;
}
.hero::after {
  width: 280px; height: 280px; background: radial-gradient(circle, rgba(236,72,153,0.4), transparent 70%);
  bottom: -120px; left: 10%; animation-delay: -6s;
}
.hero-particles span {
  position: absolute; width: 5px; height: 5px; border-radius: 50%;
  background: #5eead4; box-shadow: 0 0 8px 2px rgba(94,234,212,0.9);
  animation: particleFloat 4s ease-in-out infinite;
}
.hero-content { position: relative; z-index: 2; }
.hero h1 {
  color: white; font-weight: 800; font-size: 2.35rem; margin: 0;
  text-shadow: 0 0 22px rgba(45,212,191,0.45);
}
.hero p { color: rgba(226,232,240,0.85); font-size: 1.05rem; margin-top: 0.6rem; max-width: 680px; }
.hero .dna-emoji { display: inline-block; filter: drop-shadow(0 0 10px rgba(45,212,191,0.8)); }
.hero-helix { position: absolute; right: 28px; top: 50%; transform: translateY(-50%); opacity: 0.85; z-index: 1; }

.kpi-card {
  background: linear-gradient(160deg, #101a2e 0%, #0b1220 100%);
  border-radius: 16px; padding: 1.1rem 1.3rem;
  box-shadow: 0 4px 18px rgba(0,0,0,0.35);
  border: 1px solid rgba(148,163,184,0.12);
  border-top: 3px solid var(--kpi-accent, #14b8a6);
  transition: transform 0.25s ease, box-shadow 0.25s ease;
  animation: floatIn 0.6s ease-out;
}
.kpi-card:hover { transform: translateY(-5px); box-shadow: 0 14px 30px rgba(45,212,191,0.22); }
.kpi-label { color: #94a3b8; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
.kpi-value { color: #f8fafc; font-size: 1.9rem; font-weight: 800; margin-top: 0.2rem; }
.kpi-sub { font-size: 0.85rem; margin-top: 0.15rem; font-weight: 600; }

.insight-card {
  background: linear-gradient(135deg, rgba(20,184,166,0.14), rgba(219,39,119,0.10));
  border-left: 4px solid #2dd4bf;
  border-radius: 12px; padding: 1rem 1.3rem; margin-bottom: 0.9rem;
  animation: floatIn 0.5s ease-out;
  transition: transform 0.2s ease;
  color: #e2e8f0;
}
.insight-card:hover { transform: translateX(4px); box-shadow: 0 6px 20px rgba(45,212,191,0.15); }
.insight-card b { color: #5eead4; }

div[data-testid="stMetric"] { display: none; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
  border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 600;
}

div[data-testid="stRadio"] > div { gap: 6px; border-bottom: 1px solid rgba(148,163,184,0.15); padding-bottom: 0; }
div[data-testid="stRadio"] label {
  border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 600;
  background: transparent; color: #94a3b8 !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
  background: rgba(45,212,191,0.12); border-bottom: 3px solid #2dd4bf;
  color: #f4f7fb !important;
}
div[data-testid="stRadio"] label:has(input:checked) p { color: #f4f7fb !important; }

.table-wrap {
  max-height: 320px; overflow: auto; border-radius: 10px;
  border: 1px solid rgba(148,163,184,0.15); margin-bottom: 1rem;
  background: #0b1220;
}
.dash-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.dash-table th {
  position: sticky; top: 0; background: #101a2e; color: #5eead4; font-weight: 700;
  text-align: left; padding: 8px 12px; border-bottom: 1px solid rgba(148,163,184,0.2);
}
.dash-table td { padding: 6px 12px; border-bottom: 1px solid rgba(148,163,184,0.08); color: #dbe4f0; }
.dash-table tr:hover td { background: rgba(45,212,191,0.06); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data & model loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, low_memory=False)


@st.cache_data
def load_model_ready():
    return pd.read_csv(MODEL_READY_PATH, low_memory=False)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features


df = load_data()
model_ready_df = load_model_ready()
rf_model, model_features = load_model()

IMPACT_ORDER = ['MODIFIER', 'LOW', 'MODERATE', 'HIGH']
IMPACT_MAP = {name: i for i, name in enumerate(IMPACT_ORDER)}

# ---------------------------------------------------------------------------
# Sidebar: dataset description + filters
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🧬 About This Dataset")
st.sidebar.markdown(
    "**ClinVar Conflicting Classifications** — ~65K genetic variants from the "
    "NCBI ClinVar database. The target marks whether clinical laboratories "
    "*agree* or *conflict* on a variant's clinical interpretation.\n\n"
    "[View source on Kaggle](https://www.kaggle.com/datasets/kevinarvai/clinvar-conflicting)"
)
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔍 Filters")

# Filters live inside a form so adjusting several of them only triggers one
# rerun (on "Apply Filters"), not one rerun per widget touch. Regenerating
# charts on every single click is what exhausts the environment's chart
# rendering resources over a session (see the note above the section radio).
chrom_options = sorted(df['chrom'].astype(str).unique(), key=lambda x: (x.isdigit() is False, x.zfill(2) if x.isdigit() else x))
cadd_min, cadd_max = float(df['cadd_phred'].min()), float(df['cadd_phred'].max())

with st.sidebar.form("filters_form"):
    chrom_filter = st.multiselect("Chromosome", options=chrom_options, default=[])
    impact_filter = st.multiselect("Impact", options=IMPACT_ORDER, default=IMPACT_ORDER)
    cadd_range = st.slider("CADD_PHRED range", cadd_min, cadd_max, (cadd_min, cadd_max))
    class_filter = st.multiselect(
        "Class", options=['Concordant', 'Conflicting'], default=['Concordant', 'Conflicting']
    )
    st.form_submit_button("Apply Filters", type="primary", width='stretch')

filtered_df = df.copy()
if chrom_filter:
    filtered_df = filtered_df[filtered_df['chrom'].astype(str).isin(chrom_filter)]
if impact_filter:
    filtered_df = filtered_df[filtered_df['impact'].isin(impact_filter)]
filtered_df = filtered_df[(filtered_df['cadd_phred'] >= cadd_range[0]) & (filtered_df['cadd_phred'] <= cadd_range[1])]
class_map_reverse = {'Concordant': 0, 'Conflicting': 1}
if class_filter:
    filtered_df = filtered_df[filtered_df['class'].isin([class_map_reverse[c] for c in class_filter])]

st.sidebar.markdown(f"**{len(filtered_df):,}** variants match your filters")

if len(filtered_df) == 0:
    st.warning("⚠️ No variants match the current filter combination. Please widen your filters in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <svg class="hero-helix" width="60" height="320" viewBox="0 0 60 320" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="strandGrad1" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#2dd4bf"/><stop offset="100%" stop-color="#0ea5a0"/>
      </linearGradient>
      <linearGradient id="strandGrad2" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#f472b6"/><stop offset="100%" stop-color="#a855f7"/>
      </linearGradient>
      <linearGradient id="rungGrad" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#2dd4bf"/><stop offset="100%" stop-color="#f472b6"/>
      </linearGradient>
    </defs>
    <g opacity="0.9">
      <line x1="30.0" y1="0.0" x2="30.0" y2="0.0" stroke="url(#rungGrad)" stroke-width="1.5" opacity="0.55"/>
      <line x1="50.7" y1="32.0" x2="9.3" y2="32.0" stroke="url(#rungGrad)" stroke-width="1.5" opacity="0.55"/>
      <line x1="43.8" y1="64.0" x2="16.2" y2="64.0" stroke="url(#rungGrad)" stroke-width="1.5" opacity="0.55"/>
      <line x1="18.5" y1="96.0" x2="41.5" y2="96.0" stroke="url(#rungGrad)" stroke-width="1.5" opacity="0.55"/>
      <line x1="8.5" y1="128.0" x2="51.5" y2="128.0" stroke="url(#rungGrad)" stroke-width="1.5" opacity="0.55"/>
      <line x1="27.2" y1="160.0" x2="32.8" y2="160.0" stroke="url(#rungGrad)" stroke-width="1.5" opacity="0.55"/>
      <line x1="49.6" y1="192.0" x2="10.4" y2="192.0" stroke="url(#rungGrad)" stroke-width="1.5" opacity="0.55"/>
      <line x1="45.9" y1="224.0" x2="14.1" y2="224.0" stroke="url(#rungGrad)" stroke-width="1.5" opacity="0.55"/>
      <line x1="21.0" y1="256.0" x2="39.0" y2="256.0" stroke="url(#rungGrad)" stroke-width="1.5" opacity="0.55"/>
      <line x1="8.1" y1="288.0" x2="51.9" y2="288.0" stroke="url(#rungGrad)" stroke-width="1.5" opacity="0.55"/>
      <path d="M 30.0,0.0 L 36.7,8.0 L 42.7,16.0 L 47.5,24.0 L 50.7,32.0 L 52.0,40.0 L 51.2,48.0 L 48.4,56.0 L 43.8,64.0 L 38.0,72.0 L 31.4,80.0 L 24.7,88.0 L 18.5,96.0 L 13.4,104.0 L 9.8,112.0 L 8.1,120.0 L 8.5,128.0 L 10.9,136.0 L 15.1,144.0 L 20.7,152.0 L 27.2,160.0 L 33.9,168.0 L 40.3,176.0 L 45.7,184.0 L 49.6,192.0 L 51.7,200.0 L 51.8,208.0 L 49.8,216.0 L 45.9,224.0 L 40.6,232.0 L 34.2,240.0 L 27.5,248.0 L 21.0,256.0 L 15.3,264.0 L 11.1,272.0 L 8.6,280.0 L 8.1,288.0 L 9.6,296.0 L 13.1,304.0 L 18.2,312.0 L 24.4,320.0"
            fill="none" stroke="url(#strandGrad1)" stroke-width="3" stroke-linecap="round"/>
      <path d="M 30.0,0.0 L 23.3,8.0 L 17.3,16.0 L 12.5,24.0 L 9.3,32.0 L 8.0,40.0 L 8.8,48.0 L 11.6,56.0 L 16.2,64.0 L 22.0,72.0 L 28.6,80.0 L 35.3,88.0 L 41.5,96.0 L 46.6,104.0 L 50.2,112.0 L 51.9,120.0 L 51.5,128.0 L 49.1,136.0 L 44.9,144.0 L 39.3,152.0 L 32.8,160.0 L 26.1,168.0 L 19.7,176.0 L 14.3,184.0 L 10.4,192.0 L 8.3,200.0 L 8.2,208.0 L 10.2,216.0 L 14.1,224.0 L 19.4,232.0 L 25.8,240.0 L 32.5,248.0 L 39.0,256.0 L 44.7,264.0 L 48.9,272.0 L 51.4,280.0 L 51.9,288.0 L 50.4,296.0 L 46.9,304.0 L 41.8,312.0 L 35.6,320.0"
            fill="none" stroke="url(#strandGrad2)" stroke-width="3" stroke-linecap="round"/>
    </g>
  </svg>
  <div class="hero-particles">
    <span style="left:12%; top:20%; animation-delay:0s;"></span>
    <span style="left:22%; top:65%; animation-delay:0.6s;"></span>
    <span style="left:8%;  top:80%; animation-delay:1.2s;"></span>
    <span style="left:30%; top:35%; animation-delay:1.8s;"></span>
    <span style="left:18%; top:50%; animation-delay:2.4s;"></span>
  </div>
  <div class="hero-content">
    <h1><span class="dna-emoji">🧬</span> ClinVar Variant Insights Dashboard</h1>
    <p>Exploring what makes a genetic variant's clinical interpretation conflicting vs. concordant —
    from raw data to a live predictive model.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
conflict_pct = 100 * filtered_df['class'].mean() if len(filtered_df) else 0
avg_cadd = filtered_df['cadd_phred'].mean() if len(filtered_df) else 0
avg_af = filtered_df['mean_af'].mean() if len(filtered_df) else 0

k1, k2, k3, k4 = st.columns(4)
kpi_data = [
    (k1, "Variants Shown", f"{len(filtered_df):,}", f"of {len(df):,} total", '#0d9488'),
    (k2, "Conflicting Rate", f"{conflict_pct:.1f}%", "of filtered variants", '#db2777'),
    (k3, "Avg CADD Score", f"{avg_cadd:.1f}", "pathogenicity score", '#7c3aed'),
    (k4, "Avg Allele Freq.", f"{avg_af:.4f}", "population frequency", '#d97706'),
]
for col, label, value, sub, color in kpi_data:
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="--kpi-accent: {color};">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-sub" style="color:{color};">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Section navigation
# ---------------------------------------------------------------------------
# A plain radio (not st.tabs) is used deliberately: st.tabs executes the code
# inside every tab on every rerun regardless of which tab is visible, which
# means every filter change would regenerate all ~9 charts across all 4
# sections. Repeated Plotly figure creation across many reruns crashes the
# Python process in this environment (a native segfault, not a Streamlit
# exception) — see notebooks/04_modeling.ipynb for context on the model used
# below. Branching on a plain if/elif means only the active section's charts
# are generated per rerun, keeping cumulative chart creation low.
section = st.radio(
    "Section", ["📋 Overview", "📊 Explore", "🤖 Predict", "💡 Insights"],
    horizontal=True, label_visibility="collapsed"
)

if section == "📋 Overview":
    st.subheader("Data Preview")
    # Rendered as a plain HTML table, not st.dataframe/st.table: both of those
    # elements serialize through pyarrow, and repeated calls combined with
    # repeated Plotly chart creation segfault this environment's pyarrow build
    # after enough reruns (confirmed with faulthandler — the crash lands
    # inside pyarrow's dataframe_to_arrays every time). A plain HTML string
    # via st.markdown never touches Arrow at all.
    st.markdown(
        f'<div class="table-wrap">{filtered_df.head(15).to_html(index=False, classes="dash-table")}</div>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Summary Statistics")
        st.markdown(
            f'<div class="table-wrap">{filtered_df.describe().T.round(3).to_html(classes="dash-table")}</div>',
            unsafe_allow_html=True
        )
    with c2:
        st.subheader("Class Distribution")
        if len(filtered_df):
            st.plotly_chart(plot_class_distribution(filtered_df), width='stretch')
        else:
            st.info("No variants match the current filters.")

elif section == "📊 Explore":
    st.subheader("Univariate Analysis")
    num_cols = ['cadd_phred', 'mean_af', 'loftool', 'clinical_disease_count']
    cat_cols = ['impact', 'consequence', 'clnvc', 'chrom']

    e1, e2 = st.columns([1, 1])
    with e1:
        num_choice = st.selectbox("Numeric feature (Histogram + Boxplot)", num_cols)
        if len(filtered_df):
            st.plotly_chart(plot_histogram(filtered_df, num_choice), width='stretch')
            st.plotly_chart(plot_box(filtered_df, num_choice), width='stretch')
    with e2:
        cat_choice = st.selectbox("Categorical feature (Count by Class)", cat_cols)
        if len(filtered_df):
            st.plotly_chart(plot_count(filtered_df, cat_choice), width='stretch')

    st.markdown("---")
    st.subheader("Bivariate & Multivariate Analysis")

    b1, b2 = st.columns([1, 1])
    with b1:
        x_axis = st.selectbox("Scatter — X axis", num_cols, index=0)
        y_axis = st.selectbox("Scatter — Y axis", num_cols, index=1)
        if len(filtered_df):
            st.plotly_chart(plot_scatter(filtered_df, x_axis, y_axis), width='stretch')
    with b2:
        bar_cat = st.selectbox("Bar Plot — average by category", cat_cols, index=0)
        bar_val = st.selectbox("Bar Plot — value", num_cols, index=0)
        if len(filtered_df):
            st.plotly_chart(plot_bar_by_category(filtered_df, bar_cat, bar_val), width='stretch')

    st.subheader("Correlation Heatmap")
    if len(filtered_df):
        st.plotly_chart(plot_correlation_heatmap(filtered_df), width='stretch')

elif section == "🤖 Predict":
    st.subheader("🤖 Live Prediction: Will This Variant Be Conflicting?")
    st.markdown(
        "Adjust the key characteristics below — the rest of the variant's features are held at "
        "the dataset's typical (median) values — and see the trained Random Forest's live prediction."
    )

    base_row = model_ready_df.drop(columns=['class']).median(numeric_only=True)

    p1, p2 = st.columns([1, 1])
    with p1:
        in_cadd = st.slider("CADD_PHRED (pathogenicity score)", 0.0, 99.0, float(base_row['cadd_phred']))
        in_af = st.slider("Mean Allele Frequency", 0.0, 0.5, 0.01, step=0.001, format="%.3f")
        in_loftool = st.slider("LoFtool score", 0.0, 1.0, float(base_row['loftool']))
    with p2:
        in_disease_count = st.slider("Clinical Disease Count", 0, 40, int(base_row['clinical_disease_count']))
        in_impact = st.selectbox("Impact", IMPACT_ORDER, index=2)
        predict_clicked = st.button("🔮 Predict", width='stretch', type="primary")

    if predict_clicked:
        input_row = base_row.copy()
        input_row['cadd_phred'] = in_cadd
        input_row['mean_af'] = np.log1p(in_af)
        input_row['loftool'] = in_loftool
        input_row['clinical_disease_count'] = in_disease_count
        input_row['impact'] = IMPACT_MAP[in_impact]

        X_input = pd.DataFrame([input_row.reindex(model_features)], columns=model_features)
        probability = rf_model.predict_proba(X_input)[0][1]

        g1, g2 = st.columns([1, 1])
        with g1:
            st.plotly_chart(plot_confidence_gauge(probability), width='stretch')
        with g2:
            label = "⚠️ Likely Conflicting" if probability >= 0.5 else "✅ Likely Concordant"
            st.markdown(f"### {label}")
            st.markdown(f"Predicted probability of **conflicting** classification: **{probability * 100:.1f}%**")
            st.markdown(
                "This reflects the Random Forest model trained in `notebooks/04_modeling.ipynb` "
                "(ROC-AUC 0.79), using `class_weight='balanced'` to account for the dataset's "
                "~75/25 class imbalance."
            )

elif section == "💡 Insights":
    st.subheader("💡 Key Insights")
    insights = [
        ("Class Imbalance", "The dataset is ~75% Concordant vs. ~25% Conflicting — models are trained with <code>class_weight='balanced'</code> rather than plain accuracy optimization."),
        ("Strongest Predictor", "<code>mean_af</code> (population allele frequency) dominates feature importance — rarer variants are far more likely to receive conflicting interpretations."),
        ("Pathogenicity Signal", "Average CADD_PHRED score rises with variant impact severity (MODIFIER → LOW → MODERATE → HIGH), matching biological expectation."),
        ("Model Performance", "Random Forest reaches ROC-AUC 0.79 with 75% recall on the Conflicting class, clearly outperforming Logistic Regression (0.66)."),
        ("Cleaning Impact", "The raw dataset had 898,429 missing values across 46 features; after cleaning, the working dataset has zero missing values across 30 features."),
    ]
    for title, body in insights:
        st.markdown(f"""
        <div class="insight-card"><b>{title}:</b> {body}</div>
        """, unsafe_allow_html=True)
