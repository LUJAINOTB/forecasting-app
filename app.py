import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Adaptive Forecasting Platform", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: #0d0f14; color: #e0e6f0; }
.main { background-color: #0d0f14; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #00d4aa; }
.stButton>button { background: #00d4aa; color: #0d0f14; font-family: 'IBM Plex Mono', monospace; font-weight: 600; border: none; border-radius: 4px; padding: 0.5rem 1.5rem; }
.stButton>button:hover { background: #00f0c4; }
.metric-card { background: #161b24; border: 1px solid #00d4aa22; border-radius: 8px; padding: 1.2rem 1.5rem; margin-bottom: 0.8rem; }
.metric-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #8899aa; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #00d4aa; }
.score-card { background: #161b24; border: 1px solid #00d4aa22; border-radius: 8px; padding: 1.5rem; text-align: center; margin-bottom: 1rem; }
.score-value { font-family: 'IBM Plex Mono', monospace; font-size: 3rem; font-weight: 600; }
.section-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #8899aa; text-transform: uppercase; letter-spacing: 0.15em; border-bottom: 1px solid #00d4aa22; padding-bottom: 0.5rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

def generate_demo_data():
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=104, freq="W")
    rows = []
    for d in dates:
        base_load = 500
        seasonal = 1 + 0.35 * np.sin(2 * np.pi * d.dayofyear / 365)
        cooling_load = int(base_load * seasonal + np.random.normal(0, 25))
        energy_cons = int(cooling_load * np.random.uniform(1.1, 1.4))
        rows.append({"Timestamp": d, "Cooling_Load_kW": cooling_load,
                     "Energy_Consumption_kWh": energy_cons,
                     "Operational_Status": np.random.choice(["Optimal", "Maintenance Required"], p=[0.9, 0.1])})
    return pd.DataFrame(rows)

def compute_quality_score(df):
    score = 100
    missing = int(df.isnull().sum().sum())
    dupes = int(df.duplicated().sum())
    rows, _ = df.shape
    if missing > 0: score -= 20
    if dupes > 0: score -= 10
    if rows < 50: score -= 20
    return score, missing, dupes

st.markdown("# 📊 Adaptive Forecasting SaaS Platform")
st.markdown("<div class='section-title'>Step 01 — Data Ingestion & Mapping</div>", unsafe_allow_html=True)
st.markdown("Upload any historical dataset. The platform automatically adapts to your file structure.")
st.divider()

if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = generate_demo_data()
    st.session_state["data_source"] = "demo"

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
col_btn1, _, __ = st.columns([1.5, 1.5, 4])
with col_btn1:
    if st.button("🔄 Reset to Demo Data"):
        st.session_state["raw_df"] = generate_demo_data()
        st.session_state["data_source"] = "demo"
        st.rerun()

if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        st.session_state["raw_df"] = df_upload
        st.session_state["data_source"] = uploaded_file.name
        st.success(f"✅ File uploaded successfully: **{uploaded_file.name}**")
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")

df = st.session_state["raw_df"]
source = st.session_state["data_source"]
if source == "demo":
    st.info("🧪 Running on Demo HVAC Dataset. Upload your data above to scale.")

with st.sidebar:
    st.markdown("### ⚙️ Dynamic Platform Mapping")
    st.markdown("Map your dataset columns to the ML core.")
    st.divider()
    date_column = st.selectbox("Select Time/Date Column:", df.columns, index=0)
    target_column = st.selectbox("Select Target Variable:", df.columns, index=1 if len(df.columns) > 1 else 0)
    st.success("Target Engine Linked! 🚀")

st.markdown("<div class='section-title'>Dataset Overview</div>", unsafe_allow_html=True)
score, missing, dupes = compute_quality_score(df)
rows, cols = df.shape
mem_kb = df.memory_usage(deep=True).sum() / 1024
c1, c2, c3, c4, c5 = st.columns(5)
for col_ui, label, value in zip([c1,c2,c3,c4,c5],
    ["Total Rows","Columns","Missing Values","Duplicate Rows","Memory"],
    [f"{rows:,}", cols, f"{missing:,}", f"{dupes:,}", f"{mem_kb:.1f} KB"]):
    with col_ui:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)

st.divider()
st.markdown("<div class='section-title'>Data Quality & Core Readiness</div>", unsafe_allow_html=True)
score_color = "#00d4aa" if score >= 80 else "#ffaa00" if score >= 60 else "#ff6666"
score_label = "Excellent ✅ Ready for ML" if score >= 80 else "Fair ⚠️ Requires Cleaning" if score >= 60 else "Poor ❌"
sc1, sc2 = st.columns([1, 2])
with sc1:
    st.markdown(f"<div class='score-card'><div class='metric-label'>Quality Score</div><div class='score-value' style='color:{score_color}'>{score}%</div><div style='color:{score_color}; font-family:IBM Plex Mono,monospace; font-size:0.85rem'>{score_label}</div></div>", unsafe_allow_html=True)
with sc2:
    st.progress(score / 100)
    try:
        pd.to_datetime(df[date_column])
        st.success(f"🎯 Time Mapping Verified: **{date_column}** is parseable as Datetime.")
    except Exception:
        st.error(f"❌ Time Mapping Error: **{date_column}** cannot be parsed.")
    if missing > 0: st.warning(f"⚠️ {missing} missing value(s) detected.")
    if dupes > 0: st.warning(f"⚠️ {dupes} duplicate row(s) detected.")
    if rows < 50: st.warning(f"⚠️ Dataset too small ({rows} rows).")

st.divider()
st.markdown("<div class='section-title'>Data Preview & Live Profile</div>", unsafe_allow_html=True)
n_rows = st.slider("Rows to preview", 5, min(100, len(df)), 10, step=5)
st.dataframe(df.head(n_rows), use_container_width=True)
st.markdown("### 📈 Mathematical Distribution")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if numeric_cols:
    st.dataframe(df[numeric_cols].describe().T.round(2), use_container_width=True)

st.divider()
st.markdown("<div class='section-title'>Time Series Visualization</div>", unsafe_allow_html=True)
try:
    df_plot = df.copy()
    df_plot[date_column] = pd.to_datetime(df_plot[date_column])
    df_plot = df_plot.sort_values(date_column)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot[date_column], y=df_plot[target_column],
        mode="lines", name=target_column,
        line=dict(color="#00d4aa", width=2),
        fill="tozeroy", fillcolor="rgba(0, 212, 170, 0.08)"
    ))
    fig.update_layout(
        plot_bgcolor="#0d0f14", paper_bgcolor="#0d0f14",
        font=dict(family="IBM Plex Mono", color="#e0e6f0"),
        xaxis=dict(showgrid=False, color="#8899aa"),
        yaxis=dict(showgrid=True, gridcolor="#1e2530", color="#8899aa"),
        margin=dict(l=20, r=20, t=30, b=20), height=350,
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"❌ Chart Error: {e}")

st.divider()
st.markdown("<div class='section-title'>Forecast Configuration</div>", unsafe_allow_html=True)
cfg1, cfg2, cfg3 = st.columns(3)
with cfg1:
    horizon = st.selectbox("Forecast Horizon", ["7 Days","14 Days","30 Days","60 Days","90 Days"], index=2)
with cfg2:
    model_type = st.selectbox("Forecasting Model", ["Auto (Recommended)","Prophet","LSTM Neural Network","XGBoost","ARIMA"], index=0)
with cfg3:
    confidence = st.slider("Confidence Interval %", 80, 99, 95, step=1)

st.markdown("<br>", unsafe_allow_html=True)
sum1, sum2, sum3, sum4 = st.columns(4)
for col_ui, label, value in zip([sum1,sum2,sum3,sum4],
    ["Date Column","Target Variable","Horizon","Model"],
    [date_column, target_column, horizon, model_type]):
    with col_ui:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value' style='font-size:1rem'>{value}</div></div>", unsafe_allow_html=True)

st.divider()
col_go, _ = st.columns([2, 5])
with col_go:
    if st.button("🚀 Proceed to Forecasting Engine →"):
        st.session_state["forecast_config"] = {
            "date_column": date_column,
            "target_column": target_column,
            "horizon": horizon,
            "model_type": model_type,
            "confidence": confidence,
        }
        st.success("✅ Configuration saved!")
        st.balloons()
