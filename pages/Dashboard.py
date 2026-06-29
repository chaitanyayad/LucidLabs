import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

_ICON = Path(__file__).parent.parent / "assets" / "lucidlabs-icon.svg"
st.set_page_config(page_title="LucidLabss", page_icon=str(_ICON), layout="wide")

# ── Plotly dark template ───────────────────────────────────────────────────────
pio.templates["lucidlab"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(13,17,23,0.0)",
        plot_bgcolor="rgba(13,17,23,0.0)",
        font=dict(color="#CBD5E1", family="Exo 2, Inter, sans-serif", size=12),
        colorway=["#7C3AED","#06B6D4","#10B981","#F59E0B","#EF4444","#A78BFA","#38BDF8"],
        xaxis=dict(
            gridcolor="rgba(99,102,241,0.1)",
            linecolor="rgba(99,102,241,0.25)",
            tickcolor="rgba(99,102,241,0.3)",
            zerolinecolor="rgba(99,102,241,0.15)",
        ),
        yaxis=dict(
            gridcolor="rgba(99,102,241,0.1)",
            linecolor="rgba(99,102,241,0.25)",
            tickcolor="rgba(99,102,241,0.3)",
            zerolinecolor="rgba(99,102,241,0.15)",
        ),
        title=dict(font=dict(color="#A5B4FC", size=13, family="Exo 2, sans-serif")),
        legend=dict(bgcolor="rgba(13,17,23,0.6)", bordercolor="rgba(99,102,241,0.2)", borderwidth=1),
        hoverlabel=dict(bgcolor="#0D1117", bordercolor="#6366F1", font=dict(color="#F1F5F9")),
    )
)
pio.templates.default = "lucidlab"

# ── CSS ────────────────────────────────────────────────────────────────────────
css_path = Path(__file__).parent.parent / "assets" / "styles.css"
_extra = ""
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        _extra = f.read()
st.markdown(f"""<style>
{_extra}
[data-testid='stSidebarNav']{{display:none!important}}
/* Suppress rerun blur overlay */
[data-testid="stAppViewContainer"] > div[class*="withScreencast"] {{
    opacity: 1 !important;
}}
/* Matplotlib figures */
[data-testid="stPyplotFigure"] {{
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(99,102,241,0.18) !important;
}}

/* Step progress bar animations */
@keyframes travelLine{{0%{{transform:translateX(-110%)}}100%{{transform:translateX(210%)}}}}
@keyframes spGlow{{0%,100%{{box-shadow:0 0 10px rgba(124,58,237,0.45)}}50%{{box-shadow:0 0 24px rgba(124,58,237,0.95),0 0 44px rgba(99,102,241,0.4)}}}}
.sp-done{{flex:1;height:2px;background:linear-gradient(90deg,#059669,#10B981,#06B6D4);border-radius:2px;}}
.sp-active{{flex:1;height:2px;background:rgba(99,102,241,0.14);border-radius:2px;position:relative;overflow:hidden;}}
.sp-active::after{{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent 0%,#7C3AED 35%,#06B6D4 65%,transparent 100%);animation:travelLine 1.3s ease-in-out infinite;}}
.sp-future{{flex:1;height:2px;background:rgba(255,255,255,0.06);border-radius:2px;}}
.sp-circ-active{{animation:spGlow 2s ease-in-out infinite;}}

/* Animated gradient blobs */
@keyframes _b1{{0%,100%{{transform:translate(0,0) scale(1)}}33%{{transform:translate(300px,-200px) scale(1.25)}}66%{{transform:translate(-180px,260px) scale(.8)}}}}
@keyframes _b2{{0%,100%{{transform:translate(0,0) scale(1)}}33%{{transform:translate(-240px,180px) scale(1.2)}}66%{{transform:translate(200px,-250px) scale(1.25)}}}}
@keyframes _b3{{0%,100%{{transform:translate(-50%,-50%) scale(1)}}33%{{transform:translate(calc(-50% + 320px),calc(-50% + 220px)) scale(.75)}}66%{{transform:translate(calc(-50% - 240px),calc(-50% - 180px)) scale(1.2)}}}}
.grad-blob{{position:fixed;border-radius:50%;filter:blur(80px);pointer-events:none;z-index:0;}}
.gb1{{width:750px;height:650px;background:radial-gradient(circle,rgba(124,58,237,.22) 0%,transparent 65%);top:-200px;left:-200px;animation:_b1 9s ease-in-out infinite;}}
.gb2{{width:650px;height:550px;background:radial-gradient(circle,rgba(6,182,212,.16) 0%,transparent 65%);bottom:-160px;right:-160px;animation:_b2 11s ease-in-out infinite;}}
.gb3{{width:550px;height:480px;background:radial-gradient(circle,rgba(99,102,241,.12) 0%,transparent 65%);top:50%;left:50%;transform:translate(-50%,-50%);animation:_b3 13s ease-in-out infinite;}}
</style>
<div class="grad-blob gb1"></div>
<div class="grad-blob gb2"></div>
<div class="grad-blob gb3"></div>""", unsafe_allow_html=True)

# ── Matplotlib dark style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor":   "#0D1117",
    "axes.edgecolor":   "#1E2540",
    "axes.labelcolor":  "#94A3B8",
    "xtick.color":      "#64748B",
    "ytick.color":      "#64748B",
    "text.color":       "#CBD5E1",
    "grid.color":       "#1A1F35",
    "grid.linestyle":   "--",
    "grid.linewidth":   0.5,
    "figure.dpi":       110,
})

# ── Session state (init once) ──────────────────────────────────────────────────
DEFAULTS = {
    "task": None, "data": None, "target_col": None, "dataset_source": None,
    "algo": None, "model": None, "y_pred": None,
    "cleaned_df": None, "prep_log": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

CLF_DATASETS = {
    "Iris":                    {"id": 53,  "target": "class",     "desc": "4 features · 3 classes · 150 rows"},
    "Heart Disease":           {"id": 45,  "target": "num",       "desc": "13 features · binary · 303 rows"},
    "Breast Cancer Wisconsin": {"id": 17,  "target": "Diagnosis", "desc": "30 features · binary · 569 rows"},
    "Banknote Authentication": {"id": 267, "target": "class",     "desc": "4 features · binary · 1 372 rows"},
}
REG_DATASETS = {
    "Auto MPG":                      {"id": 9,   "target": "mpg",    "desc": "7 features · 398 rows"},
    "Concrete Compressive Strength": {"id": 165, "target": "Concrete compressive strength(MPa, megapascals)",
                                                                      "desc": "8 features · 1 030 rows"},
    "Wine Quality — Red":            {"id": 186, "target": "quality","desc": "11 features · 1 599 rows"},
    "Abalone":                       {"id": 1,   "target": "Rings",  "desc": "8 features · 4 177 rows"},
}
CLF_MODELS = ["Logistic Regression","Random Forest","Gradient Boosting","AdaBoost",
              "Support Vector Machine","K-Nearest Neighbors","Decision Tree","Naive Bayes","MLP Neural Network"]
REG_MODELS = ["Linear Regression","Gradient Boosting Regressor","AdaBoost Regressor",
              "Support Vector Regressor","Random Forest (Regressor)","MLP Neural Network"]

# ── Cached helpers ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading from UCI…")
def fetch_uci(dataset_id: int) -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=dataset_id)
    parts = [p for p in (ds.data.features, ds.data.targets) if p is not None]
    df = pd.concat(parts, axis=1)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def _col_info(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Column":   df.columns,
        "Dtype":    df.dtypes.astype(str).values,
        "Non-null": df.notnull().sum().values,
        "Unique":   df.nunique().values,
    })

@st.cache_data
def _describe(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").round(3)

@st.cache_data
def _skew_kurt(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    return pd.DataFrame({
        "Feature":  num.columns,
        "Skewness": num.skew().round(3).values,
        "Kurtosis": num.kurt().round(3).values,
    })

@st.cache_data
def _missing_df(df: pd.DataFrame) -> pd.DataFrame:
    m = df.isnull().sum()
    m = m[m > 0]
    return pd.DataFrame({"Column": m.index, "Missing": m.values,
                          "Pct %": (m / len(df) * 100).round(2).values})

@st.cache_data
def _outlier_df(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    rows = []
    for c in num.columns:
        q1, q3 = df[c].quantile(.25), df[c].quantile(.75); iqr = q3 - q1
        n = int(((df[c] < q1 - 1.5*iqr) | (df[c] > q3 + 1.5*iqr)).sum())
        rows.append({"Feature": c, "Outliers": n, "Pct %": round(n/len(df)*100, 2)})
    return pd.DataFrame(rows)

@st.cache_data
def _corr(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="number").corr()

@st.cache_data
def _feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        cd = df[c]; is_num = pd.api.types.is_numeric_dtype(cd)
        rows.append({"Feature": c, "Dtype": str(cd.dtype),
                     "Non-null": int(cd.notnull().sum()), "Missing": int(cd.isnull().sum()),
                     "Unique": int(cd.nunique()),
                     "Mean":     round(cd.mean(), 3) if is_num else "—",
                     "Std":      round(cd.std(),  3) if is_num else "—",
                     "Min":      round(cd.min(),  3) if is_num else "—",
                     "Max":      round(cd.max(),  3) if is_num else "—",
                     "Skewness": round(cd.skew(), 3) if is_num else "—"})
    return pd.DataFrame(rows)

# ── UI helpers ─────────────────────────────────────────────────────────────────
def step_badge(n, label, done=False):
    if done:
        circle_bg = "linear-gradient(135deg,#059669,#10B981)"
        icon       = "&#10003;"
        label_col  = "#6EE7B7"
        border_col = "rgba(16,185,129,0.25)"
    else:
        circle_bg = "linear-gradient(135deg,#7C3AED,#6366F1)"
        icon       = str(n)
        label_col  = "#C4B5FD"
        border_col = "rgba(99,102,241,0.25)"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:14px;margin:1.75rem 0 0.85rem;'
        f'padding:0.9rem 1.25rem;'
        f'background:rgba(99,102,241,0.06);'
        f'border:1px solid {border_col};'
        f'border-radius:14px;">'
        f'<div style="width:38px;height:38px;border-radius:50%;background:{circle_bg};display:flex;'
        f'align-items:center;justify-content:center;color:#fff;font-weight:800;font-size:.9rem;'
        f'flex-shrink:0;box-shadow:0 3px 14px rgba(124,58,237,0.45);font-family:Exo 2,sans-serif">{icon}</div>'
        f'<span style="font-size:1.05rem;font-weight:700;color:{label_col};'
        f'font-family:Exo 2,sans-serif;letter-spacing:0.1px">{label}</span>'
        f'</div>',
        unsafe_allow_html=True)


def step_progress(current):
    steps = [("Task","1"),("Dataset","2"),("EDA","3"),("Preprocess","4"),("Model","5"),("Train","6")]
    n = len(steps)

    # Row 1: circles + connector lines — align-items:center puts 2px lines at circle midpoint
    circles = []
    for i, (lbl, num) in enumerate(steps, 1):
        if i < current:
            c_bg = "linear-gradient(135deg,#059669,#10B981)"; c_col = "#fff"; ico = "✓"; cls = ""
        elif i == current:
            c_bg = "linear-gradient(135deg,#7C3AED,#6366F1)"; c_col = "#fff"; ico = num;  cls = "sp-circ-active"
        else:
            c_bg = "rgba(255,255,255,0.06)";                   c_col = "#475569"; ico = num;  cls = ""
        circles.append(
            f'<div class="{cls}" style="flex-shrink:0;width:32px;height:32px;border-radius:50%;'
            f'background:{c_bg};display:flex;align-items:center;justify-content:center;'
            f'color:{c_col};font-weight:700;font-size:0.75rem;font-family:Exo 2,sans-serif">{ico}</div>'
        )
        if i < n:
            seg = "sp-done" if i < current else ("sp-active" if i == current else "sp-future")
            circles.append(f'<div class="{seg}"></div>')

    # Row 2: labels, each taking equal space to sit under its circle
    labels = []
    for i, (lbl, _) in enumerate(steps, 1):
        l_col = "#6EE7B7" if i < current else ("#C4B5FD" if i == current else "#334155")
        labels.append(
            f'<div style="flex:1;text-align:center;font-size:0.6rem;color:{l_col};'
            f'text-transform:uppercase;letter-spacing:0.9px;font-weight:600;'
            f'font-family:Exo 2,sans-serif">{lbl}</div>'
        )

    html = (
        '<div style="padding:0.85rem 1.25rem;margin-bottom:0.5rem;'
        'background:rgba(99,102,241,0.04);border:1px solid rgba(99,102,241,0.14);border-radius:14px;">'
        '<div style="display:flex;align-items:center;gap:0;margin-bottom:7px">'
        + "".join(circles) +
        '</div>'
        '<div style="display:flex;gap:0">'
        + "".join(labels) +
        '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

# ── on_change callbacks (single-rerun pattern) ────────────────────────────────
def _on_task_change():
    v = st.session_state._task_sel
    if v == "— select —" or v == st.session_state.task:
        return
    for k in DEFAULTS:
        st.session_state[k] = None
    st.session_state.task = v
    # clear stale algo widget value so the new task's model list renders fully
    for wkey in ("_algo_sel", "ds_sel"):
        if wkey in st.session_state:
            del st.session_state[wkey]

def _on_algo_change():
    v = st.session_state._algo_sel
    if v == "— select —" or v == st.session_state.algo:
        return
    st.session_state.algo   = v
    st.session_state.model  = None
    st.session_state.y_pred = None

# ── Header ─────────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([6, 1])
hcol1.markdown("""
<div style="display:flex;align-items:center;gap:14px;padding:0.25rem 0 0.75rem">
  <div style="width:42px;height:42px;border-radius:12px;flex-shrink:0;
    background:linear-gradient(135deg,#7C3AED,#06B6D4);
    display:flex;align-items:center;justify-content:center;
    box-shadow:0 4px 16px rgba(124,58,237,0.4)">
    <svg fill="#ffffff" width="22" height="22" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
      <path d="M27,24a2.9609,2.9609,0,0,0-1.2854.3008L21.4141,20H18v2h2.5859l3.7146,3.7148A2.9665,2.9665,0,0,0,24,27a3,3,0,1,0,3-3Zm0,4a1,1,0,1,1,1-1A1.0009,1.0009,0,0,1,27,28Z"/>
      <path d="M27,13a2.9948,2.9948,0,0,0-2.8157,2H18v2h6.1843A2.9947,2.9947,0,1,0,27,13Zm0,4a1,1,0,1,1,1-1A1.0009,1.0009,0,0,1,27,17Z"/>
      <path d="M27,2a3.0033,3.0033,0,0,0-3,3,2.9657,2.9657,0,0,0,.3481,1.373L20.5957,10H18v2h3.4043l4.3989-4.2524A2.9987,2.9987,0,1,0,27,2Zm0,4a1,1,0,1,1,1-1A1.0009,1.0009,0,0,1,27,6Z"/>
      <path d="M18,6h2V4H18a3.9756,3.9756,0,0,0-3,1.3823A3.9756,3.9756,0,0,0,12,4H11a9.01,9.01,0,0,0-9,9v6a9.01,9.01,0,0,0,9,9h1a3.9756,3.9756,0,0,0,3-1.3823A3.9756,3.9756,0,0,0,18,28h2V26H18a2.0023,2.0023,0,0,1-2-2V8A2.0023,2.0023,0,0,1,18,6ZM12,26H11a7.0047,7.0047,0,0,1-6.92-6H6V18H4V14H7a3.0033,3.0033,0,0,0,3-3V9H8v2a1.0009,1.0009,0,0,1-1,1H4.08A7.0047,7.0047,0,0,1,11,6h1a2.0023,2.0023,0,0,1,2,2v4H12v2h2v4H12a3.0033,3.0033,0,0,0-3,3v2h2V21a1.0009,1.0009,0,0,1,1-1h2v4A2.0023,2.0023,0,0,1,12,26Z"/>
    </svg>
  </div>
  <div>
    <div style="font-size:1.35rem;font-weight:800;color:#C4B5FD;
      font-family:'Exo 2',sans-serif;letter-spacing:-0.3px;line-height:1.1">
      LucidLabs
    </div>
    <div style="font-size:0.7rem;color:#334155;font-family:'Roboto Mono',monospace;
      letter-spacing:0.5px;margin-top:2px">
      scikit-learn &middot; streamlit &middot; plotly
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
with hcol2:
    if st.button("← Home", key="btn_home"):
        st.switch_page("main.py")

# ── Step progress indicator ─────────────────────────────────────────────────────
def _current_step():
    if st.session_state.get("algo"): return 6
    if st.session_state.get("cleaned_df") is not None: return 5
    if st.session_state.get("data") is not None: return 4
    if st.session_state.get("task"): return 2
    return 1

step_progress(_current_step())
st.markdown('<hr style="border:none;border-top:1px solid rgba(99,102,241,0.18);margin:0 0 0.5rem">', unsafe_allow_html=True)

# ══ STEP 1 — Task ════════════════════════════════════════════════════════════
step_badge(1, "Choose task type", done=st.session_state.task is not None)
opts  = ["— select —", "Classification", "Regression"]
t_idx = opts.index(st.session_state.task) if st.session_state.task in opts else 0
st.selectbox("What kind of problem are you solving?", opts, index=t_idx,
             key="_task_sel", on_change=_on_task_change)
if not st.session_state.task:
    st.stop()

st.divider()
task      = st.session_state.task
catalogue = CLF_DATASETS if task == "Classification" else REG_DATASETS

# ══ STEP 2 — Dataset ══════════════════════════════════════════════════════════
step_badge(2, "Select dataset", done=st.session_state.data is not None)
tab_uci, tab_up = st.tabs([f"UCI {task} datasets", "Upload CSV / Excel"])

with tab_uci:
    ds_opts = ["— select —"] + list(catalogue.keys())
    cur     = ds_opts.index(st.session_state.dataset_source) \
              if st.session_state.dataset_source in catalogue else 0
    chosen  = st.selectbox("Dataset", ds_opts, index=cur, key="ds_sel")
    if chosen != "— select —":
        st.caption(catalogue[chosen]["desc"])
        if st.button("Load dataset", key="btn_uci"):
            with st.spinner(f"Fetching {chosen}…"):
                try:
                    meta   = catalogue[chosen]
                    df_raw = fetch_uci(meta["id"])
                    target = meta["target"]
                    # if exact name missing, find closest stripped match
                    if target not in df_raw.columns:
                        match = next(
                            (c for c in df_raw.columns if c.strip() == target.strip()),
                            None
                        )
                        if match is None:
                            st.error(f"Could not find target column '{target}' in dataset. "
                                     f"Available columns: {df_raw.columns.tolist()}")
                            st.stop()
                        target = match
                    if isinstance(df_raw[target], pd.DataFrame):
                        df_raw[target] = df_raw[target].iloc[:, 0]
                    if meta["id"] == 45:
                        df_raw[target] = (df_raw[target] > 0).astype(int)
                    st.session_state.data           = df_raw
                    st.session_state.target_col     = target
                    st.session_state.dataset_source = chosen
                    st.session_state.cleaned_df     = None
                    st.session_state.prep_log       = None
                    st.session_state.algo           = None
                    st.session_state.y_pred         = None
                except Exception as e:
                    st.error(f"Failed: {e}")

with tab_up:
    up = st.file_uploader("Upload", type=["csv","xlsx"], key="upload")
    if up:
        try:
            df_up = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
            st.dataframe(df_up.head(3), use_container_width=True)
            tc = st.selectbox("Target column", df_up.columns.tolist(), key="tc_sel")
            if st.button("Confirm", key="btn_upload"):
                st.session_state.data           = df_up
                st.session_state.target_col     = tc
                st.session_state.dataset_source = up.name
                st.session_state.cleaned_df     = None
                st.session_state.prep_log       = None
                st.session_state.algo           = None
                st.session_state.y_pred         = None
        except Exception as e:
            st.error(str(e))

if st.session_state.data is None:
    st.stop()

df         = st.session_state.data
target_col = st.session_state.target_col

m1, m2, m3, m4 = st.columns(4)
m1.metric("Dataset",  st.session_state.dataset_source)
m2.metric("Rows",     df.shape[0])
m3.metric("Features", df.shape[1] - 1)
m4.metric("Target",   target_col)
st.divider()

# ══ STEP 3 — Full EDA ════════════════════════════════════════════════════════
step_badge(3, "Exploratory Data Analysis")

(t_ov, t_st, t_mv, t_di, t_bx, t_co, t_tg, t_fs) = st.tabs([
    "Overview", "Statistics", "Missing Values",
    "Distributions", "Box Plots", "Correlation", "Target", "Per-Feature Summary"])

with t_ov:
    c1, c2 = st.columns(2)
    c1.subheader("First 10 rows"); c1.dataframe(df.head(10), use_container_width=True)
    c2.subheader("Last 10 rows");  c2.dataframe(df.tail(10), use_container_width=True)
    st.subheader("Column info")
    st.dataframe(_col_info(df), use_container_width=True)
    n_num = len(df.select_dtypes(include="number").columns)
    n_cat = len(df.select_dtypes(exclude="number").columns)
    st.plotly_chart(px.pie(values=[n_num, n_cat], names=["Numeric","Categorical"],
        title="Feature type breakdown", color_discrete_sequence=["#7C3AED","#06B6D4"]),
        use_container_width=True)

with t_st:
    st.subheader("Summary statistics")
    st.dataframe(_describe(df), use_container_width=True)
    sk = _skew_kurt(df)
    if not sk.empty:
        st.subheader("Skewness & Kurtosis")
        st.dataframe(sk, use_container_width=True)
        st.plotly_chart(px.bar(sk, x="Feature", y="Skewness", title="Skewness per feature",
            color="Skewness", color_continuous_scale="RdBu"), use_container_width=True)

with t_mv:
    miss_df = _missing_df(df)
    if miss_df.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(miss_df, use_container_width=True)
        st.plotly_chart(px.bar(miss_df, x="Column", y="Pct %",
            title="Missing value % per column", color="Pct %", color_continuous_scale="Reds"),
            use_container_width=True)
        st.subheader("Missing heatmap (first 100 rows)")
        fig, ax = plt.subplots(figsize=(max(6, df.shape[1] * .4), 4), facecolor="#0D1117")
        ax.set_facecolor("#0D1117")
        sns.heatmap(df.head(100).isnull(), cbar=False, cmap="plasma", ax=ax)
        ax.set_title("Yellow = missing", color="#A5B4FC")
        ax.tick_params(colors="#64748B")
        fig.tight_layout()
        st.pyplot(fig); plt.close()

with t_di:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        sel = st.selectbox("Feature", num_cols, key="dist_sel")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x=sel, nbins=30, color_discrete_sequence=["#7C3AED"],
                               title=f"Histogram — {sel}")
            fig.update_layout(bargap=0.05); st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Violin(y=df[sel], box_visible=True, meanline_visible=True,
                fillcolor="#7C3AED", opacity=0.75, line_color="#06B6D4", name=sel))
            fig2.update_layout(title=f"Violin — {sel}", showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        with st.expander("All numeric features — histogram grid"):
            nc = min(3, len(num_cols)); nr = int(np.ceil(len(num_cols) / nc))
            fig3, axes = plt.subplots(nr, nc, figsize=(nc*4, nr*3), facecolor="#0D1117")
            axes = np.array(axes).flatten()
            for i, col in enumerate(num_cols):
                axes[i].set_facecolor("#0D1117")
                axes[i].hist(df[col].dropna(), bins=25, color="#7C3AED",
                             edgecolor="#06B6D4", linewidth=.4, alpha=0.85)
                axes[i].set_title(col, fontsize=8, color="#A5B4FC")
                axes[i].tick_params(labelsize=6, colors="#64748B")
                for spine in axes[i].spines.values():
                    spine.set_edgecolor((99/255, 102/255, 241/255, 0.2))
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout(); st.pyplot(fig3); plt.close()

with t_bx:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        sel2 = st.selectbox("Feature", num_cols, key="box_sel")
        grp  = st.checkbox("Group by target", value=True)
        if grp and df[target_col].nunique() <= 10:
            fig = px.box(df, x=target_col, y=sel2, color=target_col,
                color_discrete_sequence=px.colors.sequential.Purples_r,
                title=f"{sel2} by {target_col}")
        else:
            fig = px.box(df, y=sel2, color_discrete_sequence=["#7C3AED"],
                         title=f"Box plot — {sel2}")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Outlier summary (IQR method)")
        st.dataframe(_outlier_df(df), use_container_width=True)

with t_co:
    corr = _corr(df)
    if corr.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(max(6, len(corr)*.65), max(5, len(corr)*.55)), facecolor="#0D1117")
        ax.set_facecolor("#0D1117")
        sns.heatmap(corr, annot=len(corr) <= 15, fmt=".2f", cmap="coolwarm",
                    ax=ax, linewidths=.3, square=True,
                    linecolor="#0D1117", annot_kws={"size": 8, "color": "#F1F5F9"})
        ax.tick_params(colors="#94A3B8", labelsize=8)
        ax.set_title("Correlation Matrix", color="#A5B4FC", pad=12)
        fig.tight_layout()
        st.pyplot(fig); plt.close()
        if target_col in corr:
            top = corr[target_col].drop(target_col).abs().sort_values(ascending=False)
            st.plotly_chart(px.bar(x=top.index, y=top.values,
                labels={"x":"Feature","y":f"|corr with {target_col}|"},
                title="Feature correlation with target",
                color=top.values, color_continuous_scale="Purples_r"),
                use_container_width=True)

with t_tg:
    c1, c2 = st.columns(2)
    with c1:
        if df[target_col].nunique() <= 15:
            vc = df[target_col].value_counts().reset_index(); vc.columns=[target_col,"count"]
            st.plotly_chart(px.bar(vc, x=target_col, y="count",
                color_discrete_sequence=["#A78BFA"], title="Class distribution"),
                use_container_width=True)
            if task == "Classification":
                maj = vc["count"].max() / vc["count"].sum()
                if maj > 0.7: st.warning(f"Class imbalance: majority class is {maj:.0%}")
        else:
            st.plotly_chart(px.histogram(df, x=target_col, nbins=30,
                color_discrete_sequence=["#A78BFA"], title="Target distribution"),
                use_container_width=True)
    with c2:
        st.write(df[target_col].describe())
        if task == "Regression":
            figv = go.Figure()
            figv.add_trace(go.Violin(y=df[target_col], box_visible=True, meanline_visible=True,
                fillcolor="#A78BFA", opacity=0.75, line_color="#7C3AED", name=target_col))
            st.plotly_chart(figv, use_container_width=True)

with t_fs:
    st.dataframe(_feature_summary(df), use_container_width=True)

st.divider()

# ══ STEP 4 — Cleaning & Preprocessing ════════════════════════════════════════
step_badge(4, "Data Cleaning & Preprocessing", done=st.session_state.cleaned_df is not None)

# ── Dataset health analysis ─────────────────────────────────────────────────
_num_cols_h  = df.select_dtypes(include="number").columns.tolist()
_cat_cols_h  = [c for c in df.columns if df[c].dtype == object]
_const_cols  = [c for c in df.columns if df[c].nunique() <= 1]
_hicard_cols = [c for c in _cat_cols_h if df[c].nunique() > 20]
_neg_cols    = [c for c in _num_cols_h if (df[c] < 0).any()]
_miss_cols   = [c for c in df.columns if df[c].isnull().sum() > 0]
_enc_cols    = [c for c in df.columns if c != target_col and df[c].dtype == object]
_dup_cnt     = int(df.duplicated().sum())
_ws_cols     = [c for c in _cat_cols_h
                if df[c].dropna().apply(lambda x: x != x.strip()).any()]
_mixed_cols  = [c for c in _cat_cols_h
                if pd.to_numeric(df[c], errors="coerce").notna().sum()
                > 0.5 * df[c].notna().sum()]

h1, h2, h3, h4, h5, h6, h7, h8 = st.columns(8)
h1.metric("Missing cols",     len(_miss_cols))
h2.metric("Duplicate rows",   _dup_cnt)
h3.metric("Numeric cols",     len(_num_cols_h))
h4.metric("Categorical cols", len(_cat_cols_h))
h5.metric("Constant cols",    len(_const_cols))
h6.metric("High-cardinality", len(_hicard_cols))
h7.metric("With negatives",   len(_neg_cols))
h8.metric("Whitespace cols",  len(_ws_cols))

with st.expander("Dataset Health Report", expanded=True):
    ht1, ht2, ht3, ht4, ht5 = st.tabs(
        ["Summary", "Missing Values", "Constants & High-Cardinality", "Negatives", "Duplicates"])

    with ht1:
        health_rows = [
            {"Check": "Missing values",              "Status": "⚠️ Found" if _miss_cols   else "✅ None", "Count": len(_miss_cols)},
            {"Check": "Duplicate rows",              "Status": "⚠️ Found" if _dup_cnt      else "✅ None", "Count": _dup_cnt},
            {"Check": "Constant columns",            "Status": "⚠️ Found" if _const_cols  else "✅ None", "Count": len(_const_cols)},
            {"Check": "High-cardinality cols (>20)", "Status": "⚠️ Found" if _hicard_cols else "✅ None", "Count": len(_hicard_cols)},
            {"Check": "Columns with negatives",      "Status": "ℹ️ Found" if _neg_cols    else "✅ None", "Count": len(_neg_cols)},
            {"Check": "Whitespace in strings",       "Status": "⚠️ Found" if _ws_cols     else "✅ None", "Count": len(_ws_cols)},
            {"Check": "Mixed-type columns",          "Status": "⚠️ Found" if _mixed_cols  else "✅ None", "Count": len(_mixed_cols)},
        ]
        st.dataframe(pd.DataFrame(health_rows), use_container_width=True, hide_index=True)

    with ht2:
        if not _miss_cols:
            st.success("No missing values.")
        else:
            st.dataframe(pd.DataFrame({
                "Column":  _miss_cols,
                "Missing": [df[c].isnull().sum() for c in _miss_cols],
                "Pct %":   [round(df[c].isnull().sum() / len(df) * 100, 2) for c in _miss_cols],
            }), use_container_width=True, hide_index=True)

    with ht3:
        c1h, c2h = st.columns(2)
        with c1h:
            st.write("**Constant columns** (1 unique value)")
            if _const_cols:
                st.dataframe(pd.DataFrame({
                    "Column": _const_cols,
                    "Value":  [df[c].iloc[0] for c in _const_cols],
                }), use_container_width=True, hide_index=True)
            else:
                st.success("None found.")
        with c2h:
            st.write("**High-cardinality columns** (>20 unique)")
            if _hicard_cols:
                st.dataframe(pd.DataFrame({
                    "Column": _hicard_cols,
                    "Unique": [df[c].nunique() for c in _hicard_cols],
                }), use_container_width=True, hide_index=True)
            else:
                st.success("None found.")

    with ht4:
        if not _neg_cols:
            st.success("No columns with negative values.")
        else:
            st.dataframe(pd.DataFrame({
                "Column":         _neg_cols,
                "Min":            [round(df[c].min(), 3) for c in _neg_cols],
                "Negative count": [int((df[c] < 0).sum()) for c in _neg_cols],
            }), use_container_width=True, hide_index=True)

    with ht5:
        if _dup_cnt == 0:
            st.success("No duplicate rows.")
        else:
            st.warning(f"{_dup_cnt} duplicate rows found. Preview (up to 5):")
            st.dataframe(df[df.duplicated(keep=False)].head(5), use_container_width=True)

# ── Cleaning options ───────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("**Data Fixes**")
    do_whitespace = st.checkbox("Strip whitespace from string columns",
                                value=bool(_ws_cols), key="chk_ws")
    do_mixed      = st.checkbox("Convert mixed-type columns to numeric",
                                value=bool(_mixed_cols), key="chk_mixed")
    do_dedup      = st.checkbox(f"Remove duplicate rows  ({_dup_cnt} found)",
                                value=bool(_dup_cnt), key="chk_dedup")
    impute_strategy = st.selectbox("Missing value strategy",
        ["Fill with median", "Fill with mean", "Fill with mode", "Drop rows", "Leave as-is"])
    do_const  = st.checkbox(f"Remove constant columns  ({len(_const_cols)} found)",
                            value=bool(_const_cols), key="chk_const")
    do_hicard = st.checkbox(f"Remove high-cardinality columns  ({len(_hicard_cols)} found)",
                            value=False, key="chk_hicard")
    hicard_thresh = 20
    if do_hicard:
        hicard_thresh = st.slider("Cardinality threshold", 5, 100, 20, key="sl_hicard")

with col_r:
    st.markdown("**Transformations**")
    outlier_mode = st.selectbox("Outlier removal (IQR)",
        ["None", "All numerical columns", "Selected columns"])
    outlier_cols_sel = []
    if outlier_mode == "Selected columns":
        outlier_cols_sel = st.multiselect(
            "Columns for outlier removal",
            [c for c in _num_cols_h if c != target_col])

    encode_strategy = st.selectbox("Categorical encoding",
        ["Label Encoding", "One-Hot Encoding"])
    scale_strategy  = st.selectbox("Feature scaling",
        ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])
    if scale_strategy != "None":
        st.caption("Applied before train/test split — uncheck 'Standardise features' in Step 6 to avoid double-scaling.")

if st.button("Apply Preprocessing", type="primary"):
    log  = []
    df_p = df.copy()

    # 1 — Whitespace
    if do_whitespace:
        n_fixed = 0
        for c in df_p.select_dtypes(include="object").columns:
            before = df_p[c].copy()
            df_p[c] = df_p[c].str.strip()
            n_fixed += int((before != df_p[c]).sum())
        if n_fixed:
            log.append(f"Stripped whitespace from {n_fixed} cell(s)")

    # 2 — Mixed dtype
    if do_mixed:
        converted = []
        for c in list(df_p.select_dtypes(include="object").columns):
            conv = pd.to_numeric(df_p[c], errors="coerce")
            if conv.notna().sum() > 0.5 * df_p[c].notna().sum():
                df_p[c] = conv; converted.append(c)
        if converted:
            log.append(f"Converted mixed-type to numeric: {', '.join(converted)}")

    # 3 — Constant columns
    if do_const:
        drop_c = [c for c in df_p.columns if df_p[c].nunique() <= 1 and c != target_col]
        if drop_c:
            df_p = df_p.drop(columns=drop_c)
            log.append(f"Removed constant column(s): {', '.join(drop_c)}")

    # 4 — High-cardinality
    if do_hicard:
        drop_hc = [c for c in df_p.columns
                   if df_p[c].dtype == object
                   and df_p[c].nunique() > hicard_thresh
                   and c != target_col]
        if drop_hc:
            df_p = df_p.drop(columns=drop_hc)
            log.append(f"Removed high-cardinality column(s) (>{hicard_thresh} unique): {', '.join(drop_hc)}")

    # 5 — Duplicates
    if do_dedup:
        b = len(df_p); df_p = df_p.drop_duplicates()
        removed = b - len(df_p)
        if removed:
            log.append(f"Removed {removed} duplicate row(s) ({b} → {len(df_p)})")

    # 6 — Missing values
    curr_miss = [c for c in df_p.columns if df_p[c].isnull().sum() > 0]
    if curr_miss and impute_strategy != "Leave as-is":
        if impute_strategy == "Drop rows":
            b = len(df_p); df_p = df_p.dropna()
            log.append(f"Dropped {b - len(df_p)} row(s) with missing values")
        else:
            for c in curr_miss:
                if c not in df_p.columns: continue
                n = int(df_p[c].isnull().sum())
                if n == 0: continue
                if impute_strategy == "Fill with median" and pd.api.types.is_numeric_dtype(df_p[c]):
                    val = df_p[c].median(); df_p[c] = df_p[c].fillna(val)
                    log.append(f"Filled {n} in '{c}' with median ({val:.3g})")
                elif impute_strategy == "Fill with mean" and pd.api.types.is_numeric_dtype(df_p[c]):
                    val = df_p[c].mean(); df_p[c] = df_p[c].fillna(val)
                    log.append(f"Filled {n} in '{c}' with mean ({val:.3g})")
                else:
                    val = df_p[c].mode()[0]; df_p[c] = df_p[c].fillna(val)
                    log.append(f"Filled {n} in '{c}' with mode ('{val}')")

    # 7 — Outlier removal (IQR)
    if outlier_mode != "None":
        oc = ([c for c in df_p.select_dtypes(include="number").columns if c != target_col]
              if outlier_mode == "All numerical columns" else outlier_cols_sel)
        b = len(df_p)
        mask = pd.Series(True, index=df_p.index)
        for c in oc:
            if c not in df_p.columns: continue
            q1, q3 = df_p[c].quantile(0.25), df_p[c].quantile(0.75); iqr = q3 - q1
            mask &= (df_p[c] >= q1 - 1.5 * iqr) & (df_p[c] <= q3 + 1.5 * iqr)
        df_p = df_p[mask]
        removed = b - len(df_p)
        if removed:
            log.append(f"Removed {removed} outlier row(s) via IQR across {len(oc)} column(s)")

    # 8 — Encoding
    enc_now = [c for c in df_p.columns if c != target_col and df_p[c].dtype == object]
    if enc_now:
        if encode_strategy == "Label Encoding":
            for c in enc_now:
                u = df_p[c].nunique()
                df_p[c] = LabelEncoder().fit_transform(df_p[c].astype(str))
                log.append(f"Label-encoded '{c}' ({u} unique)")
        else:
            bc = df_p.shape[1]
            df_p = pd.get_dummies(df_p, columns=enc_now, drop_first=False)
            log.append(f"One-hot encoded {len(enc_now)} column(s) → +{df_p.shape[1] - bc} binary cols")

    # Catch remaining object cols
    for c in df_p.columns:
        if df_p[c].dtype == object:
            df_p[c] = LabelEncoder().fit_transform(df_p[c].astype(str))
            log.append(f"Label-encoded remaining object column '{c}'")

    # 9 — Feature scaling
    if scale_strategy != "None":
        feat_cols = [c for c in df_p.select_dtypes(include="number").columns if c != target_col]
        if feat_cols:
            if scale_strategy == "StandardScaler":
                sc = StandardScaler()
            elif scale_strategy == "MinMaxScaler":
                sc = MinMaxScaler()
            else:
                sc = RobustScaler()
            df_p[feat_cols] = sc.fit_transform(df_p[feat_cols])
            log.append(f"Applied {scale_strategy} to {len(feat_cols)} feature column(s)")

    if not log:
        log.append("No changes required — dataset was already clean.")

    st.session_state.cleaned_df = df_p
    st.session_state.prep_log   = log
    st.session_state.algo       = None
    st.session_state.y_pred     = None

if st.session_state.cleaned_df is None:
    st.stop()

df_clean = st.session_state.cleaned_df

st.subheader("Cleaning Report")
r1, r2, r3, r4, r5 = st.columns(5)
r1.metric("Original shape",    f"{df.shape[0]} × {df.shape[1]}")
r2.metric("New shape",         f"{df_clean.shape[0]} × {df_clean.shape[1]}")
r3.metric("Rows removed",      df.shape[0] - df_clean.shape[0])
r4.metric("Columns removed",   df.shape[1] - df_clean.shape[1])
r5.metric("Missing remaining", int(df_clean.isnull().sum().sum()))

st.subheader("Preprocessing log")
for i, entry in enumerate(st.session_state.prep_log):
    st.markdown(f"**{i+1}.** {entry}")

with st.expander("Preview cleaned dataset"):
    st.dataframe(df_clean.head(10), use_container_width=True)

st.divider()

# ══ STEP 5 — Model ════════════════════════════════════════════════════════════
step_badge(5, "Choose model", done=st.session_state.algo is not None)
model_list = CLF_MODELS if task == "Classification" else REG_MODELS
m_opts     = ["— select —"] + model_list
m_idx      = m_opts.index(st.session_state.algo) if st.session_state.algo in model_list else 0
st.selectbox("Algorithm", m_opts, index=m_idx, key="_algo_sel", on_change=_on_algo_change)
if not st.session_state.algo:
    st.stop()

algo = st.session_state.algo
st.divider()

# ══ STEP 6 — Configure & Train ════════════════════════════════════════════════
step_badge(6, "Configure & Train")

X             = df_clean.drop(columns=[target_col])
y             = df_clean[target_col]
feature_names = X.columns.tolist()

left, right = st.columns(2)
with left:
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    scale     = st.checkbox("Standardise features", value=True,
                            help="Recommended for SVM, KNN, MLP")

params = {}
with right:
    if algo == "Logistic Regression":
        params["C"]        = st.slider("C (regularisation)", 0.01, 10.0, 1.0, 0.01)
        params["max_iter"] = st.slider("Max iterations", 100, 1000, 200, 50)
    elif algo == "Linear Regression":
        params["variant"] = st.selectbox("Variant", ["OLS","Ridge","Lasso"])
        if params["variant"] != "OLS":
            params["alpha"] = st.slider("Alpha", 0.001, 10.0, 1.0, 0.001)
    elif algo in ("Random Forest","Random Forest (Regressor)"):
        params["n_estimators"] = st.slider("Trees", 10, 300, 100, 10)
        params["max_depth"]    = st.select_slider("Max depth",
            options=[None,3,5,10,15,20], value=None)
    elif algo in ("Gradient Boosting","Gradient Boosting Regressor"):
        params["n_estimators"]  = st.slider("Rounds", 50, 500, 100, 10)
        params["learning_rate"] = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
        params["max_depth"]     = st.slider("Depth per tree", 1, 8, 3)
        params["subsample"]     = st.slider("Subsample", 0.5, 1.0, 1.0, 0.05)
    elif algo in ("AdaBoost","AdaBoost Regressor"):
        params["n_estimators"]  = st.slider("Rounds", 10, 300, 50, 10)
        params["learning_rate"] = st.slider("Learning rate", 0.01, 2.0, 1.0, 0.01)
        params["max_depth"]     = st.slider("Base tree depth", 1, 5, 1)
    elif algo in ("Support Vector Machine","Support Vector Regressor"):
        params["C"]      = st.slider("C", 0.01, 10.0, 1.0, 0.01)
        params["kernel"] = st.selectbox("Kernel", ["rbf","linear","poly"])
        if "Regressor" in algo:
            params["epsilon"] = st.slider("Epsilon", 0.01, 1.0, 0.1, 0.01)
    elif algo == "K-Nearest Neighbors":
        params["n_neighbors"] = st.slider("k", 1, 20, 5)
        params["weights"]     = st.selectbox("Weights", ["uniform","distance"])
        params["metric"]      = st.selectbox("Metric", ["minkowski","euclidean","manhattan"])
    elif algo == "Decision Tree":
        params["max_depth"]         = st.slider("Max depth", 1, 15, 5)
        params["criterion"]         = st.selectbox("Criterion", ["gini","entropy"])
        params["min_samples_split"] = st.slider("Min samples split", 2, 20, 2)
    elif algo == "Naive Bayes":
        params["variant"] = st.selectbox("Variant", ["Gaussian","Bernoulli","Multinomial"])
        if params["variant"] == "Gaussian":
            params["var_smoothing"] = st.select_slider("Var smoothing",
                options=[1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6], value=1e-9)
        else:
            params["alpha"] = st.slider("Alpha", 0.01, 2.0, 1.0, 0.01)
    elif algo == "MLP Neural Network":
        arch = st.selectbox("Architecture",
            ["(100,)","(100, 50)","(128, 64, 32)","(256, 128)","(64,)","(200, 100, 50)"])
        params["hidden_layer_sizes"] = eval(arch)
        params["activation"]         = st.selectbox("Activation", ["relu","tanh","logistic"])
        params["solver"]             = st.selectbox("Solver", ["adam","sgd","lbfgs"])
        params["alpha"]              = st.select_slider("L2 penalty",
            options=[1e-5,1e-4,1e-3,0.01,0.1,1.0], value=1e-4)
        params["learning_rate_init"] = st.select_slider("LR",
            options=[1e-4,5e-4,1e-3,5e-3,0.01], value=1e-3)
        params["max_iter"] = st.slider("Max epochs", 100, 1000, 300, 50)

st.markdown("")
if st.button("Train Model", type="primary", use_container_width=True):
    # Split & scale only when actually training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    if scale:
        sc      = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test  = sc.transform(X_test)

    with st.spinner(f"Training {algo}…"):
        if algo == "Logistic Regression":
            from ml_models.log_reg import logistic_regression
            logistic_regression(X_train, X_test, y_train, y_test, params)
        elif algo == "Linear Regression":
            from ml_models.lin_reg import linear_regression
            linear_regression(X_train, X_test, y_train, y_test, params)
        elif algo in ("Random Forest","Random Forest (Regressor)"):
            from ml_models.ran_for import random_forest_classifier
            _rf_task = "Classification" if algo == "Random Forest" else "Regression"
            random_forest_classifier(X_train, X_test, y_train, y_test, params, feature_names, _rf_task)
        elif algo == "Gradient Boosting":
            from ml_models.gradient_boost import gradient_boosting
            gradient_boosting(X_train, X_test, y_train, y_test, params, feature_names, "Classification")
        elif algo == "Gradient Boosting Regressor":
            from ml_models.gradient_boost import gradient_boosting
            gradient_boosting(X_train, X_test, y_train, y_test, params, feature_names, "Regression")
        elif algo == "AdaBoost":
            from ml_models.adaboost import adaboost
            adaboost(X_train, X_test, y_train, y_test, params, feature_names, "Classification")
        elif algo == "AdaBoost Regressor":
            from ml_models.adaboost import adaboost
            adaboost(X_train, X_test, y_train, y_test, params, feature_names, "Regression")
        elif algo == "Support Vector Machine":
            from ml_models.svm import support_vector_machine
            support_vector_machine(X_train, X_test, y_train, y_test, params)
        elif algo == "Support Vector Regressor":
            from ml_models.svr import support_vector_regressor
            support_vector_regressor(X_train, X_test, y_train, y_test, params)
        elif algo == "K-Nearest Neighbors":
            from ml_models.knn import knn_classifier
            knn_classifier(X_train, X_test, y_train, y_test, params)
        elif algo == "Decision Tree":
            from ml_models.decision_tree import decision_tree
            decision_tree(X_train, X_test, y_train, y_test, params, feature_names)
        elif algo == "Naive Bayes":
            from ml_models.naive_bayes import naive_bayes
            naive_bayes(X_train, X_test, y_train, y_test, params)
        elif algo == "MLP Neural Network":
            from ml_models.mlp import mlp
            mlp(X_train, X_test, y_train, y_test, params, task=task)

    # ── Post-training extra visualizations ───────────────────────────────────
    if st.session_state.y_pred is not None:
        y_pred = np.array(st.session_state.y_pred)
        y_true = np.array(y_test)
        st.divider()
        st.subheader("Additional Result Visualizations")

        if task == "Classification":
            from sklearn.metrics import (classification_report, confusion_matrix,
                                         accuracy_score)
            rpt     = classification_report(y_true, y_pred, output_dict=True)
            classes = [k for k in rpt if k not in ("accuracy", "macro avg", "weighted avg")]
            wrong   = int((y_true != y_pred).sum()); total = len(y_true)

            # ── Summary cards ────────────────────────────────────────────────
            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm1.metric("Accuracy",        f"{rpt['accuracy']:.3f}")
            sm2.metric("Macro Precision", f"{rpt['macro avg']['precision']:.3f}")
            sm3.metric("Macro Recall",    f"{rpt['macro avg']['recall']:.3f}")
            sm4.metric("Macro F1",        f"{rpt['macro avg']['f1-score']:.3f}")
            sm5.metric("Misclassified",   f"{wrong} / {total}")

            # ── Per-class metrics table ───────────────────────────────────────
            st.subheader("Per-class Metrics")
            clf_rows = [
                {"Class": str(c), "Precision": round(rpt[c]["precision"], 4),
                 "Recall": round(rpt[c]["recall"], 4),
                 "F1-score": round(rpt[c]["f1-score"], 4),
                 "Support": int(rpt[c]["support"])}
                for c in classes
            ]
            for avg in ("macro avg", "weighted avg"):
                clf_rows.append({"Class": avg,
                                  "Precision": round(rpt[avg]["precision"], 4),
                                  "Recall":    round(rpt[avg]["recall"],    4),
                                  "F1-score":  round(rpt[avg]["f1-score"],  4),
                                  "Support":   int(rpt[avg]["support"])})
            st.dataframe(pd.DataFrame(clf_rows), use_container_width=True, hide_index=True)

            # ── Precision / Recall / F1 grouped bar ──────────────────────────
            cls_labels = [str(c) for c in classes]
            fig_prf = go.Figure()
            fig_prf.add_bar(x=cls_labels, y=[rpt[c]["precision"] for c in classes],
                            name="Precision", marker_color="#7C3AED")
            fig_prf.add_bar(x=cls_labels, y=[rpt[c]["recall"]    for c in classes],
                            name="Recall",    marker_color="#06B6D4")
            fig_prf.add_bar(x=cls_labels, y=[rpt[c]["f1-score"]  for c in classes],
                            name="F1-score",  marker_color="#10B981")
            fig_prf.update_layout(barmode="group",
                                   title="Precision · Recall · F1 per class",
                                   yaxis=dict(range=[0, 1.05]))
            st.plotly_chart(fig_prf, use_container_width=True)

            # ── Distribution charts + confusion matrix ────────────────────────
            ev1, ev2, ev3 = st.columns(3)
            with ev1:
                pvc = pd.Series(y_pred).value_counts().reset_index()
                pvc.columns = ["Class", "Count"]
                st.plotly_chart(px.pie(pvc, names="Class", values="Count",
                    title="Predicted class distribution",
                    color_discrete_sequence=px.colors.sequential.Purples_r),
                    use_container_width=True)
            with ev2:
                act = pd.Series(y_true).value_counts().reset_index(); act.columns = ["Class", "Actual"]
                prd = pd.Series(y_pred).value_counts().reset_index(); prd.columns = ["Class", "Predicted"]
                mg  = act.merge(prd, on="Class", how="outer").fillna(0)
                fig_cmp = go.Figure()
                fig_cmp.add_bar(x=mg["Class"].astype(str), y=mg["Actual"],    name="Actual",    marker_color="#7C3AED")
                fig_cmp.add_bar(x=mg["Class"].astype(str), y=mg["Predicted"], name="Predicted", marker_color="#06B6D4")
                fig_cmp.update_layout(barmode="group", title="Actual vs Predicted counts")
                st.plotly_chart(fig_cmp, use_container_width=True)
            with ev3:
                cm = confusion_matrix(y_true, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5), facecolor="#0D1117")
                ax_cm.set_facecolor("#0D1117")
                sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax_cm,
                            linewidths=0.4, linecolor="#0D1117",
                            annot_kws={"size": 9, "color": "#F1F5F9"})
                ax_cm.set_xlabel("Predicted", color="#94A3B8", fontsize=9)
                ax_cm.set_ylabel("Actual",    color="#94A3B8", fontsize=9)
                ax_cm.set_title("Confusion Matrix", color="#A5B4FC", fontsize=10)
                ax_cm.tick_params(colors="#64748B", labelsize=8)
                fig_cm.tight_layout()
                st.pyplot(fig_cm); plt.close()

            st.info(f"Correctly classified: **{total-wrong}/{total}** ({(total-wrong)/total:.1%}) | Misclassified: **{wrong}**")

        else:
            residuals = y_true - y_pred
            ev1, ev2  = st.columns(2)
            with ev1:
                fig_rh = px.histogram(x=residuals, nbins=30, color_discrete_sequence=["#7C3AED"],
                    labels={"x":"Residual"}, title="Residual distribution")
                fig_rh.add_vline(x=0, line_dash="dash", line_color="#A78BFA")
                st.plotly_chart(fig_rh, use_container_width=True)
            with ev2:
                try:
                    from scipy import stats as scipy_stats
                    (osm, osr), (slope, intercept, _) = scipy_stats.probplot(residuals)
                    fig_qq = go.Figure()
                    fig_qq.add_scatter(x=list(osm), y=list(osr), mode="markers",
                        marker=dict(color="#7C3AED", size=4), name="Residuals")
                    fig_qq.add_scatter(x=list(osm), y=list(slope*np.array(osm)+intercept),
                        mode="lines", line=dict(color="#06B6D4"), name="Normal line")
                    fig_qq.update_layout(title="Q-Q Plot")
                    st.plotly_chart(fig_qq, use_container_width=True)
                except ImportError:
                    st.info("Install scipy for Q-Q plot.")
            err = np.abs(residuals)
            fig_err = go.Figure(go.Scatter(
                x=list(y_true), y=list(err), mode="markers",
                marker=dict(color=list(err), colorscale="Reds", opacity=0.6,
                            colorbar=dict(title="Abs Error"), size=5),
                text=[f"Actual: {a:.2f}<br>Err: {e:.2f}" for a, e in zip(y_true, err)],
                hoverinfo="text"))
            fig_err.update_layout(title="Absolute error vs actual value",
                                  xaxis_title="Actual", yaxis_title="Absolute Error")
            st.plotly_chart(fig_err, use_container_width=True)

st.divider()
if st.button("Start Over", use_container_width=True):
    for k in DEFAULTS:
        st.session_state[k] = None
    st.rerun()
