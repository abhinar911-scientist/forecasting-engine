"""
Forecasting Engine — o9-inspired Demand Planning workbench (Streamlit)
Workflow: Login -> Outlier Review -> Segmentation -> Best-fit Forecast ->
Planner Workbench -> Consensus -> Forecast Accuracy -> Export

v3: neon high-visibility theme, parallel all-key best-fit, interactive
backtesting model filter, and a Model Override tab with an audit log.
"""
import hashlib
import io
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E

# =============================================================================
# PAGE CONFIG + HIGH-VISIBILITY NEON DARK THEME
# =============================================================================
st.set_page_config(page_title="Forecasting Engine", page_icon="📈", layout="wide")

NEON = "#00d4ff"          # neon blue accent
TEXT = "#eaf4ff"          # near-white body text
SUBTEXT = "#9fd8ff"       # bright cyan-tinted secondary text

CSS = f"""
<style>
.stApp {{background-color:#0a0e14;}}
section[data-testid="stSidebar"] {{background-color:#0e131b; border-right:1px solid #1d2735;}}

/* ---- High-visibility text everywhere ---- */
html, body, p, span, label, li, td, th, div {{color:{TEXT};}}
h1,h2,h3,h4 {{color:{NEON} !important; font-weight:700; text-shadow:0 0 14px rgba(0,212,255,.25);}}
small, .stCaption, div[data-testid="stCaptionContainer"] p {{color:{SUBTEXT} !important;}}
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {{color:{TEXT} !important;}}
div[data-testid="stWidgetLabel"] p {{color:{SUBTEXT} !important; font-weight:600;}}

.block-container {{padding-top:1.1rem;}}
div[data-testid="stMetric"] {{background:#101724; border:1px solid #1f3a52; border-radius:10px;
                              padding:10px 14px; box-shadow:0 0 12px rgba(0,212,255,.07);}}
div[data-testid="stMetric"] label {{color:{SUBTEXT} !important;}}
div[data-testid="stMetricValue"] {{color:{NEON} !important;}}

.stTabs [data-baseweb="tab-list"] {{background:#0e131b; border-radius:8px;}}
.stTabs [data-baseweb="tab"] {{color:{SUBTEXT};}}
.stTabs [aria-selected="true"] {{color:{NEON} !important; border-bottom-color:{NEON} !important;}}

.stButton>button, .stDownloadButton>button {{background:linear-gradient(90deg,#0077b6,{NEON});
   color:#02131c; font-weight:700; border:0; border-radius:8px;}}
.stButton>button:hover, .stDownloadButton>button:hover {{filter:brightness(1.15);}}

div[data-testid="stDataFrame"] {{border:1px solid #1f3a52; border-radius:10px;}}
.badge {{display:inline-block; padding:3px 12px; border-radius:14px; font-size:12px; font-weight:600;
        background:rgba(0,212,255,.12); color:{NEON}; border:1px solid rgba(0,212,255,.45); margin:0 6px 6px 0;}}
.login-card {{max-width:430px; margin:8vh auto; background:#101724; padding:38px;
             border-radius:14px; border:1px solid #1f3a52; box-shadow:0 0 30px rgba(0,212,255,.12);}}
a {{color:{NEON} !important;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="#0a0e14", plot_bgcolor="#0e131b",
    font=dict(color=TEXT, size=13), margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(orientation="h", y=-0.22, font=dict(color=TEXT)),
    xaxis=dict(gridcolor="#1d2735", color=SUBTEXT),
    yaxis=dict(gridcolor="#1d2735", color=SUBTEXT),
)

C_HIST, C_ACT, C_BEST, C_THRESH = "#00d4ff", "#ff5c7a", "#39ff8e", "#7e8aa6"
PALETTE = ["#ffd166", "#c77dff", "#ff9e64", "#4cc9f0", "#f72585",
           "#80ffdb", "#fca311", "#bde0fe", "#e9ff70", "#ff70a6"]

# =============================================================================
# 1. LOGIN
# =============================================================================
def _secret(key: str, default: str) -> str:
    """st.secrets raises if no secrets.toml exists at all — fall back safely."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def _clean(v) -> str:
    """Strip whitespace and accidental wrapping quotes from secret values."""
    return str(v).strip().strip('"').strip("'").strip()


import re as _re

DEFAULT_HASH = hashlib.sha256("Abhi@123".encode()).hexdigest()

VALID_USER = _clean(_secret("APP_USER", "Abhishek"))

_raw_hash = _clean(_secret("APP_PW_SHA256", "")).lower()
_raw_plain = _clean(_secret("APP_PASSWORD", ""))
if _re.fullmatch(r"[0-9a-f]{64}", _raw_hash):
    VALID_PW_HASH = _raw_hash                 # valid SHA-256 hex from secrets
elif _raw_plain:
    VALID_PW_HASH = hashlib.sha256(_raw_plain.encode()).hexdigest()  # plaintext secret
else:
    VALID_PW_HASH = DEFAULT_HASH              # no/malformed secret → safe default


def login_screen():
    st.markdown("<div class='login-card'>", unsafe_allow_html=True)
    st.markdown("## 📈 Forecasting Engine")
    st.caption("Demand Planner Workbench — please sign in")
    with st.form("login"):
        user = st.text_input("User ID")
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in", use_container_width=True)
    if ok:
        u_ok = _clean(user).lower() == VALID_USER.lower()
        p_ok = hashlib.sha256(pw.encode()).hexdigest() == VALID_PW_HASH
        if u_ok and p_ok:
            st.session_state["auth"] = True
            st.session_state["user"] = _clean(user)
            st.rerun()
        else:
            st.error("Invalid User ID or Password")
    st.markdown("</div>", unsafe_allow_html=True)


if not st.session_state.get("auth"):
    login_screen()
    st.stop()

CURRENT_USER = st.session_state.get("user", VALID_USER)

# =============================================================================
# 2. DATA INGESTION (auto-refresh on file change via hash-keyed cache)
# =============================================================================
@st.cache_data(show_spinner="Loading sales history…")
def cached_load(file_bytes: bytes) -> pd.DataFrame:
    return E.load_history(io.BytesIO(file_bytes))


@st.cache_data(show_spinner="Segmenting portfolio…")
def cached_segmentation(file_hash: str, _df: pd.DataFrame, method: str, k: float) -> pd.DataFrame:
    rows = []
    for key, g in _df.groupby("Key"):
        s = E.to_series(g)
        c, lo, hi, fl = E.cleanse_series(s, method, k)
        seg = E.classify_series(c)
        seg.update(Key=key, Category=g["Category"].iloc[0], Region=g["Region"].iloc[0],
                   outliers=int(fl.sum()))
        rows.append(seg)
    seg_df = pd.DataFrame(rows).set_index("Key")
    seg_df["vol_class"] = E.volume_class(seg_df["volume"])
    rules = seg_df.apply(lambda r: E.rule_for(r.to_dict()), axis=1)
    seg_df["Rule"] = [r[0] for r in rules]
    seg_df["Model Pool"] = [", ".join(r[1]) for r in rules]
    return seg_df


@st.cache_data(show_spinner=False)
def cached_forecast_one(file_hash: str, key: str, model: str, horizon: int,
                        method: str, k: float, _series: pd.Series) -> pd.Series:
    c, *_ = E.cleanse_series(_series, method, k)
    return E.forecast_one(c, model, horizon)


@st.cache_data(show_spinner="Generating 24-month forecasts for candidate models…")
def cached_forecast_pool24(file_hash: str, key: str, models: tuple,
                           method: str, k: float, _series: pd.Series) -> pd.DataFrame:
    c, *_ = E.cleanse_series(_series, method, k)
    return E.forecast_all(c, list(models), 24)


with st.sidebar:
    st.markdown("### 📈 Forecasting Engine")
    st.caption(f"Signed in as **{CURRENT_USER}**")
    if st.button("Log out", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.divider()

    up = st.file_uploader("Upload Sales History (.xlsx)", type=["xlsx"],
                          help="Same layout as Sales_History_last_36_months_v1.xlsx")
    st.divider()
    st.markdown("**Cleansing settings**")
    method = st.selectbox("Outlier method", ["IQR", "Sigma"])
    kk = st.slider("Threshold (k)", 1.0, 3.0, 1.5, 0.25)
    st.markdown("**Forecast settings**")
    horizon = st.slider("Horizon (months)", 3, 24, 12)
    holdout = st.slider("Holdout for best-fit (months)", 3, 12, 6)
    fast_mode = st.toggle("Fast mode (skip Prophet)", value=True)
    st.markdown("**Selection score weights** (sum normalised to 1)")
    weights = {}
    for mname, dflt in E.DEFAULT_WEIGHTS.items():
        weights[mname] = st.slider(mname, 0.0, 1.0, dflt, 0.05, key=f"w_{mname}")

if up is None:
    st.info("⬅️ Upload the sales history workbook to begin. Expected columns: "
            "**Month, Key, History For Forecast (kg)**.")
    st.stop()

file_bytes = up.getvalue()
file_hash = hashlib.md5(file_bytes).hexdigest()

# Auto-refresh: a new file clears all per-run state
if st.session_state.get("file_hash") != file_hash:
    for k in ["bestfit", "adjustments", "consensus_pick", "locked",
              "overrides", "override_log", "settings_sig"]:
        st.session_state.pop(k, None)
    st.session_state["file_hash"] = file_hash

# Settings signature: if cleansing / scoring settings change, prior best-fits are stale
settings_sig = hashlib.md5(
    f"{method}|{kk}|{horizon}|{holdout}|{fast_mode}|{sorted(weights.items())}".encode()
).hexdigest()
if st.session_state.get("settings_sig") not in (None, settings_sig):
    st.session_state.pop("bestfit", None)
st.session_state["settings_sig"] = settings_sig

df = cached_load(file_bytes)
seg_df = cached_segmentation(file_hash, df, method, kk)

# =============================================================================
# NAVIGATION + GLOBAL CONTEXT
# =============================================================================
PAGES = ["1 · Outlier Review", "2 · Segmentation", "3 · Best-fit Forecast",
         "4 · Planner Workbench", "5 · Consensus", "6 · Forecast Accuracy", "7 · Export"]
page = st.sidebar.radio("Workflow", PAGES)

all_keys = seg_df.sort_values("volume", ascending=False).index.tolist()
sel_key = st.sidebar.selectbox("Stat Item (Key)", all_keys)

g = df[df["Key"] == sel_key]
raw_s = E.to_series(g)
clean_s, lo_b, hi_b, flags = E.cleanse_series(raw_s, method, kk)
seg_row = seg_df.loc[sel_key].to_dict()
rule_name, model_pool = E.rule_for(seg_row)


def get_res(key):
    r = st.session_state.get("bestfit", {}).get(key)
    return None if (r is None or "error" in r) else r


def effective_model(key):
    """Override model if present, else best-fit model."""
    ov = st.session_state.get("overrides", {}).get(key)
    res = get_res(key)
    best = res["best"] if res else None
    return (ov["model"] if ov else best), best, ov


def effective_forecast(key, series, hz):
    """Forecast from the effective (override-or-best) model at horizon hz."""
    model, best, _ = effective_model(key)
    if model is None:
        return None, None
    res = get_res(key)
    if model == best and hz == len(res["forecasts"][best]):
        return model, res["forecasts"][best]
    return model, cached_forecast_one(file_hash, key, model, hz, method, kk, series)


def header(title, subtitle=""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    eff, best, ov = effective_model(sel_key)
    ov_badge = (f"<span class='badge' style='background:rgba(255,209,102,.15);"
                f"color:#ffd166;border-color:#ffd16677;'>Override → {eff}</span>") if ov else ""
    st.markdown(
        f"<span class='badge'>{sel_key}</span>"
        f"<span class='badge'>Pattern: {seg_row['pattern']}</span>"
        f"<span class='badge'>PLC: {seg_row['plc']}</span>"
        f"<span class='badge'>{seg_row['vol_class']}{seg_row['variability']}</span>"
        f"<span class='badge'>{rule_name}</span>{ov_badge}", unsafe_allow_html=True)


# =============================================================================
# PAGE 1 — OUTLIER REVIEW
# =============================================================================
if page == PAGES[0]:
    header("Outlier Review", "Collect actuals, cleanse outliers automatically or on review basis")
    fig = go.Figure()
    fig.add_scatter(x=raw_s.index, y=raw_s.values, name="Actuals",
                    mode="lines+markers", line=dict(color=C_ACT, width=2))
    fig.add_scatter(x=clean_s.index, y=clean_s.values, name="Actual Cleansed (System)",
                    mode="lines+markers", line=dict(color=C_HIST, width=2))
    fig.add_scatter(x=raw_s.index, y=[hi_b] * len(raw_s), name="Outlier Upper Threshold",
                    line=dict(color=C_THRESH, dash="dash"))
    fig.add_scatter(x=raw_s.index, y=[lo_b] * len(raw_s), name="Outlier Lower Threshold",
                    line=dict(color=C_THRESH, dash="dash"))
    fig.update_layout(height=420, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Outliers detected", int(flags.sum()))
    c2.metric("Upper threshold", f"{hi_b:,.0f}")
    c3.metric("Lower threshold", f"{lo_b:,.0f}")
    c4.metric("36-mo volume", f"{raw_s.sum():,.0f} kg")

    st.markdown("#### Outlier Review Details")
    tbl = pd.DataFrame({"Actuals": raw_s, "Actual Cleansed System": clean_s,
                        "Outlier": np.where(flags, "⚠️", "")}).T
    tbl.columns = [c.strftime("%b-%y") for c in raw_s.index]
    st.dataframe(tbl, use_container_width=True)

# =============================================================================
# PAGE 2 — SEGMENTATION
# =============================================================================
elif page == PAGES[1]:
    header("Portfolio Segmentation",
           "Quadrants by Volume × Coefficient of Variability + rule-based algorithm pools")
    plot_df = seg_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["cov"])
    plot_df = plot_df[plot_df["volume"] > 0]
    v80 = plot_df["volume"].quantile(0.80)

    quad = np.select(
        [(plot_df["volume"] >= v80) & (plot_df["cov"] < 0.5),
         (plot_df["volume"] < v80) & (plot_df["cov"] < 0.5),
         (plot_df["volume"] < v80) & (plot_df["cov"] >= 0.5)],
        ["Q1 · AX High impact / stable", "Q2 · BX Low impact / forecastable",
         "Q3 · BY Low impact / variable"], "Q4 · AY High impact / variable")
    colors = {"Q1 · AX High impact / stable": "#39ff8e",
              "Q2 · BX Low impact / forecastable": "#ffd166",
              "Q3 · BY Low impact / variable": "#ff5c7a",
              "Q4 · AY High impact / variable": "#c77dff"}

    fig = go.Figure()
    for q, col in colors.items():
        m = quad == q
        fig.add_scatter(x=plot_df.loc[m, "volume"], y=plot_df.loc[m, "cov"].clip(upper=2.5),
                        mode="markers", name=q, marker=dict(color=col, size=7, opacity=0.8),
                        text=plot_df.index[m])
    fig.add_vline(x=v80, line_dash="dot", line_color=C_THRESH,
                  annotation_text="80th %tile volume")
    fig.add_hline(y=0.5, line_dash="dot", line_color=C_THRESH, annotation_text="CoV = 0.5")
    fig.update_layout(height=460, xaxis_title="Volume (kg)",
                      yaxis_title="Coefficient of Variability", **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    c = st.columns(4)
    for i, (q, col) in enumerate(colors.items()):
        m = quad == q
        c[i].metric(q.split("·")[1].strip(), f"{m.sum()} items",
                    f"{plot_df.loc[m,'volume'].sum()/max(plot_df['volume'].sum(),1):.0%} of vol")

    st.markdown("#### Decision-tree rule assignment "
                "(intermittency → PLC → variability → volume → trend → seasonality)")
    show = seg_df[["Category", "Region", "pattern", "plc", "variability", "vol_class",
                   "trend", "seasonal", "zero_pct", "cov", "adi", "cv2", "volume",
                   "outliers", "Rule", "Model Pool"]].sort_values("volume", ascending=False)
    st.dataframe(show, use_container_width=True, height=420)

# =============================================================================
# PAGE 3 — BEST-FIT FORECAST  (parallel batch + filters + override)
# =============================================================================
elif page == PAGES[2]:
    header("Best-fit Forecast",
           "Backcasting on holdout → scaled, weighted Model Selection Score → lowest score wins")
    st.markdown(f"**Candidate pool for selected key ({rule_name}):** " +
                " ".join(f"<span class='badge'>{m}</span>" for m in model_pool),
                unsafe_allow_html=True)

    # ---- Batch scope + parallel run -----------------------------------------
    n_workers = min(max(os.cpu_count() or 1, 2), 8)
    cc1, cc2 = st.columns([2, 3])
    scope = cc1.selectbox("Run scope", ["Current key only", "Top 50 by volume",
                                        "Top 200 by volume", "All keys"])
    scope_keys = {"Current key only": [sel_key],
                  "Top 50 by volume": all_keys[:50],
                  "Top 200 by volume": all_keys[:200],
                  "All keys": all_keys}[scope]
    pending = [k for k in scope_keys if k not in st.session_state.get("bestfit", {})]
    cc2.markdown(f"<br><span class='badge'>{len(scope_keys)} keys in scope</span>"
                 f"<span class='badge'>{len(pending)} to run</span>"
                 f"<span class='badge'>{n_workers} parallel workers</span>",
                 unsafe_allow_html=True)

    if st.button("⚙️ Run Best-fit (parallel, per-key candidate pools)", type="primary"):
        keys_to_run = pending or scope_keys
        tasks = []
        grouped = {k: v for k, v in df[df["Key"].isin(keys_to_run)].groupby("Key")}
        for k in keys_to_run:
            tasks.append((k, E.to_series(grouped[k]), seg_df.loc[k, "vol_class"],
                          method, kk, horizon, holdout, weights, fast_mode))
        prog = st.progress(0.0, text=f"Running best-fit on {len(tasks)} keys…")
        t0 = time.time()
        results = E.run_batch_parallel(
            tasks, max_workers=n_workers,
            progress_cb=lambda d, n: prog.progress(
                d / n, text=f"Best-fit {d}/{n} keys · {time.time()-t0:,.0f}s elapsed"))
        prog.empty()
        store = st.session_state.setdefault("bestfit", {})
        store.update(results)
        errs = [k for k, v in results.items() if "error" in v]
        st.success(f"Completed {len(results)} keys in {time.time()-t0:,.1f}s "
                   f"({len(results)-len(errs)} ok, {len(errs)} fallback-skipped).")
        if errs:
            with st.expander("Keys that failed (excluded)"):
                st.write({k: results[k]["error"] for k in errs})

    # ---- Portfolio summary of all completed best-fits ------------------------
    store = st.session_state.get("bestfit", {})
    done_ok = {k: v for k, v in store.items() if "error" not in v}
    if done_ok:
        summ = pd.DataFrame({
            "Best-fit Model": {k: v["best"] for k, v in done_ok.items()},
            "Rule": {k: v.get("rule", "") for k, v in done_ok.items()},
            "Selection Score": {k: round(float(v["metric_table"]["Selection Score"].min()), 3)
                                for k, v in done_ok.items()},
        })
        ov_map = st.session_state.get("overrides", {})
        summ["Final Model"] = [ov_map.get(k, {}).get("model", b)
                               for k, b in summ["Best-fit Model"].items()]
        with st.expander(f"📋 Portfolio best-fit summary — {len(summ)} keys completed", expanded=False):
            mix = summ["Final Model"].value_counts()
            fig = go.Figure(go.Bar(x=mix.index, y=mix.values, marker_color=NEON))
            fig.update_layout(height=260, yaxis_title="# keys", **PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(summ.sort_values("Selection Score"), use_container_width=True, height=300)

    # ---- Selected-key detail --------------------------------------------------
    res = get_res(sel_key)
    if not res:
        st.info("Run best-fit with the selected key in scope to see its detail below.")
        st.stop()

    best = res["best"]
    eff, _, ov = effective_model(sel_key)
    st.success(f"🏆 Best-fit model: **{best}** (lowest Model Selection Score)"
               + (f" — ⚠️ overridden to **{eff}**" if ov else ""))

    t1, t2, t3, t4 = st.tabs(["Forecast Chart", "Backtesting", "Score & Rank Details",
                              "Model Override"])

    # ---- Tab 1: forecast chart (effective model) ------------------------------
    with t1:
        mdl, fc = effective_forecast(sel_key, raw_s, horizon)
        fig = go.Figure()
        fig.add_scatter(x=clean_s.index, y=clean_s.values, name="History (cleansed)",
                        line=dict(color=C_HIST, width=2))
        fig.add_scatter(x=fc.index, y=fc.values, name=f"Forecast ({mdl})",
                        line=dict(color=C_BEST, width=2.5))
        fig.update_layout(height=420, **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 2: backtesting with interactive model filter ---------------------
    with t2:
        others = [m for m in res["backtests"] if m != best]
        chosen = st.multiselect(
            "Compare models against the best-fit on the holdout "
            f"(best-fit **{best}** is always shown)",
            others, default=others[: min(3, len(others))], key="bt_filter")
        fig = go.Figure()
        fig.add_scatter(x=res["train"].index, y=res["train"].values, name="Train",
                        line=dict(color=C_HIST, width=1.5), opacity=0.7)
        fig.add_scatter(x=res["test"].index, y=res["test"].values, name="Holdout Actuals",
                        mode="lines+markers", line=dict(color=C_ACT, width=3))
        fig.add_scatter(x=res["test"].index, y=res["backtests"][best],
                        name=f"⭐ Best-fit · {best}",
                        mode="lines+markers", line=dict(color=C_BEST, width=3))
        for i, nm in enumerate(chosen):
            fig.add_scatter(x=res["test"].index, y=res["backtests"][nm], name=nm,
                            line=dict(color=PALETTE[i % len(PALETTE)], dash="dot", width=2),
                            opacity=0.9)
        fig.update_layout(height=450, **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        bt_tbl = pd.DataFrame({"Holdout Actuals": res["test"].values,
                               f"⭐ {best}": np.round(res["backtests"][best], 1),
                               **{nm: np.round(res["backtests"][nm], 1) for nm in chosen}},
                              index=[d.strftime("%b-%y") for d in res["test"].index]).T
        st.dataframe(bt_tbl, use_container_width=True)

    # ---- Tab 3: score & rank ----------------------------------------------------
    with t3:
        st.caption("Score = Σ (weight × scaled metric); metrics scaled 0–1 across models; "
                   "range 0 (best) → 1 (worst). Includes MAPE, WMAPE, NFM (bias), "
                   "Tracking Signal, ZMetric (reasonability) and MASE.")
        mt = res["metric_table"].round(3)
        st.dataframe(mt.style.highlight_min(subset=["Selection Score"], color="#0c4a2f"),
                     use_container_width=True)

    # ---- Tab 4: MODEL OVERRIDE (decision on next-24-month outcome) --------------
    with t4:
        st.caption("Override decision is based on the **next 24 months forecast outcome**, "
                   "not the holdout. Last 36 months of history + 24-month forecasts shown.")
        pool_models = tuple(res.get("pool", list(res["backtests"].keys())))
        fc24 = cached_forecast_pool24(file_hash, sel_key, pool_models, method, kk, raw_s)

        show_models = st.multiselect(
            f"Models to display (best-fit **{best}** is always shown)",
            [m for m in fc24.columns if m != best],
            default=[m for m in fc24.columns if m != best][:3], key="ov_filter")

        hist36 = clean_s.iloc[-36:]
        fig = go.Figure()
        fig.add_scatter(x=hist36.index, y=hist36.values, name="History (36 mo, cleansed)",
                        line=dict(color=C_HIST, width=2))
        fig.add_scatter(x=fc24.index, y=fc24[best], name=f"⭐ Best-fit · {best}",
                        line=dict(color=C_BEST, width=3))
        if ov and ov["model"] in fc24.columns and ov["model"] != best:
            fig.add_scatter(x=fc24.index, y=fc24[ov["model"]],
                            name=f"✅ Current override · {ov['model']}",
                            line=dict(color="#ffd166", width=3))
        for i, nm in enumerate(show_models):
            if ov and nm == ov.get("model"):
                continue
            fig.add_scatter(x=fc24.index, y=fc24[nm], name=nm,
                            line=dict(color=PALETTE[i % len(PALETTE)], dash="dot", width=2))
        fig.add_vline(x=hist36.index[-1], line_dash="dash", line_color=C_THRESH,
                      annotation_text="Forecast start")
        fig.update_layout(height=460, **PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        oc1, oc2 = st.columns([2, 1])
        choice = oc1.selectbox("Final model for downstream workflows",
                               ["(keep best-fit)"] + [m for m in fc24.columns],
                               index=(list(fc24.columns).index(ov["model"]) + 1)
                               if ov and ov["model"] in fc24.columns else 0)
        oc2.markdown("<br>", unsafe_allow_html=True)
        if oc2.button("Apply decision", type="primary", use_container_width=True):
            log = st.session_state.setdefault("override_log", [])
            ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            if choice == "(keep best-fit)":
                if ov:
                    st.session_state["overrides"].pop(sel_key, None)
                    log.append(dict(Timestamp=ts, Key=sel_key, Action="Override removed",
                                    InitialBestFit=best, FinalModel=best, ChangedBy=CURRENT_USER))
                st.success(f"Keeping best-fit **{best}** for {sel_key}.")
            else:
                st.session_state.setdefault("overrides", {})[sel_key] = dict(
                    model=choice, by=CURRENT_USER, at=ts, initial_best=best)
                log.append(dict(Timestamp=ts, Key=sel_key, Action="Override applied",
                                InitialBestFit=best, FinalModel=choice, ChangedBy=CURRENT_USER))
                st.success(f"Override applied: **{choice}** will drive downstream "
                           f"workflows for {sel_key}.")
            # downstream artefacts for this key are now stale
            st.session_state.get("adjustments", {}).pop(sel_key, None)
            st.session_state.get("locked", {}).pop(sel_key, None)
            st.rerun()

        log = st.session_state.get("override_log", [])
        if log:
            st.markdown("#### 📜 Override audit log")
            st.dataframe(pd.DataFrame(log).iloc[::-1], use_container_width=True, height=220)

# =============================================================================
# PAGE 4 — PLANNER WORKBENCH (uses effective model)
# =============================================================================
elif page == PAGES[3]:
    header("Planner Workbench",
           "Aggregate, review and overlay planner adjustments on the system forecast")
    if not get_res(sel_key):
        st.warning("Run Best-fit (page 3) with this key in scope first.")
        st.stop()
    mdl, fc = effective_forecast(sel_key, raw_s, horizon)
    st.caption(f"System forecast driven by final model: **{mdl}**")

    adj_store = st.session_state.setdefault("adjustments", {})
    adj = adj_store.get(sel_key)
    if adj is None or not adj.index.equals(fc.index):     # horizon/model changed → rebuild
        adj = pd.DataFrame({"System Forecast": fc.values, "Promo Adj": 0.0,
                            "Pricing Adj": 0.0, "Distribution Adj": 0.0,
                            "Other Adj": 0.0}, index=fc.index)

    edited = st.data_editor(
        adj.assign(**{"Planner Forecast": lambda d: d.sum(axis=1)}),
        disabled=["System Forecast", "Planner Forecast"], use_container_width=True,
        column_config={c: st.column_config.NumberColumn(format="%.0f") for c in adj.columns})
    adj_store[sel_key] = edited.drop(columns=["Planner Forecast"])
    planner = edited.drop(columns=["Planner Forecast"]).sum(axis=1)

    fig = go.Figure()
    fig.add_bar(x=fc.index, y=fc.values, name=f"System Forecast ({mdl})", marker_color="#c77dff")
    for col, colr in [("Promo Adj", "#ff9e64"), ("Pricing Adj", "#4cc9f0"),
                      ("Distribution Adj", "#39ff8e"), ("Other Adj", "#ffd166")]:
        fig.add_bar(x=fc.index, y=edited[col], name=col, marker_color=colr)
    fig.add_scatter(x=fc.index, y=planner, name="Planner Forecast",
                    line=dict(color=C_ACT, dash="dot", width=2.5))
    fig.update_layout(barmode="stack", height=420, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE 5 — CONSENSUS (uses effective model)
# =============================================================================
elif page == PAGES[4]:
    header("Consensus", "Compare forecasting streams side-by-side and lock the consensus forecast")
    if not get_res(sel_key):
        st.warning("Run Best-fit (page 3) with this key in scope first.")
        st.stop()
    mdl, fc = effective_forecast(sel_key, raw_s, horizon)
    adj = st.session_state.get("adjustments", {}).get(sel_key)
    planner = adj.sum(axis=1) if (adj is not None and adj.index.equals(fc.index)) else fc
    naive_stream = pd.Series(E.MODEL_REGISTRY["Seasonal Naive"](clean_s, len(fc)), index=fc.index)

    streams = pd.DataFrame({f"System Stat Forecast ({mdl})": fc, "Planner Forecast": planner,
                            "Sales Forecast (seasonal-naive proxy)": naive_stream})
    pick = st.radio("Consensus forecast =", streams.columns.tolist(), horizontal=True)
    st.session_state["consensus_pick"] = pick

    fig = go.Figure()
    for (nm, sdata), colr in zip(streams.items(), ["#4cc9f0", "#ff5c7a", "#c77dff"]):
        fig.add_bar(x=streams.index, y=sdata, name=nm, marker_color=colr)
    fig.add_scatter(x=streams.index, y=streams[pick], name="Consensus Fcst",
                    line=dict(color=C_BEST, dash="dot", width=3))
    fig.update_layout(barmode="group", height=430, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(streams.round(0).T, use_container_width=True)

    if st.button("🔒 Lock consensus forecast for this key", type="primary"):
        st.session_state.setdefault("locked", {})[sel_key] = streams[pick]
        st.success(f"Locked **{pick}** as consensus for {sel_key}.")

# =============================================================================
# PAGE 6 — FORECAST ACCURACY / POST-GAME
# =============================================================================
elif page == PAGES[5]:
    header("Forecast Accuracy (Post-game)",
           "Compare streams on the holdout: does the adjustment add value vs the system stat forecast?")
    res = get_res(sel_key)
    if not res:
        st.warning("Run Best-fit (page 3) with this key in scope first.")
        st.stop()
    test, train = res["test"], res["train"]
    rows = {}
    for nm, f in res["backtests"].items():
        m = E.metrics(test.values, f, train.values)
        rows[nm] = {"Accuracy % (1−WMAPE)": max(0, 100 - m["WMAPE"]), "MAPE": m["MAPE"],
                    "Bias (NFM)": m["NFM"], "MASE": m["MASE"]}
    acc = pd.DataFrame(rows).T.sort_values("Accuracy % (1−WMAPE)", ascending=False).round(1)
    best = res["best"]
    eff, _, _ = effective_model(sel_key)

    fig = go.Figure()
    fig.add_bar(x=acc.index, y=acc["Accuracy % (1−WMAPE)"],
                marker_color=[C_BEST if i == best else
                              ("#ffd166" if i == eff else "#ff5c7a") for i in acc.index])
    fig.update_layout(height=380, yaxis_title="Accuracy %", **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green = best-fit · Yellow = current override · Red = others")
    st.dataframe(acc, use_container_width=True)
    va = acc.loc[best, "Accuracy % (1−WMAPE)"] - acc["Accuracy % (1−WMAPE)"].drop(best).max()
    st.metric("Value-Add of best-fit vs next stream", f"{va:+.1f} pts",
              "Use system stat" if va >= 0 else "Review adjustments")

# =============================================================================
# PAGE 7 — EXPORT
# =============================================================================
else:
    header("Export Data", "Download forecasts, cleansed history, segmentation and the override log")

    locked = st.session_state.get("locked", {})
    bestfits = {k: v for k, v in st.session_state.get("bestfit", {}).items() if "error" not in v}
    overrides = st.session_state.get("overrides", {})
    log = st.session_state.get("override_log", [])

    sheets = {}
    sheets["Segmentation"] = (seg_df.reset_index()
                              .replace([np.inf, -np.inf], np.nan))
    if bestfits:
        recs = []
        for k, r in bestfits.items():
            final_model = overrides.get(k, {}).get("model", r["best"])
            mdl, f = effective_forecast(k, E.to_series(df[df["Key"] == k]), horizon) \
                if k in overrides else (r["best"], r["forecasts"][r["best"]])
            recs.append(pd.DataFrame({"Key": k, "Month": f.index,
                                      "Initial Best-fit": r["best"],
                                      "Final Model": final_model,
                                      "Forecast (kg)": np.round(np.asarray(f.values, float), 2)}))
        sheets["Final_Forecasts"] = pd.concat(recs, ignore_index=True)
    if locked:
        recs = [pd.DataFrame({"Key": k, "Month": v.index,
                              "Consensus Fcst (kg)": np.round(v.values, 2)})
                for k, v in locked.items()]
        sheets["Consensus_Locked"] = pd.concat(recs, ignore_index=True)
    if log:
        sheets["Override_Audit_Log"] = pd.DataFrame(log)
    sheets["Cleansed_Selected_Key"] = pd.DataFrame(
        {"Month": clean_s.index, "Key": sel_key,
         "Cleansed History (kg)": np.round(clean_s.values, 2)})

    @st.cache_data(show_spinner=False)
    def build_workbook(_sheets: dict, cache_key: str) -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            for name, sdf in _sheets.items():
                sdf = sdf.copy()
                for col in sdf.columns:
                    if pd.api.types.is_datetime64_any_dtype(sdf[col]):
                        sdf[col] = sdf[col].dt.strftime("%Y-%m-%d")
                sdf.to_excel(xw, sheet_name=str(name)[:31], index=False)
        return buf.getvalue()

    cache_key = "|".join([file_hash, sel_key, str(len(bestfits)), str(len(locked)),
                          str(len(log)), str(horizon)])
    try:
        payload = build_workbook(sheets, cache_key)
        st.download_button("⬇️ Download Forecast Workbook (.xlsx)", payload,
                           file_name="Forecasting_Engine_Output.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           type="primary")
    except Exception as exc:
        st.error(f"Excel export failed ({type(exc).__name__}). Offering CSV instead.")
        st.download_button("⬇️ Download Segmentation (.csv)",
                           sheets["Segmentation"].to_csv(index=False).encode(),
                           file_name="Forecasting_Engine_Segmentation.csv", mime="text/csv")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Keys segmented", len(seg_df))
    c2.metric("Best-fits run", len(bestfits))
    c3.metric("Overrides active", len(overrides))
    c4.metric("Consensus locked", len(locked))
