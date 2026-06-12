"""
Forecasting Engine — o9-inspired Demand Planning workbench (Streamlit)
Workflow: Login -> Outlier Review -> Segmentation -> Best-fit Forecast ->
Planner Workbench -> Consensus -> Forecast Accuracy -> Export
"""
import hashlib
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E

# =============================================================================
# PAGE CONFIG + o9-STYLE DARK THEME
# =============================================================================
st.set_page_config(page_title="Forecasting Engine", page_icon="📈", layout="wide")

CSS = """
<style>
.stApp {background-color:#0e1117;}
section[data-testid="stSidebar"] {background-color:#11151c; border-right:1px solid #232a35;}
h1,h2,h3,h4 {color:#e8eaed; font-weight:600;}
.block-container {padding-top:1.2rem;}
div[data-testid="stMetric"] {background:#161b24; border:1px solid #232a35; border-radius:8px; padding:10px 14px;}
div[data-testid="stMetric"] label {color:#8b93a7;}
.stTabs [data-baseweb="tab-list"] {background:#11151c; border-radius:6px;}
.stTabs [data-baseweb="tab"] {color:#9aa3b5;}
.stTabs [aria-selected="true"] {color:#4da3ff; border-bottom-color:#4da3ff;}
.stButton>button {background:#1f6feb; color:white; border:0; border-radius:6px;}
.stButton>button:hover {background:#2f81f7;}
div[data-testid="stDataFrame"] {border:1px solid #232a35; border-radius:8px;}
.badge {display:inline-block; padding:3px 10px; border-radius:12px; font-size:12px;
        background:#1f6feb22; color:#4da3ff; border:1px solid #1f6feb55; margin-right:6px;}
.login-card {max-width:420px; margin:8vh auto; background:#161b24; padding:36px;
             border-radius:12px; border:1px solid #232a35;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#11151c",
    font=dict(color="#cfd6e4", size=12), margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(orientation="h", y=-0.2),
)

# =============================================================================
# 1. LOGIN
# =============================================================================
VALID_USER = "Abhishek"
VALID_PW_HASH = hashlib.sha256("Abhi@123".encode()).hexdigest()


def login_screen():
    st.markdown("<div class='login-card'>", unsafe_allow_html=True)
    st.markdown("## 📈 Forecasting Engine")
    st.caption("Demand Planner Workbench — please sign in")
    with st.form("login"):
        user = st.text_input("User ID")
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in", use_container_width=True)
    if ok:
        if user.strip() == VALID_USER and hashlib.sha256(pw.encode()).hexdigest() == VALID_PW_HASH:
            st.session_state["auth"] = True
            st.rerun()
        else:
            st.error("Invalid User ID or Password")
    st.markdown("</div>", unsafe_allow_html=True)


if not st.session_state.get("auth"):
    login_screen()
    st.stop()

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


with st.sidebar:
    st.markdown("### 📈 Forecasting Engine")
    st.caption(f"Signed in as **{VALID_USER}**")
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

# Auto-refresh: if a new file arrives, clear all per-run state
if st.session_state.get("file_hash") != file_hash:
    for k in ["bestfit", "adjustments", "consensus_pick", "locked"]:
        st.session_state.pop(k, None)
    st.session_state["file_hash"] = file_hash

df = cached_load(file_bytes)
seg_df = cached_segmentation(file_hash, df, method, kk)

# =============================================================================
# NAVIGATION
# =============================================================================
PAGES = ["1 · Outlier Review", "2 · Segmentation", "3 · Best-fit Forecast",
         "4 · Planner Workbench", "5 · Consensus", "6 · Forecast Accuracy", "7 · Export"]
page = st.sidebar.radio("Workflow", PAGES)

# global key picker
all_keys = seg_df.sort_values("volume", ascending=False).index.tolist()
sel_key = st.sidebar.selectbox("Stat Item (Key)", all_keys)

g = df[df["Key"] == sel_key]
raw_s = E.to_series(g)
clean_s, lo_b, hi_b, flags = E.cleanse_series(raw_s, method, kk)
seg_row = seg_df.loc[sel_key].to_dict()
rule_name, model_pool = E.rule_for(seg_row)


def header(title, subtitle=""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown(
        f"<span class='badge'>{sel_key}</span>"
        f"<span class='badge'>Pattern: {seg_row['pattern']}</span>"
        f"<span class='badge'>PLC: {seg_row['plc']}</span>"
        f"<span class='badge'>{seg_row['vol_class']}{seg_row['variability']}</span>"
        f"<span class='badge'>{rule_name}</span>", unsafe_allow_html=True)


# =============================================================================
# PAGE 1 — OUTLIER REVIEW
# =============================================================================
if page == PAGES[0]:
    header("Outlier Review", "Collect actuals, cleanse outliers automatically or on review basis")
    fig = go.Figure()
    fig.add_scatter(x=raw_s.index, y=raw_s.values, name="Actuals",
                    mode="lines+markers", line=dict(color="#ff5c5c", width=2))
    fig.add_scatter(x=clean_s.index, y=clean_s.values, name="Actual Cleansed (System)",
                    mode="lines+markers", line=dict(color="#4da3ff", width=2))
    fig.add_scatter(x=raw_s.index, y=[hi_b] * len(raw_s), name="Outlier Upper Threshold",
                    line=dict(color="#8b93a7", dash="dash"))
    fig.add_scatter(x=raw_s.index, y=[lo_b] * len(raw_s), name="Outlier Lower Threshold",
                    line=dict(color="#8b93a7", dash="dash"))
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
    header("Portfolio Segmentation", "Quadrants by Volume × Coefficient of Variability + rule-based algorithm pools")
    plot_df = seg_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["cov"])
    plot_df = plot_df[plot_df["volume"] > 0]
    v80 = plot_df["volume"].quantile(0.80)

    quad = np.select(
        [(plot_df["volume"] >= v80) & (plot_df["cov"] < 0.5),
         (plot_df["volume"] < v80) & (plot_df["cov"] < 0.5),
         (plot_df["volume"] < v80) & (plot_df["cov"] >= 0.5)],
        ["Q1 · AX High impact / stable", "Q2 · BX Low impact / forecastable",
         "Q3 · BY Low impact / variable"], "Q4 · AY High impact / variable")
    colors = {"Q1 · AX High impact / stable": "#2dd4bf", "Q2 · BX Low impact / forecastable": "#facc15",
              "Q3 · BY Low impact / variable": "#fb7185", "Q4 · AY High impact / variable": "#a78bfa"}

    fig = go.Figure()
    for q, col in colors.items():
        m = quad == q
        fig.add_scatter(x=plot_df.loc[m, "volume"], y=plot_df.loc[m, "cov"].clip(upper=2.5),
                        mode="markers", name=q, marker=dict(color=col, size=7, opacity=0.75),
                        text=plot_df.index[m])
    fig.add_vline(x=v80, line_dash="dot", line_color="#8b93a7",
                  annotation_text="80th %tile volume")
    fig.add_hline(y=0.5, line_dash="dot", line_color="#8b93a7", annotation_text="CoV = 0.5")
    fig.update_layout(height=460, xaxis_title="Volume (kg)", yaxis_title="Coefficient of Variability",
                      **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    c = st.columns(4)
    for i, (q, col) in enumerate(colors.items()):
        m = quad == q
        c[i].metric(q.split("·")[1].strip(), f"{m.sum()} items",
                    f"{plot_df.loc[m,'volume'].sum()/max(plot_df['volume'].sum(),1):.0%} of vol")

    st.markdown("#### Decision-tree rule assignment (intermittency → PLC → variability → volume → trend → seasonality)")
    show = seg_df[["Category", "Region", "pattern", "plc", "variability", "vol_class",
                   "trend", "seasonal", "zero_pct", "cov", "adi", "cv2", "volume",
                   "outliers", "Rule", "Model Pool"]].sort_values("volume", ascending=False)
    st.dataframe(show, use_container_width=True, height=420)

# =============================================================================
# PAGE 3 — BEST-FIT FORECAST
# =============================================================================
elif page == PAGES[2]:
    header("Best-fit Forecast",
           "Backcasting on holdout → scaled, weighted Model Selection Score → lowest score wins")
    st.markdown(f"**Candidate pool ({rule_name}):** " +
                " ".join(f"<span class='badge'>{m}</span>" for m in model_pool),
                unsafe_allow_html=True)
    extra = st.multiselect("Add models beyond the rule pool",
                           [m for m in E.MODEL_REGISTRY if m not in model_pool])
    cands = model_pool + extra

    if st.button("⚙️ Run Best-fit", type="primary"):
        with st.spinner("Backtesting candidate models…"):
            res = E.run_bestfit(clean_s, cands, horizon, holdout, weights, fast_mode)
        st.session_state.setdefault("bestfit", {})[sel_key] = res

    res = st.session_state.get("bestfit", {}).get(sel_key)
    if res:
        best = res["best"]
        st.success(f"🏆 Best-fit model: **{best}** (lowest Model Selection Score)")

        t1, t2, t3 = st.tabs(["Forecast Chart", "Backcasting", "Score & Rank Details"])
        with t1:
            fc = res["forecasts"][best]
            fig = go.Figure()
            fig.add_scatter(x=clean_s.index, y=clean_s.values, name="History (cleansed)",
                            line=dict(color="#4da3ff", width=2))
            fig.add_scatter(x=fc.index, y=fc.values, name=f"Best-fit Forecast ({best})",
                            line=dict(color="#2dd4bf", width=2, dash="solid"))
            fig.update_layout(height=420, **PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            fig = go.Figure()
            fig.add_scatter(x=res["train"].index, y=res["train"].values, name="Train",
                            line=dict(color="#4da3ff"))
            fig.add_scatter(x=res["test"].index, y=res["test"].values, name="Holdout Actuals",
                            line=dict(color="#ff5c5c", width=3))
            for nm, f in res["backtests"].items():
                fig.add_scatter(x=res["test"].index, y=f, name=nm,
                                line=dict(dash="dot", width=1.5), opacity=0.8)
            fig.update_layout(height=440, **PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with t3:
            st.caption("Score = Σ (weight × scaled metric); metrics scaled 0–1 across models; "
                       "range 0 (best) → 1 (worst). Includes MAPE, WMAPE, NFM (bias), "
                       "Tracking Signal, ZMetric (reasonability) and MASE.")
            mt = res["metric_table"].round(3)
            st.dataframe(mt.style.highlight_min(subset=["Selection Score"], color="#14532d"),
                         use_container_width=True)

# =============================================================================
# PAGE 4 — PLANNER WORKBENCH (adjustments)
# =============================================================================
elif page == PAGES[3]:
    header("Planner Workbench", "Aggregate, review and overlay planner adjustments on the system forecast")
    res = st.session_state.get("bestfit", {}).get(sel_key)
    if not res:
        st.warning("Run Best-fit (page 3) for this key first.")
        st.stop()
    fc = res["forecasts"][res["best"]]
    adj_store = st.session_state.setdefault("adjustments", {})
    adj = adj_store.get(sel_key, pd.DataFrame(
        {"System Forecast": fc.values, "Promo Adj": 0.0, "Pricing Adj": 0.0,
         "Distribution Adj": 0.0, "Other Adj": 0.0}, index=fc.index))

    edited = st.data_editor(
        adj.assign(**{"Planner Forecast": lambda d: d.sum(axis=1)}),
        disabled=["System Forecast", "Planner Forecast"], use_container_width=True,
        column_config={c: st.column_config.NumberColumn(format="%.0f") for c in adj.columns})
    adj_store[sel_key] = edited.drop(columns=["Planner Forecast"])
    planner = edited.drop(columns=["Planner Forecast"]).sum(axis=1)

    fig = go.Figure()
    fig.add_bar(x=fc.index, y=fc.values, name="System Forecast", marker_color="#e879f9")
    for col, colr in [("Promo Adj", "#fb923c"), ("Pricing Adj", "#a78bfa"),
                      ("Distribution Adj", "#2dd4bf"), ("Other Adj", "#facc15")]:
        fig.add_bar(x=fc.index, y=edited[col], name=col, marker_color=colr)
    fig.add_scatter(x=fc.index, y=planner, name="Planner Forecast",
                    line=dict(color="#ff5c5c", dash="dot", width=2))
    fig.update_layout(barmode="stack", height=420, **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE 5 — CONSENSUS
# =============================================================================
elif page == PAGES[4]:
    header("Consensus", "Compare forecasting streams side-by-side and lock the consensus forecast")
    res = st.session_state.get("bestfit", {}).get(sel_key)
    if not res:
        st.warning("Run Best-fit (page 3) for this key first.")
        st.stop()
    fc = res["forecasts"][res["best"]]
    adj = st.session_state.get("adjustments", {}).get(sel_key)
    planner = adj.sum(axis=1) if adj is not None else fc
    naive_stream = pd.Series(E.MODEL_REGISTRY["Seasonal Naive"](clean_s, len(fc)), index=fc.index)

    streams = pd.DataFrame({"System Stat Forecast": fc, "Planner Forecast": planner,
                            "Sales Forecast (seasonal-naive proxy)": naive_stream})
    pick = st.radio("Consensus forecast =", streams.columns.tolist(), horizontal=True)
    st.session_state["consensus_pick"] = pick

    fig = go.Figure()
    bar_colors = ["#3b82f6", "#fb7185", "#a855f7"]
    for (nm, sdata), colr in zip(streams.items(), bar_colors):
        fig.add_bar(x=streams.index, y=sdata, name=nm, marker_color=colr)
    fig.add_scatter(x=streams.index, y=streams[pick], name="Consensus Fcst",
                    line=dict(color="#22c55e", dash="dot", width=3))
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
    res = st.session_state.get("bestfit", {}).get(sel_key)
    if not res:
        st.warning("Run Best-fit (page 3) for this key first.")
        st.stop()
    test, train = res["test"], res["train"]
    rows = {}
    for nm, f in res["backtests"].items():
        m = E.metrics(test.values, f, train.values)
        rows[nm] = {"Accuracy % (1−WMAPE)": max(0, 100 - m["WMAPE"]), "MAPE": m["MAPE"],
                    "Bias (NFM)": m["NFM"], "MASE": m["MASE"]}
    acc = pd.DataFrame(rows).T.sort_values("Accuracy % (1−WMAPE)", ascending=False).round(1)
    best = res["best"]

    fig = go.Figure()
    fig.add_bar(x=acc.index, y=acc["Accuracy % (1−WMAPE)"],
                marker_color=["#22c55e" if i == best else "#ef4444" for i in acc.index])
    fig.update_layout(height=380, yaxis_title="Accuracy %", **PLOT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(acc, use_container_width=True)
    va = acc.loc[best, "Accuracy % (1−WMAPE)"] - acc["Accuracy % (1−WMAPE)"].drop(best).max()
    st.metric("Value-Add of best-fit vs next stream", f"{va:+.1f} pts",
              "Use system stat" if va >= 0 else "Review adjustments")

# =============================================================================
# PAGE 7 — EXPORT
# =============================================================================
else:
    header("Export Data", "Download forecasts, cleansed history and segmentation")

    @st.cache_data
    def build_export(file_hash, _seg, _locked_keys):
        return _seg

    locked = st.session_state.get("locked", {})
    bestfits = st.session_state.get("bestfit", {})

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        seg_df.reset_index().to_excel(xw, "Segmentation", index=False)
        if bestfits:
            recs = []
            for k, r in bestfits.items():
                f = r["forecasts"][r["best"]]
                recs.append(pd.DataFrame({"Key": k, "Month": f.index,
                                          "Best Model": r["best"], "Forecast (kg)": f.values}))
            pd.concat(recs).to_excel(xw, "BestFit_Forecasts", index=False)
        if locked:
            recs = [pd.DataFrame({"Key": k, "Month": v.index, "Consensus Fcst (kg)": v.values})
                    for k, v in locked.items()]
            pd.concat(recs).to_excel(xw, "Consensus_Locked", index=False)
        pd.DataFrame({"Month": clean_s.index, "Key": sel_key,
                      "Cleansed History": clean_s.values}).to_excel(xw, "Cleansed_Selected_Key", index=False)

    st.download_button("⬇️ Download Forecast Workbook (.xlsx)", out.getvalue(),
                       file_name="Forecasting_Engine_Output.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Keys segmented", len(seg_df))
    c2.metric("Best-fits run", len(bestfits))
    c3.metric("Consensus locked", len(locked))
