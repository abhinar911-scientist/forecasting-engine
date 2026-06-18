# 📈 Forecasting Engine — Demand Planner Workbench

A production-grade, o9-style interactive demand planning app built in Streamlit.

**Workflow:** Login → Outlier Review → Segmentation → Best-fit Forecast → Planner Workbench → Consensus → Forecast Accuracy → Export

## Features
- **Login screen** 
- **Outlier Review** — IQR/Sigma thresholds, system-cleansed actuals, outlier flags (red cells)
- **Segmentation** — Volume × CoV quadrants (AX/AY/BX/BY), ADI/CV² demand patterns
  (Smooth / Erratic / Intermittent / Lumpy), PLC, trend & seasonality tests
- **Rule-based model pools** — intermittency → PLC → variability → volume → trend → seasonality
  decision tree assigns the right algorithm group per item
- **Model library** — SES, DES/Holt, TES/Holt-Winters, ETS, Theta, Auto-ARIMA, sARIMA,
  STL Forecast, Croston, SBA, Naive/Seasonal Naive, Moving Avg, Random Walk, **Prophet**,
  **LightGBM, XGBoost, Random Forest**, and a fast **Neural Net (MLP)** deep-learning model
- **Best-fit selection** — Model Selection Score = Σ(weight × scaled metric) over
  MAPE, WMAPE, NFM (bias), Tracking Signal, ZMetric (reasonability), MASE.
  Scaled 0–1 across models, weights sum to 1, **lowest score wins** (0 best → 1 worst)
- **Planner adjustments** (promo / pricing / distribution), **consensus lock**,
  **post-game accuracy & value-add**, **Excel export**
- **Auto-refresh** — the app hashes the uploaded file; a new upload clears caches
  and recomputes everything


# 🚀 Deploy to Streamlit Community Cloud via GitHub

## Step 1 — Create the GitHub repository
1. Go to https://github.com → **New repository** → name it `forecasting-engine`.
2. Set it to **Private** (recommended — the app has a login but the code shouldn't be public).
3. Don't initialise with a README (you already have one).

## Step 2 — Push the code
```bash
cd forecasting_engine
git init
git add app.py engine.py requirements.txt README.md .gitignore .streamlit/config.toml
git commit -m "Forecasting Engine v1"
git branch -M main
git remote add origin https://github.com/<your-username>/forecasting-engine.git
git push -u origin main
```
> Note: `.gitignore` excludes `*.xlsx` and `secrets.toml` — never commit data or secrets.

## Step 3 — Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io and sign in **with GitHub**.
2. Click **New app** → pick repo `forecasting-engine`, branch `main`, main file `app.py`.
3. **Important:** open **Advanced settings → Python version → 3.12** before deploying.
   (Streamlit Cloud's newest default Python ships with pandas 3.x, which has breaking
   API changes — the pinned `requirements.txt` pairs with Python 3.12.)
4. Click **Deploy**. First build takes ~5–10 min (Prophet compiles its backend).
5. Your app is live at `https://<app-name>.streamlit.app`. Every `git push` to `main`
   auto-redeploys. If you change the Python version later, use
   **Manage app → Reboot / Clear cache** to force a clean rebuild.



## Zero-shot foundation models (optional, auto-detected)
The app can use pretrained "zero-shot" time-series foundation models alongside the
classical/ML models. They are **optional and pluggable**: the app detects what is
installed and whether the host has enough RAM, lists the status in the sidebar
("Zero-shot models: N ready"), and **silently skips** anything unavailable — the
app never breaks if a package is missing.

| Model | Package | Notes |
|---|---|---|
| Chronos-Bolt (Amazon) | `chronos-forecasting` + `torch` | Lightest (~200 MB), CPU-friendly. Recommended first. |
| MOIRAI (Salesforce) | `uni2ts` | Small variant ~100 MB. |
| TimesFM (Google) | `timesfm` | Heavy (~1.5 GB RAM) — needs a larger host. |
| TimeGPT (Nixtla) | `nixtla` + `NIXTLA_API_KEY` | API model; sends data to Nixtla. |

When available, these models are added to **every rule's candidate pool** and
compete in the same best-fit selection. Controls in the sidebar:
- **Fast mode** (default ON) — skips Prophet and all foundation models for quick batch runs.
- **Deep mode** — includes foundation models. The first call loads weights (slow once
  per session); afterwards inference is cached and fast. Batch runs with Deep mode use
  threads so the loaded weights are shared across keys instead of reloaded per process.

**Enabling them:** uncomment the relevant lines in `requirements.txt`. Do this only on
a host with enough RAM/CPU — **not** free Streamlit Community Cloud (~1 GB), where only
the lightest models (or none) will load. To force foundation models off regardless of
what is installed, set env var `FORECAST_DISABLE_FOUNDATION=1`.

## Parallel execution on different hosts
The batch best-fit runner picks the best executor the environment allows and
degrades gracefully — no configuration needed:
1. **Process pool** ('spawn' context) on normal servers/laptops — true parallelism
2. **Thread pool** on sandboxed hosts like Streamlit Community Cloud, which block
   new process creation (numpy/statsmodels/LightGBM release the GIL in their C
   cores, so threads still overlap well)
3. **Serial loop** as the guaranteed last resort
- ⚠️ Streamlit Community Cloud is public internet — for confidential sales data,
  prefer Streamlit on an internal server / Snowflake / company cloud with VPN
