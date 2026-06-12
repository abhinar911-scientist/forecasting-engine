# 📈 Forecasting Engine — Demand Planner Workbench

A production-grade, o9-style interactive demand planning app built in Streamlit.

**Workflow:** Login → Outlier Review → Segmentation → Best-fit Forecast → Planner Workbench → Consensus → Forecast Accuracy → Export

## Features
- **Login screen** (User ID: `Abhishek`)
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

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input file
Same layout as `Sales_History_last_36_months_v1.xlsx`:
| Month | Key | History For Forecast (kg) |
|---|---|---|

---

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
3. Click **Deploy**. First build takes ~5–10 min (Prophet compiles its backend).
4. Your app is live at `https://<app-name>.streamlit.app`. Every `git push` to `main`
   auto-redeploys.

## Step 4 — Move credentials to Secrets (do this before sharing the URL)
Hard-coded passwords in source code are unsafe. On Streamlit Cloud:
1. App page → **⋮ → Settings → Secrets**, add:
   ```toml
   APP_USER = "Abhishek"
   APP_PW_SHA256 = "<sha256 of Abhi@123>"
   ```
2. In `app.py`, replace the constants with:
   ```python
   VALID_USER = st.secrets.get("APP_USER", "Abhishek")
   VALID_PW_HASH = st.secrets.get("APP_PW_SHA256", VALID_PW_HASH)
   ```
   (Locally, put the same keys in `.streamlit/secrets.toml` — already gitignored.)

## Safe-deployment checklist
- ✅ **Private repo** — keeps code, business logic and any sample data out of public view
- ✅ **Secrets, not source** — credentials live in Streamlit Secrets / secrets.toml only
- ✅ **Hash passwords** — the app compares SHA-256 hashes, never plaintext
- ✅ **Never commit data** — `.gitignore` blocks `.xlsx`; users upload at runtime,
  Streamlit Cloud keeps uploads in memory per session only
- ✅ **Pin dependencies** — `requirements.txt` uses minimum versions; for strict
  reproducibility, pin exact versions (`pip freeze`) before go-live
- ✅ **Limit upload size** — `maxUploadSize = 100` MB in config.toml
- ✅ **Rotate the password** periodically; for multiple users or SSO, consider
  `streamlit-authenticator` or putting the app behind your company's identity provider
- ⚠️ Streamlit Community Cloud is public internet — for confidential sales data,
  prefer Streamlit on an internal server / Snowflake / company cloud with VPN
