# 📈 Forecasting Engine — Demand Planner Workbench

A production-grade, dark theme interactive demand planning app built in Streamlit.

**Workflow:** Login → Outlier Review → Segmentation → Best-fit Forecast → Planner Workbench → Consensus → Forecast Accuracy → Export

## Features
- **Login screen** (User ID)
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

