"""
Forecasting Engine — core logic
Cleansing, segmentation, demand-pattern rules, model library (Stat + Prophet + ML + DL),
and weighted-score best-fit selection.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- 
# 1. DATA LOADING
# -----------------------------------------------------------------------------
REQUIRED_COLS = {"Month", "Key", "History For Forecast (kg)"}


def load_history(file) -> pd.DataFrame:
    """Read the sales history workbook and return a tidy long dataframe."""
    df = pd.read_excel(file, sheet_name=0)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing columns: {missing}")
    df = df.rename(columns={"History For Forecast (kg)": "Actual"})
    df["Month"] = pd.to_datetime(df["Month"])
    df = (
        df.groupby(["Key", "Month"], as_index=False)["Actual"].sum()
        .sort_values(["Key", "Month"])
    )
    # Parse Key -> Category / Item / Region
    parsed = df["Key"].str.extract(r"^([A-Za-z &]+?)(\d+)(.+)$")
    df["Category"] = parsed[0].fillna("Unknown")
    df["Item"] = parsed[1].fillna(df["Key"])
    df["Region"] = parsed[2].fillna("Unknown")
    return df


def to_series(df_key: pd.DataFrame) -> pd.Series:
    s = df_key.set_index("Month")["Actual"].asfreq("MS")
    return s.fillna(0.0).clip(lower=0)


# ----------------------------------------------------------------------------- 
# 2. OUTLIER CLEANSING (IQR / z-score bands like the Outlier Review screen)
# -----------------------------------------------------------------------------
def outlier_bounds(s: pd.Series, method: str = "IQR", k: float = 1.5):
    v = s[s > 0] if (s > 0).sum() >= 6 else s
    if method == "IQR":
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
    else:  # sigma
        mu, sd = v.mean(), v.std(ddof=0)
        lo, hi = mu - k * sd, mu + k * sd
    lo = max(lo, 0.0)
    return lo, hi


def cleanse_series(s: pd.Series, method: str = "IQR", k: float = 1.5):
    """Return (cleansed, lower, upper, flags). Outliers winsorised to bounds."""
    lo, hi = outlier_bounds(s, method, k)
    flags = (s < lo) | (s > hi)
    cleansed = s.clip(lower=lo, upper=hi)
    return cleansed, lo, hi, flags


# ----------------------------------------------------------------------------- 
# 3. SEGMENTATION & DEMAND-PATTERN RULES (o9-style decision tree)
# -----------------------------------------------------------------------------
def classify_series(s: pd.Series) -> dict:
    n = len(s)
    nz = s[s > 0]
    zero_pct = float((s == 0).mean())
    vol = float(s.sum())
    cov = float(nz.std(ddof=0) / nz.mean()) if len(nz) > 1 and nz.mean() > 0 else np.inf

    # ADI / CV^2 (Syntetos-Boylan) demand pattern
    idx_nz = np.flatnonzero(s.values > 0)
    adi = (n / len(idx_nz)) if len(idx_nz) else np.inf
    cv2 = cov ** 2 if np.isfinite(cov) else np.inf
    if adi < 1.32 and cv2 < 0.49:
        pattern = "Smooth"
    elif adi >= 1.32 and cv2 < 0.49:
        pattern = "Intermittent"
    elif adi < 1.32 and cv2 >= 0.49:
        pattern = "Erratic"
    else:
        pattern = "Lumpy"

    intermittent = zero_pct > 0.5

    # Product lifecycle
    last12, first12 = s.iloc[-12:], s.iloc[:12]
    if last12.sum() == 0:
        plc = "End of Life"
    elif first12.sum() == 0 and s.iloc[: max(n - 12, 0)].sum() == 0:
        plc = "New Launch"
    elif (s > 0).any():
        plc = "Mature"
    else:
        plc = "No Demand"

    variability = "X" if cov < 0.5 else "Y"  # X = stable, Y = variable

    # Trend significance (simple linear regression t-test proxy)
    trend = False
    if n >= 12 and s.std() > 0:
        x = np.arange(n)
        r = np.corrcoef(x, s.values)[0, 1]
        t = abs(r) * np.sqrt(max(n - 2, 1) / max(1 - r ** 2, 1e-9))
        trend = t > 2.0

    # Seasonality check (lag-12 autocorrelation)
    seasonal = False
    if n >= 24 and s.std() > 0:
        a = s.values - s.values.mean()
        ac12 = np.sum(a[12:] * a[:-12]) / np.sum(a * a)
        seasonal = ac12 > 0.30

    return dict(
        zero_pct=round(zero_pct * 100, 1), volume=vol, cov=round(cov, 3) if np.isfinite(cov) else np.nan,
        adi=round(adi, 2) if np.isfinite(adi) else np.nan, cv2=round(cv2, 2) if np.isfinite(cv2) else np.nan,
        pattern=pattern, intermittent=intermittent, plc=plc, variability=variability,
        trend=trend, seasonal=seasonal,
    )


def volume_class(volumes: pd.Series) -> pd.Series:
    """A = items covering top 80% of volume, B = rest."""
    srt = volumes.sort_values(ascending=False)
    cum = srt.cumsum() / max(srt.sum(), 1e-9)
    cls = pd.Series(np.where(cum <= 0.80, "A", "B"), index=srt.index)
    return cls.reindex(volumes.index)


def _rule_for_base(seg: dict) -> tuple:
    """o9-style rule -> candidate model pool (classical models only)."""
    if seg["intermittent"]:
        return "Rule 1", ["Croston", "SBA", "Seasonal Naive", "Moving Average"]
    if seg["plc"] == "End of Life":
        return "Rule 2", ["DES (Holt)", "TES (Holt-Winters)", "Moving Average"]
    if seg["plc"] == "New Launch":
        return "Rule 3", ["DES (Holt)", "Moving Average", "TES (Holt-Winters)", "SES"]
    if seg["plc"] == "Mature":
        if seg["variability"] == "X":
            return ("Rule 4", ["Auto ARIMA", "sARIMA", "STL Forecast", "TES (Holt-Winters)",
                               "Theta", "Prophet", "Neural Net (MLP)", "LightGBM", "XGBoost", "Random Forest"])
        # Y - variable
        if seg["vol_class"] == "B":
            return "Rule 5", ["DES (Holt)", "Moving Average", "Naive", "Random Walk", "SES"]
        if seg["trend"]:
            return ("Rule 6", ["Auto ARIMA", "sARIMA", "TES (Holt-Winters)", "Theta",
                               "Prophet", "LightGBM", "XGBoost", "Neural Net (MLP)"])
        if seg["seasonal"]:
            return ("Rule 7", ["sARIMA", "STL Forecast", "TES (Holt-Winters)",
                               "Prophet", "LightGBM", "Random Forest"])
        return "Rule 6b", ["Auto ARIMA", "SES", "Theta", "XGBoost", "Neural Net (MLP)"]
    return "Rule 8", ["ETS", "SES", "Moving Average"]


def rule_for(seg: dict) -> tuple:
    """o9-style rule -> candidate model pool, with any available zero-shot
    foundation models appended to EVERY rule (per the requirement to make them
    available for all rules). Foundation names resolve at call time, after the
    registry has been populated."""
    rule, pool = _rule_for_base(seg)
    extra = [m for m in globals().get("FOUNDATION_NAMES", []) if m not in pool]
    return rule, pool + extra


# ----------------------------------------------------------------------------- 
# 4. MODEL LIBRARY
# -----------------------------------------------------------------------------
def _safe(fn):
    def wrap(train, h):
        try:
            f = np.asarray(fn(train, h), dtype=float)
            f = np.nan_to_num(f, nan=float(train.mean()))
            return np.clip(f, 0, None)
        except Exception:
            return np.repeat(max(train.mean(), 0), h)
    return wrap


@_safe
def m_naive(train, h):
    return np.repeat(train.iloc[-1], h)


@_safe
def m_seasonal_naive(train, h):
    if len(train) >= 12:
        last = train.iloc[-12:].values
        return np.tile(last, int(np.ceil(h / 12)))[:h]
    return np.repeat(train.iloc[-1], h)


@_safe
def m_moving_avg(train, h, w=6):
    return np.repeat(train.iloc[-w:].mean(), h)


@_safe
def m_random_walk(train, h):
    drift = (train.iloc[-1] - train.iloc[0]) / max(len(train) - 1, 1)
    return train.iloc[-1] + drift * np.arange(1, h + 1)


@_safe
def m_ses(train, h):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    return SimpleExpSmoothing(train, initialization_method="estimated").fit().forecast(h)


@_safe
def m_des(train, h):
    from statsmodels.tsa.holtwinters import Holt
    return Holt(train, damped_trend=True, initialization_method="estimated").fit().forecast(h)


@_safe
def m_tes(train, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    if len(train) < 24:
        raise ValueError("too short")
    add = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12,
                               initialization_method="estimated").fit()
    return add.forecast(h)


@_safe
def m_ets(train, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    return ExponentialSmoothing(train, trend="add", damped_trend=True,
                                initialization_method="estimated").fit().forecast(h)


@_safe
def m_theta(train, h):
    from statsmodels.tsa.forecasting.theta import ThetaModel
    return ThetaModel(train, period=12 if len(train) >= 24 else 1).fit().forecast(h)


@_safe
def m_arima(train, h):
    from statsmodels.tsa.arima.model import ARIMA
    best, best_aic = None, np.inf
    for order in [(1, 1, 1), (0, 1, 1), (1, 0, 1)]:
        try:
            r = ARIMA(train, order=order).fit()
            if r.aic < best_aic:
                best, best_aic = r, r.aic
        except Exception:
            continue
    return best.forecast(h)


@_safe
def m_sarima(train, h):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    r = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return r.forecast(h)


@_safe
def m_stl(train, h):
    from statsmodels.tsa.forecasting.stl import STLForecast
    from statsmodels.tsa.arima.model import ARIMA
    r = STLForecast(train, ARIMA, model_kwargs=dict(order=(1, 1, 1)), period=12).fit()
    return r.forecast(h)


def _croston_core(train, h, alpha=0.1, sba=False):
    y = train.values
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return np.zeros(h)
    z, p = y[nz[0]], nz[0] + 1
    q = 1
    for i in range(nz[0] + 1, len(y)):
        if y[i] > 0:
            z = alpha * y[i] + (1 - alpha) * z
            p = alpha * q + (1 - alpha) * p
            q = 1
        else:
            q += 1
    f = z / max(p, 1e-9)
    if sba:
        f *= (1 - alpha / 2)
    return np.repeat(f, h)


def m_sba(train, h):
    return m_croston_sba(train, h)


@_safe
def m_croston(train, h):
    return _croston_core(train, h, sba=False)


@_safe
def m_croston_sba(train, h):
    return _croston_core(train, h, sba=True)


@_safe
def m_prophet(train, h):
    from prophet import Prophet
    dfp = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(yearly_seasonality=len(train) >= 24, weekly_seasonality=False,
                daily_seasonality=False, uncertainty_samples=0)
    m.fit(dfp)
    fut = m.make_future_dataframe(periods=h, freq="MS")
    return m.predict(fut)["yhat"].values[-h:]


# ---- ML / DL: lag-feature regression -----------------------------------------
def _ml_features(y: np.ndarray, n_lags=12):
    X, t = [], []
    for i in range(n_lags, len(y)):
        lags = y[i - n_lags:i]
        month = (i % 12)
        X.append(np.concatenate([lags, [month, np.mean(lags[-3:]), np.mean(lags)]]))
        t.append(y[i])
    return np.array(X), np.array(t)


def _ml_forecast(train, h, model, n_lags=12):
    y = train.values.astype(float)
    if len(y) <= n_lags + 4:
        n_lags = max(3, len(y) // 2)
    X, t = _ml_features(y, n_lags)
    if len(X) < 5:
        return np.repeat(max(y.mean(), 0), h)
    model.fit(X, t)
    hist = list(y)
    out = []
    for step in range(h):
        lags = np.array(hist[-n_lags:])
        month = (len(hist) % 12)
        x = np.concatenate([lags, [month, np.mean(lags[-3:]), np.mean(lags)]]).reshape(1, -1)
        f = float(model.predict(x)[0])
        f = max(f, 0.0)
        out.append(f)
        hist.append(f)
    return np.array(out)


@_safe
def m_lightgbm(train, h):
    from lightgbm import LGBMRegressor
    return _ml_forecast(train, h, LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                                num_leaves=15, min_child_samples=3, verbose=-1))


@_safe
def m_xgboost(train, h):
    from xgboost import XGBRegressor
    return _ml_forecast(train, h, XGBRegressor(n_estimators=200, learning_rate=0.05,
                                               max_depth=3, verbosity=0))


@_safe
def m_rf(train, h):
    from sklearn.ensemble import RandomForestRegressor
    return _ml_forecast(train, h, RandomForestRegressor(n_estimators=120, min_samples_leaf=2,
                                                        random_state=42, n_jobs=-1))


@_safe
def m_mlp(train, h):
    """Fast deep-learning style model: 2-hidden-layer neural network on lag features."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mdl = make_pipeline(StandardScaler(),
                        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400,
                                     early_stopping=False, random_state=42))
    return _ml_forecast(train, h, mdl)


MODEL_REGISTRY = {
    "Naive": m_naive, "Seasonal Naive": m_seasonal_naive, "Moving Average": m_moving_avg,
    "Random Walk": m_random_walk, "SES": m_ses, "DES (Holt)": m_des,
    "TES (Holt-Winters)": m_tes, "ETS": m_ets, "Theta": m_theta,
    "Auto ARIMA": m_arima, "sARIMA": m_sarima, "STL Forecast": m_stl,
    "Croston": m_croston, "SBA": m_sba, "Prophet": m_prophet,
    "LightGBM": m_lightgbm, "XGBoost": m_xgboost, "Random Forest": m_rf,
    "Neural Net (MLP)": m_mlp,
}

# --- Zero-shot foundation models (optional; only those usable in this env) ----
try:
    from foundation import discover_foundation_models
    FOUNDATION_MODELS = discover_foundation_models()       # {name: callable}
except Exception:
    FOUNDATION_MODELS = {}
MODEL_REGISTRY.update(FOUNDATION_MODELS)
FOUNDATION_NAMES = list(FOUNDATION_MODELS.keys())

# Foundation models are excluded under Fast mode (their first call loads weights,
# which is slow once per process). Power users can turn Fast mode off to include them.
FAST_ONLY_EXCLUDE = {"Prophet", *FOUNDATION_NAMES}


# ----------------------------------------------------------------------------- 
# 5. METRICS + MODEL SELECTION SCORE (per Forecast Master spec)
# -----------------------------------------------------------------------------
def metrics(actual: np.ndarray, fcst: np.ndarray, insample: np.ndarray) -> dict:
    actual, fcst = np.asarray(actual, float), np.asarray(fcst, float)
    err = actual - fcst
    denom = np.where(actual == 0, np.nan, actual)
    mape = float(np.nanmean(np.abs(err / denom)) * 100) if np.isfinite(np.nanmean(np.abs(err / denom))) else 999.0
    wmape = float(np.sum(np.abs(err)) / max(np.sum(np.abs(actual)), 1e-9) * 100)
    nfm = float(np.sum(err) / max(np.sum(actual), 1e-9) * 100)            # net forecast bias %
    mad = np.mean(np.abs(err))
    ts = float(np.sum(err) / max(mad, 1e-9))                              # tracking signal
    naive_mae = np.mean(np.abs(np.diff(insample))) if len(insample) > 1 else 1.0
    mase = float(np.mean(np.abs(err)) / max(naive_mae, 1e-9))
    zmet = float(abs(fcst.mean() - insample.mean()) / max(insample.std(ddof=0), 1e-9))  # reasonability z
    return dict(MAPE=mape, WMAPE=wmape, NFM=abs(nfm), TS=abs(ts), MASE=mase, ZMetric=zmet)


DEFAULT_WEIGHTS = {"MAPE": 0.25, "WMAPE": 0.25, "NFM": 0.15, "TS": 0.10, "ZMetric": 0.10, "MASE": 0.15}


def selection_scores(metric_table: pd.DataFrame, weights: dict) -> pd.Series:
    """Scale each metric 0-1 across models, weight, and sum. Lowest = best."""
    scaled = metric_table.copy()
    for c in scaled.columns:
        lo, hi = scaled[c].min(), scaled[c].max()
        scaled[c] = 0.0 if hi - lo < 1e-12 else (scaled[c] - lo) / (hi - lo)
    w = pd.Series(weights).reindex(scaled.columns).fillna(0)
    if w.sum() > 0:
        w = w / w.sum()
    return (scaled * w).sum(axis=1)


def run_bestfit(s: pd.Series, candidates: list, horizon: int, holdout: int,
                weights: dict, fast_mode: bool = True,
                include_foundation: bool = None, max_models: int = None) -> dict:
    """Backtest candidates on holdout, score, pick winner, refit on full history.

    Performance controls (matter as the model count grows):
      * fast_mode           — drop slow models (Prophet + foundation) for big batches
      * include_foundation  — force-include/exclude zero-shot models regardless of
                              fast_mode (None = follow fast_mode)
      * max_models          — hard cap on how many candidates are backtested
                              (classical models are kept first, foundation last)
    """
    cands = [c for c in candidates if c in MODEL_REGISTRY]

    fnames = set(globals().get("FOUNDATION_NAMES", []))
    if include_foundation is True:
        cands = cands + [m for m in fnames if m in MODEL_REGISTRY and m not in cands]
    elif include_foundation is False:
        cands = [c for c in cands if c not in fnames]
    elif fast_mode:
        cands = [c for c in cands if c not in FAST_ONLY_EXCLUDE] or cands

    if max_models and len(cands) > max_models:
        # keep classical first (cheap), then as many foundation as the cap allows
        classical = [c for c in cands if c not in fnames]
        found = [c for c in cands if c in fnames]
        cands = (classical + found)[:max_models]

    holdout = min(holdout, max(len(s) // 4, 3))
    train, test = s.iloc[:-holdout], s.iloc[-holdout:]

    rows, backtests = {}, {}
    for name in cands:
        f = MODEL_REGISTRY[name](train, holdout)
        rows[name] = metrics(test.values, f, train.values)
        backtests[name] = f
    mt = pd.DataFrame(rows).T
    scores = selection_scores(mt, weights)
    mt["Selection Score"] = scores
    best = scores.idxmin()

    future_idx = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    forecasts = {best: pd.Series(MODEL_REGISTRY[best](s, horizon), index=future_idx)}
    return dict(best=best, metric_table=mt.sort_values("Selection Score"),
                backtests=backtests, test=test, train=train, forecasts=forecasts,
                future_idx=future_idx)


def forecast_all(s: pd.Series, models: list, horizon: int) -> pd.DataFrame:
    future_idx = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    out = {}
    for name in models:
        if name in MODEL_REGISTRY:
            out[name] = MODEL_REGISTRY[name](s, horizon)
    return pd.DataFrame(out, index=future_idx)


def forecast_one(s: pd.Series, model: str, horizon: int) -> pd.Series:
    future_idx = pd.date_range(s.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return pd.Series(MODEL_REGISTRY[model](s, horizon), index=future_idx)


# ----------------------------------------------------------------------------- 
# 6. BATCH / PARALLEL EXECUTION (one self-contained worker per key)
# -----------------------------------------------------------------------------
def batch_worker(args):
    """Process/thread-pool worker: full pipeline for one key.
    args = (key, series, vol_class, method, k, horizon, holdout, weights, fast_mode
            [, include_foundation])
    Returns (key, payload-dict) — everything pandas-pickled, safe across processes."""
    key, s, vol_class, method, k, horizon, holdout, weights, fast = args[:9]
    include_foundation = args[9] if len(args) > 9 else None
    try:
        c, lo, hi, fl = cleanse_series(s, method, k)
        seg = classify_series(c)
        seg["vol_class"] = vol_class
        rule, pool = rule_for(seg)
        res = run_bestfit(c, pool, horizon, holdout, weights, fast,
                          include_foundation=include_foundation)
        res.update(rule=rule, pool=pool, cleansed=c)
        return key, res
    except Exception as exc:                       # never kill the whole batch
        return key, {"error": f"{type(exc).__name__}: {exc}"}


def _run_serial(tasks, progress_cb=None):
    results = {}
    for i, t in enumerate(tasks, 1):
        k, payload = batch_worker(t)
        results[k] = payload
        if progress_cb:
            progress_cb(i, len(tasks))
    return results


def run_batch_parallel(tasks: list, max_workers: int = None, progress_cb=None) -> dict:
    """Run batch_worker over all tasks with the best executor the environment
    allows. Sandboxed hosts (e.g. Streamlit Community Cloud) often block new
    process creation, so the strategy degrades gracefully:

        1. ProcessPoolExecutor (true parallelism, 'spawn' context)
        2. ThreadPoolExecutor  (numpy/statsmodels/LightGBM release the GIL
                                inside their C cores, so threads still overlap)
        3. Serial loop         (always works)
    """
    import multiprocessing
    import os
    from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                    as_completed)
    n = len(tasks)
    workers = max_workers or min(max(os.cpu_count() or 1, 2), 8)
    if n == 1 or workers == 1:
        return _run_serial(tasks, progress_cb)

    def _collect(executor):
        results, done = {}, 0
        futs = [executor.submit(batch_worker, t) for t in tasks]
        for f in as_completed(futs):
            k, payload = f.result()
            results[k] = payload
            done += 1
            if progress_cb:
                progress_cb(done, n)
        return results

    # If any task requests foundation models, run in THREADS first: foundation
    # weights are a per-process singleton, so threads share one load instead of
    # each spawned process re-downloading/re-loading ~hundreds of MB.
    wants_foundation = any(len(t) > 9 and t[9] for t in tasks)
    if wants_foundation:
        try:
            with ThreadPoolExecutor(max_workers=min(workers, 4)) as ex:
                return _collect(ex)
        except Exception:
            return _run_serial(tasks, progress_cb)

    # 1) Process pool — explicit 'spawn' (Python 3.14 defaults to forkserver,
    #    which fails on sandboxed hosts with ConnectionResetError)
    try:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            return _collect(ex)
    except Exception:
        pass  # process creation blocked → degrade to threads

    # 2) Thread pool — safe everywhere, partial parallelism via GIL-releasing C code
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return _collect(ex)
    except Exception:
        pass

    # 3) Serial — guaranteed
    return _run_serial(tasks, progress_cb)
