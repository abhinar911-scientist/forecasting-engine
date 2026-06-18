"""
Zero-shot foundation forecasting models — optional, pluggable, fail-safe.

Design goals
------------
* The app must stay deployable on light hosts (Streamlit Community Cloud:
  CPU-only, ~1 GB RAM). So every foundation model is OPTIONAL: if its package
  is not installed, or there is not enough RAM, the model simply does not appear
  in the registry — nothing breaks.
* Models are loaded LAZILY and exactly ONCE per process (singleton cache). The
  expensive cost is the weight load; per-series inference is cheap, so we never
  re-instantiate a model per key.
* Each adapter exposes the same call signature as the classical models:
      fn(train: pd.Series, horizon: int) -> np.ndarray  (length == horizon, >=0)
  so they drop straight into MODEL_REGISTRY and every rule pool.

Availability is decided at import time by `discover_foundation_models()`, which
returns a dict {display_name: callable}. Anything that errors is skipped.
"""
import importlib.util
import os
import threading

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- 
# Environment guards
# --------------------------------------------------------------------------- 
def _available_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        # psutil not present — read /proc as a fallback (Linux hosts)
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) / (1024 ** 2)
        except Exception:
            pass
        return 999.0  # unknown → don't block


def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


# Allow operators to force-disable foundation models (e.g. to guarantee a light
# footprint) via env var FORECAST_DISABLE_FOUNDATION=1
_DISABLED = os.getenv("FORECAST_DISABLE_FOUNDATION", "0") == "1"

# RAM each model needs to load safely (GB). Used to skip heavy models on small hosts.
_RAM_NEED = {"Chronos-Bolt": 0.6, "MOIRAI": 0.6, "TimesFM": 2.0}

# --------------------------------------------------------------------------- 
# Singleton model cache (one instance per process, thread-safe)
# --------------------------------------------------------------------------- 
_CACHE = {}
_LOCK = threading.Lock()


def _singleton(key, builder):
    if key in _CACHE:
        return _CACHE[key]
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = builder()
    return _CACHE[key]


def _clip(arr, horizon, fallback_mean):
    a = np.asarray(arr, dtype=float).ravel()[:horizon]
    if a.size < horizon:
        a = np.concatenate([a, np.repeat(fallback_mean, horizon - a.size)])
    a = np.nan_to_num(a, nan=fallback_mean, posinf=fallback_mean, neginf=0.0)
    return np.clip(a, 0, None)


# --------------------------------------------------------------------------- 
# Chronos-Bolt (Amazon) — pip install chronos-forecasting
# --------------------------------------------------------------------------- 
def _build_chronos():
    import torch
    from chronos import BaseChronosPipeline
    # Bolt-small is light (~200 MB) and CPU-friendly; the best speed/size trade-off.
    return BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)


def _chronos_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    mean_val = max(float(train.mean()), 0.0)
    try:
        import torch
        pipe = _singleton("chronos", _build_chronos)
        ctx = torch.tensor(np.asarray(train.values, dtype=np.float32))
        q, mean = pipe.predict_quantiles(
            context=ctx, prediction_length=horizon, quantile_levels=[0.5])
        out = mean.numpy() if hasattr(mean, "numpy") else np.asarray(mean)
        return _clip(out, horizon, mean_val)
    except Exception:
        return np.repeat(mean_val, horizon)


# --------------------------------------------------------------------------- 
# MOIRAI (Salesforce) — pip install uni2ts
# --------------------------------------------------------------------------- 
def _build_moirai():
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")
    return module


def _moirai_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    mean_val = max(float(train.mean()), 0.0)
    try:
        import torch
        from uni2ts.model.moirai import MoiraiForecast
        module = _singleton("moirai_module", _build_moirai)
        ctx_len = min(len(train), 200)
        model = MoiraiForecast(
            module=module, prediction_length=horizon, context_length=ctx_len,
            patch_size=8, num_samples=20, target_dim=1,
            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        predictor = model.create_predictor(batch_size=1)
        vals = np.asarray(train.values, dtype=np.float32)[-ctx_len:]
        # Build a minimal GluonTS-style input
        past = torch.tensor(vals).reshape(1, ctx_len, 1)
        observed = torch.ones_like(past)
        pad = torch.zeros(1, ctx_len, dtype=torch.bool)
        with torch.no_grad():
            fc = model(
                past_target=past, past_observed_target=observed.bool(),
                past_is_pad=pad)
        out = fc.cpu().numpy()
        out = np.median(out, axis=1).ravel() if out.ndim >= 2 else out.ravel()
        return _clip(out, horizon, mean_val)
    except Exception:
        return np.repeat(mean_val, horizon)


# --------------------------------------------------------------------------- 
# TimesFM (Google) — pip install timesfm (heavy: ~1.5 GB RAM)
# --------------------------------------------------------------------------- 
def _build_timesfm():
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(
        max_context=1024, max_horizon=256, normalize_inputs=True,
        use_continuous_quantile_head=True, force_flip_invariance=True,
        infer_is_positive=True, fix_quantile_crossing=True))
    return model


def _timesfm_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    mean_val = max(float(train.mean()), 0.0)
    try:
        model = _singleton("timesfm", _build_timesfm)
        point, _ = model.forecast(
            horizon=horizon, inputs=[np.asarray(train.values, dtype=np.float32)])
        return _clip(np.asarray(point)[0], horizon, mean_val)
    except Exception:
        return np.repeat(mean_val, horizon)


# --------------------------------------------------------------------------- 
# TimeGPT (Nixtla) — API model, needs NIXTLA_API_KEY (sends data to Nixtla)
# --------------------------------------------------------------------------- 
def _timegpt_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    mean_val = max(float(train.mean()), 0.0)
    try:
        from nixtla import NixtlaClient
        key = os.getenv("NIXTLA_API_KEY", "")
        if not key:
            return np.repeat(mean_val, horizon)
        client = _singleton("timegpt_client", lambda: NixtlaClient(api_key=key))
        dfp = pd.DataFrame({"ds": train.index, "y": train.values})
        fc = client.forecast(df=dfp, h=horizon, time_col="ds", target_col="y", freq="MS")
        col = [c for c in fc.columns if c not in ("ds",)][0]
        return _clip(fc[col].values, horizon, mean_val)
    except Exception:
        return np.repeat(mean_val, horizon)


# --------------------------------------------------------------------------- 
# Discovery — decide which foundation models are usable in THIS environment
# --------------------------------------------------------------------------- 
_SPEC = [
    # display_name,        adapter,             package,         ram_key
    ("Chronos-Bolt (ZS)",  _chronos_forecast,   "chronos",       "Chronos-Bolt"),
    ("MOIRAI (ZS)",        _moirai_forecast,    "uni2ts",        "MOIRAI"),
    ("TimesFM (ZS)",       _timesfm_forecast,   "timesfm",       "TimesFM"),
    ("TimeGPT (ZS)",       _timegpt_forecast,   "nixtla",        None),
]


def discover_foundation_models() -> dict:
    """Return {display_name: callable} for every foundation model that can
    actually run here. Missing packages / insufficient RAM → silently skipped."""
    if _DISABLED:
        return {}
    avail = _available_ram_gb()
    out = {}
    for name, fn, pkg, ram_key in _SPEC:
        if not _has(pkg):
            continue
        if ram_key is not None and avail < _RAM_NEED.get(ram_key, 0.6):
            continue
        if pkg == "nixtla" and not os.getenv("NIXTLA_API_KEY"):
            continue  # API model with no key → not usable
        out[name] = fn
    return out


def foundation_status() -> list:
    """Human-readable availability table for the UI (name, installed, usable, note)."""
    avail = _available_ram_gb()
    rows = []
    for name, fn, pkg, ram_key in _SPEC:
        installed = _has(pkg)
        note = ""
        usable = installed
        if not installed:
            note = f"not installed (pip install {pkg})"
            usable = False
        elif ram_key and avail < _RAM_NEED.get(ram_key, 0.6):
            note = f"needs ~{_RAM_NEED[ram_key]:.1f} GB RAM (have {avail:.1f})"
            usable = False
        elif pkg == "nixtla" and not os.getenv("NIXTLA_API_KEY"):
            note = "set NIXTLA_API_KEY to enable"
            usable = False
        else:
            note = "ready"
        rows.append(dict(Model=name, Installed=installed, Usable=usable, Note=note))
    return rows
