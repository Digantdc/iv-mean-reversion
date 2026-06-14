"""features.py — per-ticker feature engineering for the calibration model.

Turns each ticker's IV & HV series into a feature row plus a forward-looking
label used to calibrate "will selling volatility have paid off?".

Label (variance-risk-premium realised):
  sell_vol_win = 1 if HV_{t+H} < IV_t   else 0
i.e. did subsequent realised vol come in below the implied vol you'd have sold.

We emit one (features, label) sample per historical day with a valid horizon,
pooled across tickers, for walk-forward calibration in vrp_calibration.py.
"""
from __future__ import annotations
import numpy as np
from volmodels import ar1_phi, half_life


def _pctile(arr, v):
    return 100.0 * np.mean(np.asarray(arr) <= v)


def feature_rows(symbol: str, iv: list[float], hv: list[float],
                 horizon: int = 20, lookback: int = 252):
    """Yield dicts of engineered features + forward label for one ticker."""
    iv = np.asarray(iv, float)
    hv = np.asarray(hv, float)
    n = min(len(iv), len(hv))
    if n < lookback + horizon + 30:
        return
    iv, hv = iv[-n:], hv[-n:]
    for t in range(lookback, n - horizon):
        win_iv = iv[t - lookback:t + 1]
        win_hv = hv[t - lookback:t + 1]
        sd = win_iv.std()
        if sd == 0:
            continue
        phi = ar1_phi(win_iv[-120:])
        feats = {
            "symbol": symbol,
            "iv": float(iv[t]),
            "iv_pctile": float(_pctile(win_iv, iv[t])),
            "iv_z": float((iv[t] - win_iv.mean()) / sd),
            "iv_hv_now": float(iv[t] / hv[t]) if hv[t] else np.nan,
            "iv_hv_avg": float(win_iv.mean() / win_hv.mean()) if win_hv.mean() else np.nan,
            "iv_slope_20": float((iv[t] - iv[t - 20]) / 20),     # recent IV trend
            "hv_pctile": float(_pctile(win_hv, hv[t])),
            "phi": float(phi) if phi == phi else np.nan,
            "half_life": float(min(half_life(phi), 300)) if phi == phi else np.nan,
            # LABEL: selling vol now wins if realised (HV) at t+H < IV now
            "sell_vol_win": int(hv[t + horizon] < iv[t]),
        }
        if any(v != v for k, v in feats.items() if k != "symbol"):
            continue
        yield feats
