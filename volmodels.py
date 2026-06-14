"""volmodels.py — implied-volatility forecasting models.

Three nested models, increasing in sophistication:
  ar1_levels  : x_{t+1} = a + b*x_t                       (the v1 model)
  ar1_log     : log x_{t+1} = a + b*log x_t               (proportional shocks)
  har         : x_{t+1} = a + b_d*x_t + b_w*x_t^(5) + b_m*x_t^(22)
                (Corsi 2009 HAR — daily/weekly/monthly memory)

walk_forward_rmse() compares their genuine out-of-sample 1-day-ahead forecast
error on an expanding window, so "HAR is better" is a measured claim, not an
assertion.
"""
from __future__ import annotations
import math
import numpy as np


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with an intercept column already in X."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def ar1_phi(series) -> float:
    x = np.asarray(series[:-1], float)
    y = np.asarray(series[1:], float)
    if len(x) < 10 or x.std() == 0:
        return float("nan")
    return float(np.cov(x, y, bias=True)[0, 1] / x.var())


def half_life(phi: float) -> float:
    return math.log(0.5) / math.log(phi) if 0 < phi < 1 else float("inf")


def _har_design(v: np.ndarray):
    """Build HAR regressors (daily, weekly=5d avg, monthly=22d avg) and target."""
    n = len(v)
    rows, ys = [], []
    for t in range(22, n - 1):
        daily = v[t]
        weekly = v[t - 4:t + 1].mean()
        monthly = v[t - 21:t + 1].mean()
        rows.append([1.0, daily, weekly, monthly])
        ys.append(v[t + 1])
    return np.array(rows), np.array(ys)


def har_fit(series):
    v = np.asarray(series, float)
    X, y = _har_design(v)
    if len(y) < 30:
        return None
    beta = _ols(X, y)
    return {"intercept": beta[0], "b_daily": beta[1],
            "b_weekly": beta[2], "b_monthly": beta[3]}


def _rmse(pred, actual) -> float:
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def walk_forward_rmse(series, min_train: int = 120):
    """Expanding-window 1-step-ahead OOS RMSE for the three models.
    Returns dict of RMSEs (lower = better) + HAR improvement vs AR1-levels."""
    v = np.asarray(series, float)
    n = len(v)
    if n < min_train + 30:
        return None
    preds = {"ar1_levels": [], "ar1_log": [], "har": [], "rw": []}
    actual = []
    for t in range(min_train, n - 1):
        train = v[:t + 1]
        actual.append(v[t + 1])
        # random walk benchmark
        preds["rw"].append(v[t])
        # AR1 levels
        x, y = train[:-1], train[1:]
        b = np.polyfit(x, y, 1)
        preds["ar1_levels"].append(np.polyval(b, v[t]))
        # AR1 log
        lx, ly = np.log(x), np.log(y)
        bl = np.polyfit(lx, ly, 1)
        preds["ar1_log"].append(math.exp(np.polyval(bl, math.log(v[t]))))
        # HAR
        X, yy = _har_design(train)
        if len(yy) >= 30:
            beta = _ols(X, yy)
            feat = [1.0, v[t], v[t - 4:t + 1].mean(), v[t - 21:t + 1].mean()]
            preds["har"].append(float(np.dot(beta, feat)))
        else:
            preds["har"].append(v[t])
    out = {k: _rmse(p, actual) for k, p in preds.items()}
    if out["ar1_levels"] > 0:
        out["har_improve_vs_ar1_pct"] = 100 * (1 - out["har"] / out["ar1_levels"])
    return out
