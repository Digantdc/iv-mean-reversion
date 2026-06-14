"""vrp_calibration.py — calibrated probability that selling volatility wins.

Pools engineered (features, label) samples across the universe, then does a
TIME-ORDERED train/test split (no shuffling — avoids look-ahead leakage),
trains gradient boosting, and CALIBRATES the predicted probabilities with
isotonic regression. Reports the metrics that actually matter for a probability
model:

  * Brier score   (lower = better; vs a base-rate benchmark)
  * ROC AUC       (ranking power)
  * reliability    (predicted vs realised win-rate, printed as a table; PNG if
                    matplotlib is available)

The label is "did subsequent realised vol come in below the implied vol you'd
have sold" — the variance risk premium, realised. A well-calibrated p=0.7 should
win ~70% of the time.

Run:  python vrp_calibration.py --universe sp500
(reads data_cache/, no IBKR needed)
"""
from __future__ import annotations
import argparse
import numpy as np
from features import feature_rows
from ivdata import load_many
from universe import get_universe

FEATURES = ["iv", "iv_pctile", "iv_z", "iv_hv_now", "iv_hv_avg",
            "iv_slope_20", "hv_pctile", "phi", "half_life"]


def build_dataset(data: dict):
    rows = []
    for s, d in data.items():
        rows.extend(feature_rows(s, d.get("iv", []), d.get("hv", [])))
    if not rows:
        return None
    X = np.array([[r[f] for f in FEATURES] for r in rows], float)
    y = np.array([r["sell_vol_win"] for r in rows], int)
    return X, y, rows


def reliability_table(y_true, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum() >= 20:
            out.append((0.5 * (edges[i] + edges[i + 1]),
                        float(p[m].mean()), float(y_true[m].mean()), int(m.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="ai_semi")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--plot", default="calibration.png")
    args = ap.parse_args()

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score

    data = load_many(get_universe(args.universe))
    built = build_dataset(data)
    if not built:
        print("No samples — run ivdata.py to populate the cache first.")
        return
    X, y, rows = built

    # Holdout split. feature_rows() emits oldest->newest WITHIN each ticker; to make
    # the test set genuinely out-of-time we take the most-recent slice of every
    # ticker's own rows and pool those, so no future bars leak into training.
    idx = np.arange(len(rows))
    tickers = np.array([r["symbol"] for r in rows])
    train_mask = np.zeros(len(rows), bool)
    for s in np.unique(tickers):
        si = idx[tickers == s]
        train_mask[si[:int(len(si) * (1 - args.test_frac))]] = True
    Xtr, Xte = X[train_mask], X[~train_mask]
    ytr, yte = y[train_mask], y[~train_mask]
    n, cut = len(y), int(train_mask.sum())

    base_rate = ytr.mean()
    clf = GradientBoostingClassifier(max_depth=3, n_estimators=200, learning_rate=0.05)
    clf.fit(Xtr, ytr)
    p_raw = clf.predict_proba(Xte)[:, 1]

    # isotonic calibration fitted on a holdout slice of TRAIN (last 20% of train)
    cal_cut = int(len(ytr) * 0.8)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(clf.predict_proba(Xtr[cal_cut:])[:, 1], ytr[cal_cut:])
    p_cal = iso.predict(p_raw)

    print(f"Samples: {n:,}  (train {cut:,} / test {n - cut:,})")
    print(f"Base rate (sell-vol-win): {base_rate:.3f}")
    print(f"Brier  base : {brier_score_loss(yte, np.full_like(yte, base_rate, float)):.4f}")
    print(f"Brier  raw  : {brier_score_loss(yte, p_raw):.4f}")
    print(f"Brier  calib: {brier_score_loss(yte, p_cal):.4f}  (lower is better)")
    try:
        print(f"ROC AUC     : {roc_auc_score(yte, p_cal):.3f}")
    except ValueError:
        pass
    print("\nReliability (calibrated):  bin_mid  pred  actual    n")
    tbl = reliability_table(yte, p_cal)
    for mid, pred, act, cnt in tbl:
        print(f"   {mid:5.2f}   {pred:5.2f}  {act:5.2f}  {cnt:5d}")

    # feature importance
    imp = sorted(zip(FEATURES, clf.feature_importances_), key=lambda kv: -kv[1])
    print("\nFeature importance:")
    for f, w in imp:
        print(f"   {f:14s} {w:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        if tbl:
            xs = [t[1] for t in tbl]; ys = [t[2] for t in tbl]
            plt.plot(xs, ys, "o-", color="#3b6fb5", label="calibrated model")
        plt.xlabel("predicted P(sell-vol wins)")
        plt.ylabel("observed win rate")
        plt.title("Reliability — variance-risk-premium model")
        plt.legend(); plt.tight_layout(); plt.savefig(args.plot, dpi=140)
        print(f"\nReliability plot -> {args.plot}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
