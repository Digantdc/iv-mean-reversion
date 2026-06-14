"""har_forecast.py — does HAR actually forecast IV better than AR(1)?

For every cached ticker, runs an expanding-window out-of-sample comparison of
random-walk / AR(1)-levels / AR(1)-log / HAR one-day-ahead IV forecasts, then
aggregates the RMSEs across the universe. This turns "HAR is better" from an
assertion into a measured, reproducible result.

Run:  python har_forecast.py --universe sp500 --out har_comparison.csv
(reads data_cache/, no IBKR needed)
"""
from __future__ import annotations
import argparse
import csv
import numpy as np
from ivdata import load_many
from universe import get_universe
from volmodels import walk_forward_rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="ai_semi")
    ap.add_argument("--out", default="har_comparison.csv")
    args = ap.parse_args()

    data = load_many(get_universe(args.universe))
    rows = []
    for s, d in data.items():
        iv = d.get("iv", [])
        r = walk_forward_rmse(iv)
        if r:
            r["symbol"] = s
            rows.append(r)
    if not rows:
        print("No usable cached series — run ivdata.py first.")
        return

    keys = ["symbol", "rw", "ar1_levels", "ar1_log", "har", "har_improve_vs_ar1_pct"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    arr = {k: np.array([r[k] for r in rows], float)
           for k in ["rw", "ar1_levels", "ar1_log", "har", "har_improve_vs_ar1_pct"]}
    print(f"Out-of-sample 1-day IV forecast RMSE across {len(rows)} names "
          f"(mean, lower=better):")
    print(f"  random walk : {arr['rw'].mean():.4f}")
    print(f"  AR(1) levels: {arr['ar1_levels'].mean():.4f}")
    print(f"  AR(1) log   : {arr['ar1_log'].mean():.4f}")
    print(f"  HAR         : {arr['har'].mean():.4f}")
    print(f"\nHAR vs AR(1)-levels improvement: "
          f"mean {arr['har_improve_vs_ar1_pct'].mean():+.1f}% | "
          f"median {np.median(arr['har_improve_vs_ar1_pct']):+.1f}% | "
          f"HAR wins on {100 * np.mean(arr['har'] < arr['ar1_levels']):.0f}% of names")
    print(f"\nPer-name detail -> {args.out}")


if __name__ == "__main__":
    main()
