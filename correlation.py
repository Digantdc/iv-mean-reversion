"""correlation.py — cross-sectional IV correlation / systemic-risk analysis.

Measures how tightly the universe's implied vols move together. A high average
pairwise correlation of daily IV *changes* means single-name diversification is
illusory — exactly the condition where a sector-wide shock hits everything at
once (the "correlation goes to 1" crash mechanic).

Run:  python correlation.py --universe sp500
(reads data_cache/, no IBKR needed)
"""
from __future__ import annotations
import argparse
import numpy as np
from ivdata import load_many
from universe import get_universe


def aligned_iv_changes(data: dict, min_len: int = 252):
    """Stack last `min_len` daily IV log-changes across tickers that have them."""
    syms, mats = [], []
    for s, d in data.items():
        iv = np.asarray(d.get("iv", []), float)
        if len(iv) >= min_len + 1:
            chg = np.diff(np.log(iv[-(min_len + 1):]))
            mats.append(chg)
            syms.append(s)
    if len(mats) < 3:
        return [], np.empty((0, 0))
    return syms, np.vstack(mats)


def analyse(data: dict, window: int = 252):
    syms, M = aligned_iv_changes(data, window)
    if M.shape[0] < 3:
        return None
    C = np.corrcoef(M)
    iu = np.triu_indices_from(C, k=1)
    pair = C[iu]
    # recent (last 60d) vs full-window average correlation -> stress gauge
    syms2, Mr = aligned_iv_changes(data, 60)
    pair_recent = None
    if Mr.shape[0] >= 3:
        Cr = np.corrcoef(Mr)
        pair_recent = float(np.mean(Cr[np.triu_indices_from(Cr, k=1)]))
    # average correlation per name (its mean corr to the rest) -> most "systemic"
    per_name = {s: float((C[i].sum() - 1) / (len(syms) - 1)) for i, s in enumerate(syms)}
    top = sorted(per_name.items(), key=lambda kv: -kv[1])[:10]
    return {
        "n_names": len(syms),
        "avg_pairwise_corr": float(np.mean(pair)),
        "median_pairwise_corr": float(np.median(pair)),
        "avg_pairwise_corr_recent60": pair_recent,
        "most_systemic": top,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="ai_semi")
    ap.add_argument("--window", type=int, default=252)
    args = ap.parse_args()
    data = load_many(get_universe(args.universe))
    res = analyse(data, args.window)
    if not res:
        print("Not enough cached names — run ivdata.py first.")
        return
    print(f"Cross-sectional IV correlation ({res['n_names']} names, "
          f"{args.window}d window):")
    print(f"  average pairwise corr : {res['avg_pairwise_corr']:.3f}")
    print(f"  median  pairwise corr : {res['median_pairwise_corr']:.3f}")
    if res["avg_pairwise_corr_recent60"] is not None:
        delta = res["avg_pairwise_corr_recent60"] - res["avg_pairwise_corr"]
        flag = "  <-- RISING (systemic stress building)" if delta > 0.05 else ""
        print(f"  recent 60d avg corr   : {res['avg_pairwise_corr_recent60']:.3f}{flag}")
    print("  most systemic names (highest avg corr to the rest):")
    for s, c in res["most_systemic"]:
        print(f"    {s:6s} {c:.3f}")


if __name__ == "__main__":
    main()
