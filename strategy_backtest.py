"""strategy_backtest.py — does the assigned strategy actually make money?

The point of the whole project is *which options strategy to trade on each name*.
v1 assigned strategies from IV-behaviour heuristics (G1 calendar / G2-G3 credit
spread / G4 long premium / G5 condor). This module TESTS that hypothesis by
simulating each structure on the name's real price + implied-vol history and
measuring realised P&L — so the recommendation is evidence-based, not asserted.

Strategies simulated (defined-risk, 30-day holding, rolled every 5 trading days):

  long_straddle  : buy 30d ATM call+put at IV_t; payoff = |move| - cost
                   (the G4 "buy vol" test)
  iron_condor    : sell ~1SD strangle + wings at IV_t; win if price stays in range
                   (the G2/G3/G5 "sell vol" test — direction-neutral, defined risk)
  calendar       : sell 30d ATM / buy 60d ATM at IV_t; at t+30 the back leg is
                   re-priced at the ACTUAL IV_{t+30}. Profits when IV is sticky and
                   price pins the strike (the G1 test — uses real forward IV path)

Each trade's P&L is normalised by capital at risk so strategies are comparable.
Metrics per (name, strategy): mean return, win rate, and a simple expectancy.

Run:  python strategy_backtest.py --universe ai_semi --out strategy_backtest.csv
(reads data_cache/ — needs the 'px' field; run augment via ivdata/px fetch first)
"""
from __future__ import annotations
import argparse, csv, math, statistics
import numpy as np
from ivdata import load_many
from universe import get_universe

R = 0.04  # risk-free


def _nd(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(S, K, T, sig, cp=1):
    if K <= 0:
        return max(cp * (S - K), 0.0) if cp == 1 else 0.0
    if T <= 0 or sig <= 0:
        return max(cp * (S - K), 0.0)
    sq = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * sig * sig) * T) / sq
    d2 = d1 - sq
    if cp == 1:
        return S * _nd(d1) - K * math.exp(-R * T) * _nd(d2)
    return K * math.exp(-R * T) * _nd(-d2) - S * _nd(-d1)


def sim_name(iv, px, step=5, hold=30, dte=30):
    """Return {strategy: [returns...]} over rolling windows."""
    n = min(len(iv), len(px))
    iv, px = np.asarray(iv[-n:], float), np.asarray(px[-n:], float)
    T = dte / 252
    out = {"long_straddle": [], "iron_condor": [], "calendar": []}
    for t in range(0, n - hold - 1, step):
        S, sig = px[t], iv[t]
        if S <= 0 or sig <= 0:
            continue
        S2 = px[t + hold]
        move = abs(S2 - S)
        sd = S * sig * math.sqrt(T)  # ~1 std move over the tenor

        # --- long straddle (buy vol) ---
        cost = bs(S, S, T, sig, 1) + bs(S, S, T, sig, -1)
        if cost > 0:
            out["long_straddle"].append((move - cost) / cost)

        # --- iron condor (sell vol, defined risk) ---
        kc, kp = S + sd, max(S - sd, 0.05 * S)        # short strikes ~1SD (put floored)
        wc, wp = kc + sd, max(kp - sd, 0.02 * S)      # long wings ~2SD (floored)
        credit = (bs(S, kc, T, sig, 1) - bs(S, wc, T, sig, 1)
                  + bs(S, kp, T, sig, -1) - bs(S, wp, T, sig, -1))
        width = sd                       # wing width
        if credit > 0 and width > credit:
            loss_c = max(0, min(S2 - kc, width)) if S2 > kc else 0
            loss_p = max(0, min(kp - S2, width)) if S2 < kp else 0
            pnl = credit - loss_c - loss_p
            out["iron_condor"].append(pnl / (width - credit))

        # --- calendar (sell 30d ATM / buy 60d ATM), close at t+30 ---
        if t + hold < n:
            sig2 = iv[t + hold]
            debit = bs(S, S, 2 * dte / 252, sig, 1) - bs(S, S, dte / 252, sig, 1)
            if debit > 0:
                back_val = bs(S2, S, dte / 252, sig2, 1)   # back leg now 30 DTE @ real fwd IV
                front_intrinsic = max(S2 - S, 0.0)
                pnl = (back_val - front_intrinsic) - debit
                out["calendar"].append(pnl / debit)
    return out


def summarise(rets):
    if len(rets) < 10:
        return None
    arr = np.array(rets)
    mean = float(arr.mean())
    win = float((arr > 0).mean())
    sd = float(arr.std()) or 1e-9
    return {"n": len(arr), "mean_ret": round(mean, 3),
            "win_rate": round(win, 2), "sharpe": round(mean / sd, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="ai_semi")
    ap.add_argument("--out", default="strategy_backtest.csv")
    args = ap.parse_args()
    data = load_many(get_universe(args.universe))

    rows = []
    pooled = {"long_straddle": [], "iron_condor": [], "calendar": []}
    for s, d in data.items():
        if not d.get("px"):
            continue
        sims = sim_name(d["iv"], d["px"])
        rec = {"symbol": s}
        best, best_score = None, -1e9
        for strat, rets in sims.items():
            pooled[strat].extend(rets)
            summ = summarise(rets)
            if summ:
                rec[f"{strat}_ret"] = summ["mean_ret"]
                rec[f"{strat}_win"] = summ["win_rate"]
                # rank by expectancy (mean return per trade)
                if summ["mean_ret"] > best_score:
                    best, best_score = strat, summ["mean_ret"]
        rec["best_strategy"] = best
        rec["best_mean_ret"] = round(best_score, 3) if best else None
        rows.append(rec)

    keys = ["symbol", "best_strategy", "best_mean_ret",
            "long_straddle_ret", "long_straddle_win",
            "iron_condor_ret", "iron_condor_win",
            "calendar_ret", "calendar_win"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Strategy backtest — {len(rows)} names, rolling 30d trades, 2y history\n")
    print("Pooled across the universe (mean return per trade, normalised by risk):")
    for strat, rets in pooled.items():
        summ = summarise(rets)
        if summ:
            print(f"  {strat:14s} n={summ['n']:5d}  mean {summ['mean_ret']:+.3f}  "
                  f"win {summ['win_rate']:.2f}  sharpe {summ['sharpe']:+.2f}")
    from collections import Counter
    bc = Counter(r["best_strategy"] for r in rows if r.get("best_strategy"))
    print("\nBest strategy by name (historical expectancy):")
    for strat, cnt in bc.most_common():
        names = [r["symbol"] for r in rows if r.get("best_strategy") == strat]
        print(f"  {strat:14s} {cnt:3d} names: {', '.join(names[:12])}"
              + (" …" if len(names) > 12 else ""))
    print(f"\nPer-name detail -> {args.out}")


if __name__ == "__main__":
    main()
