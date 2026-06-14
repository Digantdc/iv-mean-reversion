#!/usr/bin/env python
"""
iv_behavior.py — implied-volatility behaviour profiler (v2).

For each ticker, builds a "volatility personality" from 1-2 years of daily
implied- and historical-volatility (Interactive Brokers):

  * IV percentile & z-score      -- how dislocated IV is in its own range
  * AR(1) phi (levels & log) and
    half-life                     -- stickiness / shock decay speed
  * HAR forecast & out-of-sample
    improvement vs AR(1)          -- multi-horizon vol model (Corsi 2009)
  * earnings-crush statistics     -- day-1 drop, 20d recovery vs peak & baseline
  * variance-risk-premium (IV/HV) -- is premium genuinely rich vs realised?

…then maps each name to an options-strategy group (G1 calendar / G2-G3 credit
spread / G4 long premium / G5 no edge).

Two data modes:
  live   : python iv_behavior.py --tickers tickers.example.txt
  cache  : python iv_behavior.py --from-cache --universe sp500
           (reads data_cache/ populated by ivdata.py — no IBKR needed)

DISCLAIMER: research / educational. Reads market data only; never places orders.
"""
from __future__ import annotations
import argparse, asyncio, csv, math, os, statistics, sys
from dataclasses import dataclass, asdict

from volmodels import ar1_phi, half_life, walk_forward_rmse, har_fit


# ----------------------------- statistics ---------------------------------- #
def percentile(series, value):
    return 100.0 * sum(1 for v in series if v <= value) / len(series)


def crush_events(iv, warmup=45, rel_drop=0.15, abs_drop=0.05):
    out, n = [], len(iv)
    for i in range(warmup + 1, n):
        pre, post = iv[i - 1], iv[i]
        if pre - post > abs_drop and (pre - post) / pre > rel_drop:
            p20 = iv[min(i + 20, n - 1)]
            base = statistics.median(iv[i - 45:i - 10]) or float("nan")
            out.append({"d1": 100 * (pre - post) / pre,
                        "rec_peak": 100 * p20 / pre,
                        "rec_base": (p20 / base) if base == base else None})
    return out


def classify(pct, phi, hl, ratio_avg):
    if pct >= 80 and 0 < phi >= 0.99:
        g, play = "G1", "Calendar after the crush (sticky vol)"
    elif pct >= 80 and hl > 30:
        g, play = "G2", "Credit spread ~10% OTM, 30-45 DTE (slow decay)"
    elif pct >= 80:
        g, play = "G3", "Credit spread 30-45 DTE (fast decay; take profit early)"
    elif pct <= 30:
        g, play = "G4", "Long premium straddle/strangle w/ catalyst in tenor"
    else:
        g, play = "G5", "No vol edge — iron condor or skip"
    if ratio_avg is None:
        vrp = "n/a"
    elif ratio_avg >= 1.15:
        vrp = "RICH (selling edge real)"
    elif ratio_avg < 0.90:
        vrp = "UNDERPRICED (buying edge)"
    else:
        vrp = "FAIR (edge is structure)"
    return g, play, vrp


@dataclass
class Profile:
    symbol: str
    iv: float = float("nan")
    iv_mean: float = float("nan")
    iv_pctile: float = float("nan")
    z_score: float = float("nan")
    phi: float = float("nan")
    phi_log: float = float("nan")
    half_life_days: float = float("nan")
    e_iv_20d: float | None = None
    har_improve_pct: float | None = None
    n_crush: int = 0
    avg_d1_crush_pct: float | None = None
    recovery_vs_peak: float | None = None
    recovery_vs_base: float | None = None
    hv: float = float("nan")
    iv_hv_now: float | None = None
    iv_hv_avg: float | None = None
    group: str = ""
    strategy: str = ""
    vrp: str = ""
    note: str = ""


def profile_from_series(symbol, iv, hv, min_obs=120, do_har=True):
    """Pure function: build a Profile from IV/HV arrays (live or cached)."""
    p = Profile(symbol=symbol)
    if len(iv) < min_obs:
        p.note = f"insufficient IV ({len(iv)})"
        return p
    p.iv = round(iv[-1], 4)
    p.iv_mean = round(statistics.mean(iv), 4)
    sd = statistics.pstdev(iv) or float("nan")
    p.iv_pctile = round(percentile(iv, iv[-1]), 1)
    p.z_score = round((iv[-1] - p.iv_mean) / sd, 2) if sd == sd else float("nan")
    phi = ar1_phi(iv)
    p.phi = round(phi, 4) if phi == phi else float("nan")
    logphi = ar1_phi([math.log(v) for v in iv])
    p.phi_log = round(logphi, 4) if logphi == logphi else float("nan")
    hl = half_life(phi)
    p.half_life_days = round(hl, 1) if hl != float("inf") else float("inf")

    if do_har:
        h = har_fit(iv)
        if h:
            import numpy as np
            v = np.asarray(iv, float)
            feat = [1.0, v[-1], v[-5:].mean(), v[-22:].mean()]
            p.e_iv_20d = round(float(h["intercept"] + h["b_daily"] * feat[1]
                                     + h["b_weekly"] * feat[2] + h["b_monthly"] * feat[3]), 4)
        wf = walk_forward_rmse(iv)
        if wf and "har_improve_vs_ar1_pct" in wf:
            p.har_improve_pct = round(wf["har_improve_vs_ar1_pct"], 1)

    ev = crush_events(iv)
    p.n_crush = len(ev)
    if ev:
        p.avg_d1_crush_pct = round(statistics.mean(e["d1"] for e in ev), 1)
        p.recovery_vs_peak = round(statistics.mean(e["rec_peak"] for e in ev), 1)
        recs = [e["rec_base"] for e in ev if e["rec_base"]]
        if recs:
            p.recovery_vs_base = round(statistics.mean(recs), 2)

    ratio_avg = None
    if len(hv) >= 60:
        p.hv = round(hv[-1], 4)
        p.iv_hv_now = round(iv[-1] / hv[-1], 2) if hv[-1] else None
        mhv = statistics.mean(hv)
        ratio_avg = statistics.mean(iv) / mhv if mhv else None
        p.iv_hv_avg = round(ratio_avg, 2) if ratio_avg else None

    p.group, p.strategy, p.vrp = classify(p.iv_pctile, p.phi, hl, ratio_avg)
    return p


# ----------------------------- live mode ----------------------------------- #
async def run_live(tickers, host, port, client_id, mdt, pace_every, pace_sleep, do_har):
    from ib_async import IB, Stock
    ib = IB()
    await ib.connectAsync(host, port, clientId=client_id, timeout=25)
    ib.reqMarketDataType(mdt)

    async def _hist(c, what):
        for _ in range(2):
            try:
                bars = await asyncio.wait_for(
                    ib.reqHistoricalDataAsync(c, "", "2 Y", "1 day", what, True, 1), timeout=45)
                if bars:
                    return [b.close for b in bars if b.close == b.close and b.close > 0]
            except Exception:
                await asyncio.sleep(2)
        return []

    results = []
    for i, sym in enumerate(tickers, 1):
        if i % pace_every == 0:
            print(f"  [pacing] sleep {pace_sleep}s after {i}", flush=True)
            await asyncio.sleep(pace_sleep)
        try:
            c = Stock(sym, "SMART", "USD"); await ib.qualifyContractsAsync(c)
            iv = await _hist(c, "OPTION_IMPLIED_VOLATILITY")
            hv = await _hist(c, "HISTORICAL_VOLATILITY")
            p = profile_from_series(sym, iv, hv, do_har=do_har)
        except Exception as e:
            p = Profile(symbol=sym, note=type(e).__name__)
        results.append(p)
        print(f"  {sym:6s} {p.group or p.note}", flush=True)
        await asyncio.sleep(0.8)
    ib.disconnect()
    return results


def run_cache(tickers, do_har):
    from ivdata import load_cached
    results = []
    for sym in tickers:
        d = load_cached(sym, max_age_days=10_000)
        if not d or not d.get("iv"):
            results.append(Profile(symbol=sym, note="not cached"))
            continue
        results.append(profile_from_series(sym, d["iv"], d.get("hv", []), do_har=do_har))
    return results


def load_tickers(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip().upper()
            if line:
                out.extend(t.strip() for t in line.split(",") if t.strip())
    return out


def main():
    ap = argparse.ArgumentParser(description="IBKR implied-volatility behaviour profiler v2")
    ap.add_argument("--tickers", help="ticker file (live mode)")
    ap.add_argument("--from-cache", action="store_true", help="read data_cache/ (no IBKR)")
    ap.add_argument("--universe", default="ai_semi", help="ai_semi | sp500 (cache mode)")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--no-har", action="store_true", help="skip HAR walk-forward (faster)")
    ap.add_argument("--host", default=os.getenv("IBKR_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("IBKR_PORT", "4001")))
    ap.add_argument("--client-id", type=int, default=int(os.getenv("IBKR_CLIENT_ID", "201")))
    ap.add_argument("--market-data-type", type=int,
                    default=int(os.getenv("IBKR_MARKET_DATA_TYPE", "3")))
    ap.add_argument("--pace-every", type=int, default=48)
    ap.add_argument("--pace-sleep", type=int, default=630)
    args = ap.parse_args()
    do_har = not args.no_har

    if args.from_cache:
        from universe import get_universe
        tickers = get_universe(args.universe)
        print(f"Profiling {len(tickers)} cached tickers ({args.universe}) …")
        results = run_cache(tickers, do_har)
    else:
        if not args.tickers:
            sys.exit("live mode needs --tickers FILE (or use --from-cache)")
        tickers = load_tickers(args.tickers)
        print(f"Profiling {len(tickers)} tickers via IBKR {args.host}:{args.port} …")
        results = asyncio.run(run_live(tickers, args.host, args.port, args.client_id,
                                       args.market_data_type, args.pace_every,
                                       args.pace_sleep, do_har))

    fields = list(asdict(Profile("x")).keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in results:
            w.writerow(asdict(p))
    ok = [p for p in results if p.group]
    census = {g: sum(1 for p in ok if p.group == g) for g in ("G1", "G2", "G3", "G4", "G5")}
    print(f"\nDone: {len(ok)}/{len(results)} profiled -> {args.out}")
    print("Group census:", " | ".join(f"{g} {n}" for g, n in census.items()))
    if ok and any(p.har_improve_pct is not None for p in ok):
        imp = [p.har_improve_pct for p in ok if p.har_improve_pct is not None]
        print(f"HAR vs AR(1) OOS RMSE: mean {statistics.mean(imp):+.1f}% improvement "
              f"({100*sum(1 for x in imp if x>0)/len(imp):.0f}% of names)")


if __name__ == "__main__":
    main()
