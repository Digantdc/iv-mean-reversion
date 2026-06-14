#!/usr/bin/env python
"""
iv_behavior.py — Implied-volatility behaviour profiler for options traders.

For each ticker, pulls 1 year of daily implied- and historical-volatility from
Interactive Brokers and computes a per-name "volatility personality":

  * IV percentile & z-score   -- how dislocated is IV in its own 1y range
  * AR(1) phi & half-life      -- how sticky is IV / how fast do shocks decay
  * earnings-crush statistics  -- day-1 IV drop, 20-day recovery vs peak & baseline
  * variance-risk-premium      -- 1y mean IV / mean HV (is premium really rich?)

It then maps each name to an options-strategy group:

  G1  CALENDAR specialist   phi >= 0.99, IV rich   (sticky vol; trade calendars
                                                    AFTER the earnings crush)
  G2  CREDIT SPREAD, slow   IV rich, half-life > 30d
  G3  CREDIT SPREAD, fast   IV rich, half-life <= 30d (best premium-selling home)
  G4  LONG PREMIUM          IV cheap (<=30th pct)   (buy straddles pre-catalyst)
  G5  NO EDGE               mid-range IV            (iron condor or skip)

Usage:
  pip install -r requirements.txt
  cp .env.example .env          # set IBKR_HOST / IBKR_PORT / IBKR_CLIENT_ID
  python iv_behavior.py --tickers tickers.example.txt --out results.csv

Requires a running IB Gateway or TWS with API enabled. Delayed data is fine.

DISCLAIMER: research / educational tool. Not investment advice. No orders are
ever placed — this code only reads market data.
"""
from __future__ import annotations
import argparse, asyncio, csv, math, os, statistics, sys
from dataclasses import dataclass, asdict

try:
    from ib_async import IB, Stock
except ImportError:
    sys.exit("ib_async not installed — run: pip install -r requirements.txt")


# ----------------------------- statistics ---------------------------------- #
def percentile(series: list[float], value: float) -> float:
    return 100.0 * sum(1 for v in series if v <= value) / len(series)


def ar1_phi(series: list[float]) -> float:
    """OLS slope of x_t on x_{t-1} — the AR(1) persistence coefficient."""
    x, y = series[:-1], series[1:]
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = sum((a - mx) ** 2 for a in x)
    return num / den if den else float("nan")


def half_life(phi: float) -> float:
    """Trading days for an AR(1) shock to decay by half. inf if non-stationary."""
    return math.log(0.5) / math.log(phi) if 0 < phi < 1 else float("inf")


def crush_events(iv: list[float], warmup: int = 45,
                 rel_drop: float = 0.15, abs_drop: float = 0.05):
    """Detect earnings-style 1-day IV crushes (>15% relative AND >5 vol pts).
    Baseline = median IV from 45..10 trading days before the event."""
    out = []
    n = len(iv)
    for i in range(warmup + 1, n):
        pre, post = iv[i - 1], iv[i]
        if pre - post > abs_drop and (pre - post) / pre > rel_drop:
            p20 = iv[min(i + 20, n - 1)]
            base = statistics.median(iv[i - 45:i - 10]) or float("nan")
            out.append({
                "d1_drop_pct": 100 * (pre - post) / pre,
                "recovery_vs_peak": 100 * p20 / pre,
                "recovery_vs_base": (p20 / base) if base and base == base else None,
            })
    return out


# ----------------------------- classification ------------------------------ #
def classify(pct: float, phi: float, hl: float, ratio_avg: float | None):
    if pct >= 80 and 0 < phi and phi >= 0.99:
        group, play = "G1", "Calendar after the crush (sticky vol)"
    elif pct >= 80 and hl > 30:
        group, play = "G2", "Credit spread ~10% OTM, 30-45 DTE (slow decay)"
    elif pct >= 80:
        group, play = "G3", "Credit spread 30-45 DTE (fast decay; take profit early)"
    elif pct <= 30:
        group, play = "G4", "Long premium straddle/strangle w/ catalyst in tenor"
    else:
        group, play = "G5", "No vol edge — iron condor in quiet window or skip"

    if ratio_avg is None:
        vrp = "n/a"
    elif ratio_avg >= 1.15:
        vrp = "RICH (selling edge real)"
    elif ratio_avg < 0.90:
        vrp = "UNDERPRICED (buying edge)"
    else:
        vrp = "FAIR (edge is structure, not mispricing)"
    return group, play, vrp


@dataclass
class Profile:
    symbol: str
    iv: float = float("nan")
    iv_mean: float = float("nan")
    iv_pctile: float = float("nan")
    z_score: float = float("nan")
    phi: float = float("nan")
    half_life_days: float = float("nan")
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


# ----------------------------- IBKR plumbing ------------------------------- #
async def _hist(ib: "IB", contract, what: str, duration="1 Y", timeout=45):
    """One historical request with a timeout + single retry."""
    for _ in range(2):
        try:
            bars = await asyncio.wait_for(
                ib.reqHistoricalDataAsync(contract, "", duration, "1 day", what, True, 1),
                timeout=timeout)
            if bars:
                return [b.close for b in bars if b.close == b.close and b.close > 0]
        except Exception:
            await asyncio.sleep(2)
    return []


async def profile_ticker(ib: "IB", symbol: str, min_obs: int = 120) -> Profile:
    p = Profile(symbol=symbol)
    try:
        c = Stock(symbol, "SMART", "USD")
        await ib.qualifyContractsAsync(c)
    except Exception:
        p.note = "no contract"
        return p
    iv = await _hist(ib, c, "OPTION_IMPLIED_VOLATILITY")
    if len(iv) < min_obs:
        p.note = f"insufficient IV history ({len(iv)})"
        return p
    hv = await _hist(ib, c, "HISTORICAL_VOLATILITY")

    p.iv = round(iv[-1], 4)
    p.iv_mean = round(statistics.mean(iv), 4)
    sd = statistics.pstdev(iv) or float("nan")
    p.iv_pctile = round(percentile(iv, iv[-1]), 1)
    p.z_score = round((iv[-1] - p.iv_mean) / sd, 2) if sd == sd else float("nan")
    p.phi = round(ar1_phi(iv), 4)
    hl = half_life(p.phi)
    p.half_life_days = round(hl, 1) if hl != float("inf") else float("inf")

    ev = crush_events(iv)
    p.n_crush = len(ev)
    if ev:
        p.avg_d1_crush_pct = round(statistics.mean(e["d1_drop_pct"] for e in ev), 1)
        p.recovery_vs_peak = round(statistics.mean(e["recovery_vs_peak"] for e in ev), 1)
        recs = [e["recovery_vs_base"] for e in ev if e["recovery_vs_base"]]
        if recs:
            p.recovery_vs_base = round(statistics.mean(recs), 2)

    ratio_avg = None
    if len(hv) >= 60:
        p.hv = round(hv[-1], 4)
        p.iv_hv_now = round(iv[-1] / hv[-1], 2) if hv[-1] else None
        ratio_avg = statistics.mean(iv) / statistics.mean(hv) if statistics.mean(hv) else None
        p.iv_hv_avg = round(ratio_avg, 2) if ratio_avg else None

    p.group, p.strategy, p.vrp = classify(p.iv_pctile, p.phi, hl, ratio_avg)
    return p


async def run(tickers: list[str], host: str, port: int, client_id: int,
              market_data_type: int, pace_every: int, pace_sleep: int):
    ib = IB()
    await ib.connectAsync(host, port, clientId=client_id, timeout=25)
    ib.reqMarketDataType(market_data_type)
    results: list[Profile] = []
    for i, sym in enumerate(tickers, 1):
        if i % pace_every == 0:
            print(f"  [pacing] sleeping {pace_sleep}s after {i} requests…", flush=True)
            await asyncio.sleep(pace_sleep)
        prof = await profile_ticker(ib, sym)
        results.append(prof)
        tag = prof.group or prof.note
        print(f"  {sym:6s} {tag}", flush=True)
        await asyncio.sleep(0.8)
    ib.disconnect()
    return results


# ----------------------------- CLI ----------------------------------------- #
def load_tickers(path: str) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip().upper()
            if line:
                out.extend(t.strip() for t in line.split(",") if t.strip())
    return out


def main():
    ap = argparse.ArgumentParser(description="IBKR implied-volatility behaviour profiler")
    ap.add_argument("--tickers", default="tickers.example.txt",
                    help="file with tickers (one per line or comma-separated; # comments ok)")
    ap.add_argument("--out", default="results.csv", help="output CSV path")
    ap.add_argument("--host", default=os.getenv("IBKR_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("IBKR_PORT", "4001")))
    ap.add_argument("--client-id", type=int, default=int(os.getenv("IBKR_CLIENT_ID", "201")))
    ap.add_argument("--market-data-type", type=int,
                    default=int(os.getenv("IBKR_MARKET_DATA_TYPE", "3")),
                    help="1=live 2=frozen 3=delayed 4=delayed-frozen")
    ap.add_argument("--pace-every", type=int, default=48,
                    help="pause after this many requests (IBKR ~60/10min limit)")
    ap.add_argument("--pace-sleep", type=int, default=630)
    args = ap.parse_args()

    tickers = load_tickers(args.tickers)
    print(f"Profiling {len(tickers)} tickers via IBKR {args.host}:{args.port} …")
    results = asyncio.run(run(tickers, args.host, args.port, args.client_id,
                              args.market_data_type, args.pace_every, args.pace_sleep))

    fields = list(asdict(results[0]).keys()) if results else []
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in results:
            w.writerow(asdict(p))

    ok = [p for p in results if p.group]
    census = {g: sum(1 for p in ok if p.group == g) for g in ("G1", "G2", "G3", "G4", "G5")}
    print(f"\nDone: {len(ok)}/{len(results)} profiled -> {args.out}")
    print("Group census:", " | ".join(f"{g} {n}" for g, n in census.items()))


if __name__ == "__main__":
    main()
