"""ivdata.py — cached, resumable IBKR volatility-history layer.

Separates slow data acquisition from fast offline analysis. Each ticker's daily
implied- and historical-volatility series is fetched once and cached to
data_cache/<SYMBOL>.json. Analysis scripts then read the cache instantly and
reproducibly, with no IBKR connection required.

Public API:
  fetch_to_cache(symbols, ...)  -> pull missing/stale tickers from IBKR (resumable)
  load_cached(symbol)           -> dict | None  ({"iv": [...], "hv": [...], "asof": ...})
  load_many(symbols)            -> {symbol: {"iv":[...], "hv":[...]}}
"""
from __future__ import annotations
import asyncio
import datetime as dt
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "data_cache")


def _path(sym: str) -> str:
    return os.path.join(CACHE_DIR, sym.replace("/", "_") + ".json")


def load_cached(symbol: str, max_age_days: int = 7) -> dict | None:
    p = _path(symbol)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception:
        return None
    asof = d.get("asof")
    if asof:
        age = (dt.date.today() - dt.date.fromisoformat(asof)).days
        if age > max_age_days:
            return None
    return d


def load_many(symbols: list[str]) -> dict[str, dict]:
    out = {}
    for s in symbols:
        d = load_cached(s, max_age_days=10_000)  # analysis: accept any cached vintage
        if d and d.get("iv"):
            out[s] = d
    return out


async def _hist(ib, contract, what, duration="2 Y", timeout=45):
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


async def fetch_to_cache(symbols, host="127.0.0.1", port=4001, client_id=240,
                         market_data_type=3, duration="2 Y",
                         pace_every=45, pace_sleep=620, max_age_days=7,
                         min_obs=120):
    """Pull IV+HV for each symbol and cache. Skips fresh cache entries (resumable)."""
    from ib_async import IB, Stock  # local import so analysis scripts need no ib_async

    todo = [s for s in symbols if load_cached(s, max_age_days) is None]
    print(f"{len(symbols)} requested, {len(symbols) - len(todo)} already cached, "
          f"{len(todo)} to fetch.", flush=True)
    if not todo:
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    ib = IB()
    await ib.connectAsync(host, port, clientId=client_id, timeout=25)
    ib.reqMarketDataType(market_data_type)
    nreq = 0
    for i, sym in enumerate(todo, 1):
        nreq += 1
        if nreq % pace_every == 0:
            print(f"  [pacing] sleep {pace_sleep}s after {nreq} reqs "
                  f"({i}/{len(todo)})", flush=True)
            await asyncio.sleep(pace_sleep)
        try:
            c = Stock(sym, "SMART", "USD")
            await ib.qualifyContractsAsync(c)
            iv = await _hist(ib, c, "OPTION_IMPLIED_VOLATILITY", duration)
            if len(iv) < min_obs:
                _write(sym, {"symbol": sym, "iv": [], "hv": [],
                             "asof": dt.date.today().isoformat(),
                             "note": f"insufficient IV ({len(iv)})"})
                print(f"  {sym:6s} skip ({len(iv)} IV obs)", flush=True)
                continue
            hv = await _hist(ib, c, "HISTORICAL_VOLATILITY", duration)
            _write(sym, {"symbol": sym, "iv": iv, "hv": hv,
                         "asof": dt.date.today().isoformat()})
            print(f"  {sym:6s} ok ({len(iv)} IV, {len(hv)} HV)", flush=True)
        except Exception as e:
            print(f"  {sym:6s} ERR {type(e).__name__}", flush=True)
        await asyncio.sleep(0.6)
    ib.disconnect()


def _write(sym: str, payload: dict):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_path(sym), "w") as f:
        json.dump(payload, f)


if __name__ == "__main__":
    import argparse
    from universe import get_universe
    ap = argparse.ArgumentParser(description="Fetch & cache IBKR vol history")
    ap.add_argument("--universe", default="ai_semi", help="ai_semi | sp500")
    ap.add_argument("--host", default=os.getenv("IBKR_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("IBKR_PORT", "4001")))
    ap.add_argument("--client-id", type=int, default=240)
    ap.add_argument("--duration", default="2 Y")
    args = ap.parse_args()
    syms = get_universe(args.universe)
    asyncio.run(fetch_to_cache(syms, host=args.host, port=args.port,
                               client_id=args.client_id, duration=args.duration))
