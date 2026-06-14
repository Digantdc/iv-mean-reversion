"""universe.py — ticker universes for the profiler.

get_sp500() reads the bundled S&P 500 constituents file (Wikipedia/datahub mirror).
get_universe(name) returns a named list. Symbols are normalised to IBKR's dotted
convention (BRK.B, BF.B) where the source uses dashes.
"""
from __future__ import annotations
import csv
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SP500_CSV = os.path.join(_HERE, "sp500_constituents.csv")

# Small curated AI / semiconductor list (default, fast to run).
AI_SEMI = [
    "NVDA", "AMD", "AVGO", "MRVL", "QCOM", "ARM", "TSM", "ASML", "AMAT", "LRCX",
    "KLAC", "TER", "MU", "INTC", "GFS", "TXN", "ADI", "ON", "ORCL", "PLTR",
    "SNOW", "NOW", "CRM", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "TSLA",
    "SMCI", "DELL", "ANET", "VRT", "CRWV",
]


def _norm(sym: str) -> str:
    return sym.strip().upper().replace("-", ".")


def get_sp500() -> list[str]:
    if not os.path.exists(_SP500_CSV):
        raise FileNotFoundError(
            f"{_SP500_CSV} not found. Download from "
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        )
    out = []
    with open(_SP500_CSV) as f:
        for row in csv.DictReader(f):
            sym = _norm(row.get("Symbol", ""))
            if sym:
                out.append(sym)
    return out


def get_universe(name: str) -> list[str]:
    name = name.lower()
    if name in ("sp500", "s&p500", "spx"):
        return get_sp500()
    if name in ("ai", "ai_semi", "semi"):
        return AI_SEMI
    raise ValueError(f"unknown universe '{name}' (use 'sp500' or 'ai_semi')")
