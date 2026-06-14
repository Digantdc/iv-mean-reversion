# IV Mean-Reversion Profiler

A research tool that builds a **volatility "personality" profile** for every stock in a
watchlist, using one year of daily implied- and historical-volatility from Interactive
Brokers, and maps each name to the options strategy its vol behaviour actually supports.

The premise: *implied volatility mean-reverts, but every name reverts differently.* Some
stocks have **sticky** vol that re-elevates after every earnings crush (good for
calendars); others have vol that **evaporates within days** (good for selling credit
spreads); a few are **structurally cheap** (the only place buying premium has an edge).
A single IV-percentile number can't tell these apart — this tool measures the dynamics.

> ⚠️ **Research / educational only. Not investment advice.** The code *reads* market data
> and never places orders. Findings are estimated on one year of data and are sensitive to
> regime; see [Limitations](#limitations).

## What it computes per ticker

| Metric | Meaning |
| --- | --- |
| `iv_pctile`, `z_score` | Where current IV sits in its own 1-year range (rank and magnitude) |
| `phi` | AR(1) persistence of IV — *stickiness*. Near 1.0 ⇒ shocks decay slowly |
| `half_life_days` | Trading days for an IV shock to decay by half (`ln 0.5 / ln φ`) |
| `n_crush`, `avg_d1_crush_pct` | Earnings-style 1-day IV crushes detected, and average day-1 drop off the peak |
| `recovery_vs_peak`, `recovery_vs_base` | Where IV sits 20 days post-crush, vs the peak and vs the pre-event baseline |
| `iv_hv_avg` | 1-year mean IV ÷ mean HV — the **variance risk premium**. >1.15 = options genuinely rich; <0.90 = stock realizes more than options price |
| `group`, `strategy`, `vrp` | Strategy bucket + plain-English play + whether the edge is real |

## Strategy groups

| Group | Condition | Play |
| --- | --- | --- |
| **G1** Calendar specialist | IV rich & `phi ≥ 0.99` (sticky) | Calendar spread **after** the earnings crush — the long back-month leg holds its vega while the front decays |
| **G2** Credit spread, slow | IV rich & half-life > 30d | Bear/bull credit spread, 30–45 DTE, held to expiry |
| **G3** Credit spread, fast | IV rich & half-life ≤ 30d | Credit spread, 30–45 DTE, take profit early — vol deflates fast |
| **G4** Long premium | IV ≤ 30th percentile | Straddle/strangle **with a catalyst inside the tenor**; exit into the pre-event IV ramp |
| **G5** No edge | mid-range IV | Iron condor in a quiet window, or skip |

The variance-risk-premium check (`iv_hv_avg`) is layered on top: a high IV percentile only
implies a *real* selling edge when options are also priced above realized vol. In a trending
universe, high IV often just means the stock genuinely moves a lot — `vrp` flags that.

## Quick start

```bash
git clone https://github.com/Digantdc/iv-mean-reversion.git
cd iv-mean-reversion
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # edit host/port for your IB Gateway or TWS

# IB Gateway / TWS must be running with the API enabled
python iv_behavior.py --tickers tickers.example.txt --out results.csv
```

Output is a CSV with one row per ticker plus a printed group census, e.g.:

```
Done: 32/34 profiled -> results.csv
Group census: G1 3 | G2 0 | G3 13 | G4 3 | G5 13
```

## Sample output

Running the bundled `tickers.example.txt` (34 large-cap AI / semiconductor names) produced
[`sample_output.csv`](sample_output.csv). Group census:

```
Group census: G1 3 | G2 4 | G3 13 | G4 3 | G5 11
```

One representative name per group (all figures market-derived, 1-year IBKR data):

| Symbol | Group | IV | IV %ile | phi | half-life | IV/HV (1y) | Read |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MU | **G1** | 1.06 | 100 | 0.994 | 112d | 1.02 | Sticky vol → calendar after the earnings crush |
| AMAT | **G2** | 0.73 | 100 | 0.989 | 61d | 1.16 | Rich + slow decay → 30-45 DTE credit spread (real selling edge, IV/HV>1.15) |
| TSM | **G3** | 0.46 | 92 | 0.951 | 14d | 1.04 | Rich + fast decay → credit spread, take profit early |
| AMD | **G4-ish** | 0.72 | 99 | 0.963 | 18d | **0.86** | High percentile but options priced *below* realized vol → a buying tell, not a selling one |
| AAPL | **G4** | 0.23 | 22 | 0.911 | 7d | 1.06 | Cheap vol → long premium with a catalyst in the tenor |
| NVDA | **G5** | 0.37 | 37 | 0.934 | 10d | 1.14 | Mid-range → no clear vol edge |

The AMD row shows why the variance-risk-premium check matters: a 99th-percentile IV looks
like a screaming sell, but `iv_hv_avg = 0.86` means the stock has been realizing *more*
volatility than its options price — the opposite signal. Percentile alone would mislead here.

> Numbers are a snapshot from one run and will differ as markets move. Regenerate any time
> with `python iv_behavior.py --tickers tickers.example.txt --out sample_output.csv`.

## Method notes

* **Data source:** IBKR `OPTION_IMPLIED_VOLATILITY` and `HISTORICAL_VOLATILITY` daily bars
  (a 30-day ATM-interpolated IV index, not a single option's IV). Delayed data is sufficient.
* **AR(1)** is the discrete-time Ornstein–Uhlenbeck process at the core of the Heston model —
  the simplest interpretable mean-reversion estimator. `phi` is the OLS slope of `IV_t` on `IV_{t-1}`.
* **Crush detection** is heuristic: a >15% relative *and* >5 vol-point single-day IV drop, after a
  45-day warm-up so a pre-event baseline exists. Events are not cross-checked against earnings dates.
* **Rate limiting:** IBKR allows ~60 historical requests per 10 minutes; the script paces itself
  (`--pace-every` / `--pace-sleep`) so large universes complete without throttling.

## Limitations

Read these before trusting any output:

* **One year, one regime.** `phi` near 1.0 is biased upward in small samples; `half_life` is
  imprecise (the gap between φ=0.99 and 0.995 doubles it). Treat rankings as meaningful at the
  extremes, noisy in the middle.
* **AR(1) on levels** is the bluntest model — log-vol AR(1) (proportional shocks) and
  [HAR](https://en.wikipedia.org/wiki/Heterogeneous_autoregressive_model) (multi-horizon) both
  forecast vol better. A natural next step.
* **In-sample, no costs.** Strategy labels are descriptive, not a backtest. Real edge requires
  out-of-sample testing net of commissions and bid/ask, plus probability calibration.
* **IV mean-reversion ≠ profit.** A credit spread's P&L is mostly realized path + theta; the
  documented options-selling edge is the *variance risk premium* (IV > subsequent realized vol),
  which is why `iv_hv_avg` is included as a reality check.

## Roadmap

- [ ] log-vol AR(1) and HAR forecasts
- [ ] earnings-date alignment for clean crush windows (vs heuristic detection)
- [ ] per-tenor term structure instead of the 30-day IV index
- [ ] walk-forward strategy backtest with transaction costs and calibrated probabilities

## License

MIT — see [LICENSE](LICENSE).
