# IV Mean-Reversion Profiler

A research toolkit that builds a **volatility "personality" profile** for every stock in a
universe (up to the full S&P 500) from 1–2 years of daily implied- and historical-volatility
(Interactive Brokers), and maps each name to the options strategy its vol behaviour supports.

The premise: *implied volatility mean-reverts, but every name reverts differently.* Some
stocks have **sticky** vol that re-elevates after every earnings crush (good for calendars);
others have vol that **evaporates within days** (good for selling credit spreads); a few are
**structurally cheap** (the only place buying premium has an edge). A single IV-percentile
number can't tell these apart — this tool measures the dynamics, then **validates whether the
apparent edge survives out-of-sample.**

> ⚠️ **Research / educational only. Not investment advice.** The code *reads* market data and
> never places orders. See [Findings & limitations](#findings--what-actually-held-up) — several
> "edges" measurably weaken out-of-sample, and that's reported honestly.

## Architecture

Data acquisition is separated from analysis so a 500-name run is feasible and reproducible:

```
ivdata.py        ── fetch IV+HV once, cache to data_cache/<SYM>.json   (slow, resumable)
   │
   ├── iv_behavior.py     profile + strategy group + HAR forecast      (per-name table)
   ├── har_forecast.py    walk-forward AR(1) vs log-AR(1) vs HAR        (model comparison)
   ├── vrp_calibration.py calibrated P(selling vol wins) + validation   (does the edge hold?)
   └── correlation.py     cross-name IV correlation / systemic risk
```

Analysis scripts read the cache — **no IBKR connection needed** once data is fetched.

## Quick start

```bash
git clone https://github.com/Digantdc/iv-mean-reversion.git
cd iv-mean-reversion
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # point at your IB Gateway / TWS

# 1. fetch & cache (IB Gateway must be running; ~3 min for ai_semi, ~3 h for sp500)
python ivdata.py --universe ai_semi       # or: --universe sp500
# 2. analyse (offline, instant, reproducible)
python iv_behavior.py   --from-cache --universe ai_semi --out results.csv
python har_forecast.py  --universe ai_semi
python vrp_calibration.py --universe ai_semi
python correlation.py   --universe ai_semi
```

## What it computes per ticker

| Metric | Meaning |
| --- | --- |
| `iv_pctile`, `z_score` | Where current IV sits in its 1y range (rank and magnitude) |
| `phi`, `phi_log` | AR(1) persistence of IV in levels and in logs — *stickiness* |
| `half_life_days` | Trading days for an IV shock to decay by half |
| `e_iv_20d`, `har_improve_pct` | HAR-model IV forecast, and its OOS RMSE improvement vs AR(1) |
| `n_crush`, `avg_d1_crush_pct`, `recovery_vs_peak/base` | Earnings-crush statistics |
| `iv_hv_avg` | 1y mean IV ÷ mean HV — the **variance risk premium** |
| `group`, `strategy`, `vrp` | Strategy bucket + plain-English play + is the edge real |

### Strategy groups

| Group | Condition | Play |
| --- | --- | --- |
| **G1** Calendar | IV rich & `phi ≥ 0.99` (sticky) | Calendar **after** the earnings crush |
| **G2** Credit, slow | IV rich & half-life > 30d | Credit spread 30–45 DTE, hold to expiry |
| **G3** Credit, fast | IV rich & half-life ≤ 30d | Credit spread 30–45 DTE, take profit early |
| **G4** Long premium | IV ≤ 30th pct | Straddle/strangle with a catalyst in the tenor |
| **G5** No edge | mid-range IV | Iron condor in a quiet window, or skip |

## Methodology

* **Three vol models** (`volmodels.py`): AR(1) on levels, AR(1) on log-vol (proportional
  shocks), and **HAR** (Corsi 2009 — daily + weekly + monthly memory). `walk_forward_rmse()`
  compares their *genuine out-of-sample* 1-day-ahead forecast error on an expanding window
  against a random-walk benchmark.
* **Feature engineering** (`features.py`): IV percentile/z-score, IV/HV ratio (now & 1y avg),
  IV slope, HV percentile, phi, half-life → with a forward label `sell_vol_win = 1[HV_{t+20} < IV_t]`
  (did selling vol actually pay).
* **Probability calibration** (`vrp_calibration.py`): gradient boosting on the pooled features
  with a **per-ticker out-of-time split** (most-recent slice of each name held out — no
  look-ahead), isotonic recalibration, and the metrics that matter for a probability model:
  Brier (vs base-rate benchmark), ROC AUC, and a reliability table/plot.
* **Correlation** (`correlation.py`): average pairwise correlation of daily IV log-changes —
  rising correlation = diversification failing = systemic-stress build-up.

## Findings — what actually held up

Run on a 34-name AI/semiconductor universe (2y data). These are **deliberately honest**,
including the parts that didn't work:

**1. HAR ≈ AR(1) ≈ random walk at the 1-day horizon.** HAR improved OOS RMSE on only ~21% of
names (mean −0.4%). Why: IBKR's `OPTION_IMPLIED_VOLATILITY` is already a 30-day smoothed index,
so its persistence is near-unit-root and the monthly HAR term is nearly collinear with the
level. HAR's documented edge appears at **multi-day horizons and on raw realized vol** — the
code measures this rather than assuming it. *Building the model and reporting that it didn't
beat the baseline here is the point.*

**2. The variance-risk-premium signal is weak out-of-sample.** In-sample / naive-split the
model looked predictive (AUC ≈ 0.64); under a **proper per-ticker out-of-time split it falls to
AUC ≈ 0.56** with Brier no better than the base rate. The IV/HV ratio is consistently the
**top feature by importance** — validating the thesis that IV-vs-realized matters more than
IV percentile — but its standalone forward edge is marginal and regime-dependent. This is the
single most useful result: *the apparent edge mostly does not survive honest validation.*

**3. Cross-name IV correlation is the cleaner signal.** Average pairwise IV-change correlation
was ~0.27, rising to ~0.29 over the recent 60 days; semicap/foundry names (TSM, LRCX, KLAC,
AMAT, NVDA) are the most "systemic" (highest average correlation to the rest) — consistent with
shocks propagating through the AI supply chain together.

See [`sample_output.csv`](sample_output.csv) and [`har_comparison.csv`](har_comparison.csv).

## Limitations

* **1–2 years, one regime.** `phi` near 1.0 is upward-biased in small samples; half-life is
  imprecise. Rankings are meaningful at the extremes, noisy in the middle.
* **In-sample strategy labels.** The G1–G5 groups are descriptive; the calibration module is
  the part that actually tests forward edge — and finds it weak.
* **30-day IV index, not per-option IV.** No skew or term structure; no transaction costs.
* **IV mean-reversion ≠ profit.** Confirmed empirically here — hence the emphasis on validation.

## Roadmap

- [ ] multi-day-horizon HAR + HAR on raw realized vol (where it should beat AR(1))
- [ ] earnings-date alignment for clean crush windows (vs heuristic detection)
- [ ] per-tenor term-structure features; transaction-cost-aware strategy backtest
- [ ] regime-conditioned models (the OOS degradation is largely a regime-shift problem)

## License

MIT — see [LICENSE](LICENSE).
