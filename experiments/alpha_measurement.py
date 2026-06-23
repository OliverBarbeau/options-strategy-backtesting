"""Alpha measurement: strategy vs benchmarks on real data.

Compares our best config (7% buffer + 20% loss limit) against:
- SPY buy-and-hold
- QQQ buy-and-hold (tech-heavy, matches our universe)
- T-bills (risk-free rate)

Computes: CAGR alpha, Sharpe, Sortino, Calmar, max DD, Information Ratio,
correlation to SPY, and capital efficiency analysis.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from tradelab.pipeline import DataPipeline
import pandas as pd

pipe = DataPipeline()

# Our best config actual returns (from long_portfolio_configs.py)
YEARS = ["2018", "2019", "2022", "2023", "2024"]
strat_ret = {"2018": -0.001, "2019": 0.265, "2022": -0.315, "2023": 0.941, "2024": 0.288}

# Benchmark: SPY
spy = pipe.fetch_stock("SPY", start="2017-12-01", end="2025-01-01")
spy_ret = {}
for yr in YEARS:
    s_ts = int(pd.Timestamp(f"{yr}-01-02").timestamp())
    e_ts = int(pd.Timestamp(f"{yr}-12-31").timestamp())
    d = spy[(spy.index >= s_ts) & (spy.index <= e_ts)]
    spy_ret[yr] = d["close"].iloc[-1] / d["close"].iloc[0] - 1

# Benchmark: QQQ
qqq = pipe.fetch_stock("QQQ", start="2017-12-01", end="2025-01-01")
qqq_ret = {}
for yr in YEARS:
    s_ts = int(pd.Timestamp(f"{yr}-01-02").timestamp())
    e_ts = int(pd.Timestamp(f"{yr}-12-31").timestamp())
    d = qqq[(qqq.index >= s_ts) & (qqq.index <= e_ts)]
    qqq_ret[yr] = d["close"].iloc[-1] / d["close"].iloc[0] - 1

# Risk-free
rf_ret = {"2018": 0.019, "2019": 0.022, "2022": 0.015, "2023": 0.052, "2024": 0.053}

# Compound each
def compound(rets):
    cap = 25000
    for yr in YEARS:
        cap *= 1 + rets[yr]
    return cap

strat_final = compound(strat_ret)
spy_final = compound(spy_ret)
qqq_final = compound(qqq_ret)
rf_final = compound(rf_ret)

W = 95

print("=" * W)
print(f"{'ALPHA MEASUREMENT: STRATEGY vs BENCHMARKS':^{W}}")
print(f"{'7% buffer + 20% loss limit | Real Theta Data | 2018-2024':^{W}}")
print("=" * W)

# Per-year comparison
print(f"\n{'Year':<6} {'Strategy':>10} {'SPY':>10} {'QQQ':>10} {'T-Bills':>10}  {'vs SPY':>10} {'vs QQQ':>10}")
print("-" * W)
for yr in YEARS:
    s, sp, q, rf = strat_ret[yr], spy_ret[yr], qqq_ret[yr], rf_ret[yr]
    print(f"{yr:<6} {s*100:>+9.1f}% {sp*100:>+9.1f}% {q*100:>+9.1f}% {rf*100:>+9.1f}%  {(s-sp)*100:>+9.1f}% {(s-q)*100:>+9.1f}%")

# Compounded
s_t = strat_final/25000-1
sp_t = spy_final/25000-1
q_t = qqq_final/25000-1
rf_t = rf_final/25000-1
print("-" * W)
print(f"{'5yr':>6} {s_t*100:>+9.1f}% {sp_t*100:>+9.1f}% {q_t*100:>+9.1f}% {rf_t*100:>+9.1f}%  {(s_t-sp_t)*100:>+9.1f}% {(s_t-q_t)*100:>+9.1f}%")

print(f"\n  Compounded from $25K:")
print(f"    Strategy:  ${strat_final:>10,.0f}")
print(f"    SPY:       ${spy_final:>10,.0f}")
print(f"    QQQ:       ${qqq_final:>10,.0f}")
print(f"    T-Bills:   ${rf_final:>10,.0f}")

# CAGR
n = len(YEARS)
s_cagr = (strat_final/25000)**(1/n) - 1
spy_cagr = (spy_final/25000)**(1/n) - 1
qqq_cagr = (qqq_final/25000)**(1/n) - 1
rf_cagr = (rf_final/25000)**(1/n) - 1

print(f"\n  CAGR:")
print(f"    Strategy:    {s_cagr*100:>+.1f}%")
print(f"    SPY:         {spy_cagr*100:>+.1f}%")
print(f"    QQQ:         {qqq_cagr*100:>+.1f}%")
print(f"    T-Bills:     {rf_cagr*100:>+.1f}%")
print(f"    Alpha(SPY):  {(s_cagr-spy_cagr)*100:>+.1f}% annualized")
print(f"    Alpha(QQQ):  {(s_cagr-qqq_cagr)*100:>+.1f}% annualized")

# Risk metrics
s_arr = np.array([strat_ret[yr] for yr in YEARS])
spy_arr = np.array([spy_ret[yr] for yr in YEARS])
qqq_arr = np.array([qqq_ret[yr] for yr in YEARS])
rf_arr = np.array([rf_ret[yr] for yr in YEARS])

excess_s = s_arr - rf_arr
excess_spy = spy_arr - rf_arr

sharpe_s = np.mean(excess_s) / np.std(excess_s, ddof=1) if np.std(excess_s) > 0 else 0
sharpe_spy = np.mean(excess_spy) / np.std(excess_spy, ddof=1) if np.std(excess_spy) > 0 else 0

# Sortino
down_s = np.sqrt(np.mean(np.minimum(excess_s, 0)**2))
down_spy = np.sqrt(np.mean(np.minimum(excess_spy, 0)**2))
sortino_s = np.mean(excess_s) / down_s if down_s > 0 else 0
sortino_spy = np.mean(excess_spy) / down_spy if down_spy > 0 else 0

# Max drawdown (annual compounded)
s_cum = np.cumprod(1 + s_arr)
spy_cum = np.cumprod(1 + spy_arr)
s_peak = np.maximum.accumulate(s_cum)
spy_peak = np.maximum.accumulate(spy_cum)
s_maxdd = np.min((s_cum - s_peak) / s_peak)
spy_maxdd = np.min((spy_cum - spy_peak) / spy_peak)

# Calmar
calmar_s = s_cagr / abs(s_maxdd) if s_maxdd != 0 else 0
calmar_spy = spy_cagr / abs(spy_maxdd) if spy_maxdd != 0 else 0

# Win counts
beats_spy = sum(1 for yr in YEARS if strat_ret[yr] > spy_ret[yr])
beats_qqq = sum(1 for yr in YEARS if strat_ret[yr] > qqq_ret[yr])

print(f"\n{'=' * W}")
print(f"{'RISK-ADJUSTED METRICS':^{W}}")
print(f"{'=' * W}")
print(f"\n{'Metric':<28} {'Strategy':>12} {'SPY B&H':>12} {'Advantage':>12}")
print("-" * 68)
print(f"{'CAGR':<28} {s_cagr*100:>+11.1f}% {spy_cagr*100:>+11.1f}% {(s_cagr-spy_cagr)*100:>+11.1f}%")
print(f"{'Sharpe Ratio':<28} {sharpe_s:>12.2f} {sharpe_spy:>12.2f} {sharpe_s-sharpe_spy:>+12.2f}")
print(f"{'Sortino Ratio':<28} {sortino_s:>12.2f} {sortino_spy:>12.2f} {sortino_s-sortino_spy:>+12.2f}")
print(f"{'Max Drawdown (annual)':<28} {s_maxdd*100:>11.1f}% {spy_maxdd*100:>11.1f}% {(abs(spy_maxdd)-abs(s_maxdd))*100:>+11.1f}%")
print(f"{'Calmar Ratio (CAGR/MaxDD)':<28} {calmar_s:>12.2f} {calmar_spy:>12.2f} {calmar_s-calmar_spy:>+12.2f}")
print(f"{'Volatility (annual std)':<28} {np.std(s_arr, ddof=1)*100:>11.1f}% {np.std(spy_arr, ddof=1)*100:>11.1f}%")
print(f"{'Years beating SPY':<28} {beats_spy:>10}/5")
print(f"{'Years beating QQQ':<28} {beats_qqq:>10}/5")

# Correlation
corr_spy = np.corrcoef(s_arr, spy_arr)[0, 1]
corr_qqq = np.corrcoef(s_arr, qqq_arr)[0, 1]

# Information ratio
tracking_error = np.std(s_arr - spy_arr, ddof=1)
info_ratio = np.mean(s_arr - spy_arr) / tracking_error if tracking_error > 0 else 0

# Beta
beta = np.cov(s_arr, spy_arr)[0, 1] / np.var(spy_arr, ddof=1)
# Jensen's alpha
jensens = np.mean(s_arr) - (np.mean(rf_arr) + beta * (np.mean(spy_arr) - np.mean(rf_arr)))

print(f"\n{'=' * W}")
print(f"{'PORTFOLIO ANALYTICS':^{W}}")
print(f"{'=' * W}")
print(f"\n{'Metric':<35} {'Value':>15}")
print("-" * 55)
print(f"{'Correlation to SPY':<35} {corr_spy:>+14.3f}")
print(f"{'Correlation to QQQ':<35} {corr_qqq:>+14.3f}")
print(f"{'Beta to SPY':<35} {beta:>+14.3f}")
print(f"{'Jensens Alpha (annualized)':<35} {jensens*100:>+13.1f}%")
print(f"{'Information Ratio (vs SPY)':<35} {info_ratio:>+14.2f}")
print(f"{'Tracking Error vs SPY':<35} {tracking_error*100:>13.1f}%")

# Capital efficiency
print(f"\n{'=' * W}")
print(f"{'CAPITAL EFFICIENCY':^{W}}")
print(f"{'=' * W}")
print(f"""
  Our strategy deploys ~15-25% of capital as collateral at any time.
  The remaining 75-85% sits as cash (earning ~0% in a brokerage sweep).

  If idle capital were invested in T-bills:""")

for deploy_pct in [0.15, 0.20, 0.25]:
    idle = 1 - deploy_pct
    cap = 25000
    for yr in YEARS:
        # Strategy return on full capital + T-bill on idle portion
        # Since strategy return is already on full capital, adding
        # T-bill income on the idle portion is additive
        blended = strat_ret[yr] + idle * rf_ret[yr]
        cap *= 1 + blended
    print(f"    {deploy_pct:.0%} deployed + T-bills on idle: ${cap:,.0f} ({(cap/25000-1)*100:+.1f}%)")

# Hybrid portfolio analysis
print(f"\n  Hybrid portfolios (strategy + SPY):")
for strat_pct in [1.0, 0.75, 0.50, 0.25]:
    spy_pct = 1 - strat_pct
    cap = 25000
    for yr in YEARS:
        blended = strat_pct * strat_ret[yr] + spy_pct * spy_ret[yr]
        cap *= 1 + blended
    print(f"    {strat_pct:.0%} strategy + {spy_pct:.0%} SPY: ${cap:,.0f} ({(cap/25000-1)*100:+.1f}%)")

# Final verdict
print(f"""
{'=' * W}
{'ALPHA VERDICT':^{W}}
{'=' * W}

  The strategy generates {(s_cagr-spy_cagr)*100:+.1f}% annualized alpha over SPY.

  Jensen's Alpha: {jensens*100:+.1f}% (risk-adjusted excess return after accounting
  for market exposure via beta of {beta:+.2f})

  Information Ratio: {info_ratio:+.2f} (excess return per unit of tracking error)
  {'> 0.5 = good, > 1.0 = excellent for systematic strategies' if info_ratio > 0 else ''}

  The alpha is {'REAL but comes with caveats' if jensens > 0 else 'NEGATIVE after risk adjustment'}:
  - Positive in 3/5 years, negative in 2/5
  - Correlation to SPY of {corr_spy:+.2f} means {'partially independent' if abs(corr_spy) < 0.5 else 'highly correlated'} returns
  - Beta of {beta:+.2f} means {'we take on more market risk' if beta > 1 else 'we take on less market risk' if beta > 0 else 'we have inverse market exposure'} than SPY
  - The strategy's edge is concentrated in recovery years (2023: +94%)
    where preserved capital from the loss limit compounds aggressively

  Bottom line: ${strat_final - spy_final:+,.0f} excess over SPY on $25K
  ({(s_t - sp_t)*100:+.1f} percentage points over 5 years)
""")
