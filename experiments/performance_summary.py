"""Performance summary for all simulation accounts."""

import warnings
warnings.filterwarnings("ignore")

import calendar as cal_mod
from tradelab.account import SimulatedAccount
from tradelab.pipeline import DataPipeline
import pandas as pd
import numpy as np

accounts = {
    'pullback_25k': SimulatedAccount.load('accounts/pullback_25k.json'),
    'regime_25k': SimulatedAccount.load('accounts/regime_25k.json'),
    'conservative_25k': SimulatedAccount.load('accounts/conservative_25k.json'),
}
labels = {'pullback_25k': 'Pullback', 'regime_25k': 'Regime Adaptive', 'conservative_25k': 'Conservative'}
keys = ['pullback_25k', 'regime_25k', 'conservative_25k']

W = 105

# Build equity timeseries
equity = {}
for name, acct in accounts.items():
    eq = pd.DataFrame([
        {'date': pd.Timestamp(e.date), 'equity': e.equity}
        for e in acct.equity_curve
    ]).set_index('date')
    eq = eq[~eq.index.duplicated(keep='last')]
    equity[name] = eq


def period_return(name, start, end):
    eq = equity[name]
    ms = eq.index[eq.index >= pd.Timestamp(start)]
    me = eq.index[eq.index <= pd.Timestamp(end)]
    if len(ms) == 0 or len(me) == 0:
        return None
    s = eq.loc[ms[0], 'equity']
    e = eq.loc[me[-1], 'equity']
    return ((e - s) / s * 100, e)


def fmt_ret(val):
    if val is None:
        return "       --      "
    ret, eq = val
    return f"  {ret:>+7.1f}% ({eq/1000:>5.0f}K)"


# =====================================================
print("=" * W)
print(f"{'SIMULATION ACCOUNT PERFORMANCE SUMMARY':^{W}}")
print(f"{'Jan 2024 -> Apr 2026 | $25,000 starting capital | 6 tickers':^{W}}")
print("=" * W)

# PERIOD RETURNS
print()
print(f"{'Period':<20} {'Pullback':>17} {'Regime Adaptive':>17} {'Conservative':>17}")
print("-" * W)

periods = [
    ('Since inception',  '2024-01-01', '2026-04-02'),
    ('',                 None,         None),
    ('2024 Full Year',   '2024-01-01', '2024-12-31'),
    ('  2024 H1',        '2024-01-01', '2024-06-30'),
    ('  2024 H2',        '2024-07-01', '2024-12-31'),
    ('2025 Full Year',   '2025-01-01', '2025-12-31'),
    ('  2025 H1',        '2025-01-01', '2025-06-30'),
    ('  2025 H2',        '2025-07-01', '2025-12-31'),
    ('YTD 2026',         '2026-01-01', '2026-04-02'),
    ('',                 None,         None),
    ('Last 12 months',   '2025-04-01', '2026-04-02'),
    ('Last 6 months',    '2025-10-01', '2026-04-02'),
    ('Last 3 months',    '2026-01-01', '2026-04-02'),
]

for label, start, end in periods:
    if start is None:
        print()
        continue
    row = f"{label:<20}"
    for k in keys:
        row += fmt_ret(period_return(k, start, end))
    print(row)

# QUARTERLY RETURNS
print()
print("=" * W)
print(f"{'QUARTERLY RETURNS':^{W}}")
print("=" * W)
print(f"{'Quarter':<10} {'Pullback':>12} {'Regime':>12} {'Conservative':>14}")
print("-" * 55)

quarters = [
    ('2024 Q1', '2024-01-01', '2024-03-31'),
    ('2024 Q2', '2024-04-01', '2024-06-30'),
    ('2024 Q3', '2024-07-01', '2024-09-30'),
    ('2024 Q4', '2024-10-01', '2024-12-31'),
    ('2025 Q1', '2025-01-01', '2025-03-31'),
    ('2025 Q2', '2025-04-01', '2025-06-30'),
    ('2025 Q3', '2025-07-01', '2025-09-30'),
    ('2025 Q4', '2025-10-01', '2025-12-31'),
    ('2026 Q1', '2026-01-01', '2026-03-31'),
]

for q_name, start, end in quarters:
    row = f"{q_name:<10}"
    for k in keys:
        r = period_return(k, start, end)
        if r is None:
            row += f"{'--':>12}"
        else:
            row += f"  {r[0]:>+9.1f}%"
    print(row)

# ANNUAL RETURNS vs SPY
print()
print("=" * W)
print(f"{'ANNUAL RETURNS vs SPY BUY-AND-HOLD':^{W}}")
print("=" * W)

pipe = DataPipeline()
spy = pipe.fetch_stock('SPY', start='2024-01-01', end='2026-04-04')

print(f"{'Year':<6} {'Pullback':>12} {'Regime':>12} {'Conservative':>14} {'SPY B&H':>10} {'Alpha(PB)':>10}")
print("-" * 70)

for year in [2024, 2025, 2026]:
    row = f"{year:<6}"
    strat_rets = []
    for k in keys:
        r = period_return(k, f'{year}-01-01', f'{year}-12-31' if year < 2026 else '2026-04-02')
        if r:
            strat_rets.append(r[0])
            row += f"  {r[0]:>+9.1f}%"
        else:
            strat_rets.append(None)
            row += f"{'--':>12}"

    # SPY
    yr_start_ts = int(cal_mod.timegm(pd.Timestamp(f'{year}-01-01').timetuple()))
    yr_end_ts = int(cal_mod.timegm(pd.Timestamp(f'{year}-12-31' if year < 2026 else '2026-04-02').timetuple()))
    spy_yr = spy[(spy.index >= yr_start_ts) & (spy.index <= yr_end_ts)]
    spy_ret = None
    if len(spy_yr) >= 2:
        spy_ret = (spy_yr['close'].iloc[-1] - spy_yr['close'].iloc[0]) / spy_yr['close'].iloc[0] * 100
        row += f"  {spy_ret:>+7.1f}%"
    else:
        row += f"{'--':>10}"

    # Alpha
    if strat_rets[0] is not None and spy_ret is not None:
        row += f"  {strat_rets[0] - spy_ret:>+7.1f}%"
    print(row)

# RISK METRICS
print()
print("=" * W)
print(f"{'RISK & PERFORMANCE METRICS':^{W}}")
print("=" * W)
print(f"{'Metric':<30} {'Pullback':>15} {'Regime':>15} {'Conservative':>15}")
print("-" * 80)

for k in keys:
    a = accounts[k]

metrics = [
    ('Starting capital',      lambda a: f"${a.starting_capital:,.0f}"),
    ('Current equity',        lambda a: f"${a.equity:,.0f}"),
    ('Total P/L',             lambda a: f"${a.total_pnl:+,.0f}"),
    ('Total return',          lambda a: f"{a.total_pnl/a.starting_capital*100:+.1f}%"),
    ('CAGR (2.25 yr)',        lambda a: f"{((a.equity/a.starting_capital)**(1/2.25)-1)*100:.1f}%"),
    ('SPACER', None),
    ('Total trades',          lambda a: f"{a.total_trades_count}"),
    ('Win rate',              lambda a: f"{a.win_rate:.1%}"),
    ('Avg P/L per trade',     lambda a: f"${a.total_pnl/a.total_trades_count:+,.0f}" if a.total_trades_count else "N/A"),
    ('Winners',               lambda a: f"{sum(1 for t in a.trades if t.winner)}"),
    ('Losers',                lambda a: f"{sum(1 for t in a.trades if not t.winner)}"),
    ('SPACER', None),
    ('Best trade',            lambda a: f"${max(t.pnl for t in a.trades):+,.0f}" if a.trades else "N/A"),
    ('Worst trade',           lambda a: f"${min(t.pnl for t in a.trades):+,.0f}" if a.trades else "N/A"),
    ('Avg winner',            lambda a: f"${np.mean([t.pnl for t in a.trades if t.winner]):+,.0f}" if any(t.winner for t in a.trades) else "N/A"),
    ('Avg loser',             lambda a: f"${np.mean([t.pnl for t in a.trades if not t.winner]):+,.0f}" if any(not t.winner for t in a.trades) else "N/A"),
    ('SPACER', None),
    ('Open positions',        lambda a: f"{len(a.positions)}"),
    ('Locked collateral',     lambda a: f"${a.locked:,.0f}"),
    ('Available buying power', lambda a: f"${a.balance:,.0f}"),
]

for label, fn in metrics:
    if fn is None:
        print()
        continue
    row = f"{label:<30}"
    for k in keys:
        row += f"{fn(accounts[k]):>15}"
    print(row)

# Max drawdown
row = "Max drawdown               "
for k in keys:
    eq = equity[k]
    peak = eq['equity'].cummax()
    dd = ((eq['equity'] - peak) / peak).min()
    row += f"{dd:>14.1%} "
print(row)

# Quarterly stats
for stat_label, stat_fn in [
    ('Best quarter',    lambda rets: f"{max(rets):+.1f}%"),
    ('Worst quarter',   lambda rets: f"{min(rets):+.1f}%"),
    ('Positive quarters', lambda rets: f"{sum(1 for r in rets if r > 0)}/{len(rets)}"),
]:
    row = f"{stat_label:<30}"
    for k in keys:
        q_rets = []
        for _, s, e in quarters:
            r = period_return(k, s, e)
            if r:
                q_rets.append(r[0])
        if q_rets:
            row += f"{stat_fn(q_rets):>15}"
        else:
            row += f"{'N/A':>15}"
    print(row)

# CURRENT POSITIONS
print()
print("=" * W)
print(f"{'CURRENT OPEN POSITIONS':^{W}}")
print("=" * W)

for k in keys:
    acct = accounts[k]
    n = len(acct.positions)
    print(f"\n  {labels[k]} ({n} positions, ${acct.locked:,.0f} locked):")
    if not acct.positions:
        print("    None")
        continue
    for pos in acct.positions:
        print(f"    {pos.ticker:<5} {pos.contracts:>2}x  "
              f"entry ${pos.entry_price:>7.2f}  "
              f"strikes {pos.short_strike:.0f}/{pos.long_strike:.0f}  "
              f"buffer {pos.buffer:.0%}  "
              f"close {pos.close_target_date[:10]}")

# STRATEGY DESCRIPTIONS
print()
print("=" * W)
print(f"{'STRATEGY DESCRIPTIONS':^{W}}")
print("=" * W)
print("""
  PULLBACK ENTRY
    Entry:  30 DTE put credit spread, only when ticker has pulled back
            3%+ from its 20-day high. 10% buffer, 2% spread width.
    Exit:   Always close at 14 DTE remaining.
    Edge:   Fewest trades, highest win rate, lowest drawdown.
            Captures elevated vol premiums at mean-reversion points.

  REGIME ADAPTIVE
    Entry:  30 DTE put credit spread every cycle. Buffer adapts to
            SPY HV30: 7% in low vol, 10% normal, 13% high vol.
    Exit:   Always close at 14 DTE remaining.
    Edge:   Highest raw returns. Aggressive in calm markets,
            defensive in volatile ones. More trades = more compounding.

  CONSERVATIVE
    Entry:  30 DTE put credit spread every cycle. Fixed 10% buffer,
            2% spread width.
    Exit:   Always close at 14 DTE remaining.
    Edge:   Simplest strategy. Consistent, no regime detection needed.
            Middle ground on returns and risk.

  ALL STRATEGIES:
    Universe:   META, AVGO, MSFT, GOOG, NVDA, CAT
    Sizing:     15% of buying power per position, max 10 contracts
    Max positions: 6 concurrent
    Friction:   $0.65/contract/leg + 2% bid-ask slippage
""")
