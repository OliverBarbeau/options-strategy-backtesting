"""Aggressive Pullback Backtest: risk-tolerant individual account.

Compares aggressive pullback against the conservative pullback baseline
across the full 10-year period. Tests the impact of:
- Tighter buffers (7% vs 10%) for richer premiums
- Wider spreads (3% vs 2%) for more credit per trade
- Position stacking on deep pullbacks (5%+)
- Streak bonus: tighten buffer to 6% after consecutive winners
"""

import warnings
warnings.filterwarnings("ignore")

from tradelab.pipeline import DataPipeline
from tradelab.options import historical_volatility
from tradelab.strategies.aggressive_pullback import AggressivePullbackStrategy
from tradelab.strategies.pullback_entry import PullbackEntryStrategy
import pandas as pd
import numpy as np

pipe = DataPipeline()

TICKERS = ['META', 'AVGO', 'MSFT', 'GOOG', 'NVDA', 'CAT', 'QQQ', 'AMD']
START = '2016-01-01'
END = '2026-04-05'
W = 115

print("Loading data...")
DATA = {}
for t in TICKERS:
    DATA[t] = pipe.fetch_stock(t, start=START, end=END)
    print(f"  {t}: {len(DATA[t])} rows")

# =====================================================
# RUN BOTH STRATEGIES
# =====================================================

aggressive = AggressivePullbackStrategy(
    buffer=0.07,
    spread_pct=0.03,
    pullback_threshold=0.03,
    deep_pullback=0.05,
    streak_bonus_threshold=2,
    streak_buffer=0.06,
)

conservative = PullbackEntryStrategy(
    buffer=0.10,
    spread_pct=0.02,
    pullback_threshold=0.03,
)

agg_results = {}
con_results = {}
for t in TICKERS:
    agg_results[t] = aggressive.run(DATA[t])
    con_results[t] = conservative.run(DATA[t])

# =====================================================
print()
print("=" * W)
print(f"{'AGGRESSIVE vs CONSERVATIVE PULLBACK':^{W}}")
print(f"{'10-Year Backtest | Risk-Tolerant Individual Account':^{W}}")
print("=" * W)

# --- HEAD-TO-HEAD ---
print()
print(f"  {'':.<6} {'|':>2} {'--- Aggressive (7% buf, 3% width) ---':^40} {'|':>2} {'--- Conservative (10% buf, 2% width) ---':^40}")
print(f"  {'Tkr':<6} {'|':>2} {'Trades':>6} {'WR':>6} {'P/L':>10} {'$/Trd':>8} {'MaxDD%':>8} {'Stk':>4} {'|':>2} {'Trades':>6} {'WR':>6} {'P/L':>10} {'$/Trd':>8} {'MaxDD%':>8}")
print("  " + "-" * (W - 2))

agg_summary = []
con_summary = []

for t in TICKERS:
    ar = agg_results[t]
    cr = con_results[t]

    agg_avg = ar.total_pnl / ar.total_trades if ar.total_trades > 0 else 0
    con_avg = cr.total_pnl / cr.total_trades if cr.total_trades > 0 else 0

    agg_summary.append({
        'ticker': t, 'trades': ar.total_trades, 'wr': ar.win_rate,
        'pnl': ar.total_pnl, 'avg': agg_avg, 'dd': ar.max_drawdown_pct,
        'stacked': ar.stacked_trades, 'streak': ar.streak_trades,
    })
    con_summary.append({
        'ticker': t, 'trades': cr.total_trades, 'wr': cr.win_rate,
        'pnl': cr.total_pnl, 'avg': con_avg, 'dd': cr.max_drawdown_pct,
    })

    print(f"  {t:<6} | {ar.total_trades:>5} {ar.win_rate:>5.1%} {ar.total_pnl:>+10,.0f} {agg_avg:>+8.2f} {ar.max_drawdown_pct:>7.1%} {ar.stacked_trades:>4}"
          f" | {cr.total_trades:>5} {cr.win_rate:>5.1%} {cr.total_pnl:>+10,.0f} {con_avg:>+8.2f} {cr.max_drawdown_pct:>7.1%}")

print("  " + "-" * (W - 2))

agg_df = pd.DataFrame(agg_summary)
con_df = pd.DataFrame(con_summary)

total_agg_trades = agg_df['trades'].sum()
total_con_trades = con_df['trades'].sum()
total_agg_pnl = agg_df['pnl'].sum()
total_con_pnl = con_df['pnl'].sum()

print(f"  {'TOTAL':<6} | {total_agg_trades:>5} {agg_df['wr'].mean():>5.1%} {total_agg_pnl:>+10,.0f} {total_agg_pnl/total_agg_trades:>+8.2f} {agg_df['dd'].mean():>7.1%} {agg_df['stacked'].sum():>4}"
      f" | {total_con_trades:>5} {con_df['wr'].mean():>5.1%} {total_con_pnl:>+10,.0f} {total_con_pnl/total_con_trades:>+8.2f} {con_df['dd'].mean():>7.1%}")

# =====================================================
# AGGRESSIVE FEATURE BREAKDOWN
# =====================================================
print()
print("=" * W)
print(f"{'AGGRESSIVE FEATURE BREAKDOWN':^{W}}")
print("=" * W)

all_agg_trades = []
for t in TICKERS:
    for trade in agg_results[t].trade_log:
        trade_copy = trade.copy()
        trade_copy['ticker'] = t
        all_agg_trades.append(trade_copy)

tdf = pd.DataFrame(all_agg_trades)

if len(tdf) > 0:
    # Stacking analysis
    stacked = tdf[tdf['stacked']]
    normal = tdf[~tdf['stacked']]
    print(f"\n  POSITION STACKING (deep pullback 5%+):")
    print(f"    Normal trades:  {len(normal):>5}  WR: {normal['winner'].mean():>5.1%}  Avg P/L: ${normal['pnl'].mean():>+8.2f}  Total: ${normal['pnl'].sum():>+10,.0f}")
    print(f"    Stacked trades: {len(stacked):>5}  WR: {stacked['winner'].mean():>5.1%}  Avg P/L: ${stacked['pnl'].mean():>+8.2f}  Total: ${stacked['pnl'].sum():>+10,.0f}")

    # Streak analysis
    streak_on = tdf[tdf['streak_entry']]
    streak_off = tdf[~tdf['streak_entry']]
    print(f"\n  STREAK BONUS (6% buffer after 2+ consecutive wins):")
    print(f"    Normal buffer:  {len(streak_off):>5}  WR: {streak_off['winner'].mean():>5.1%}  Avg P/L: ${streak_off['pnl'].mean():>+8.2f}  Total: ${streak_off['pnl'].sum():>+10,.0f}")
    if len(streak_on) > 0:
        print(f"    Streak buffer:  {len(streak_on):>5}  WR: {streak_on['winner'].mean():>5.1%}  Avg P/L: ${streak_on['pnl'].mean():>+8.2f}  Total: ${streak_on['pnl'].sum():>+10,.0f}")
    else:
        print(f"    Streak buffer:      0  (no streak trades triggered)")

    # Pullback depth analysis
    print(f"\n  PULLBACK DEPTH ANALYSIS:")
    bins = [(-100, -7), (-7, -5), (-5, -3)]
    labels = ['Deep (>7%)', 'Medium (5-7%)', 'Mild (3-5%)']
    for (lo, hi), label in zip(bins, labels):
        mask = (tdf['pullback_pct'] >= lo) & (tdf['pullback_pct'] < hi)
        subset = tdf[mask]
        if len(subset) > 0:
            print(f"    {label:<16} {len(subset):>4} trades  WR: {subset['winner'].mean():>5.1%}  Avg P/L: ${subset['pnl'].mean():>+8.2f}")

# =====================================================
# YEARLY PERFORMANCE COMPARISON
# =====================================================
print()
print("=" * W)
print(f"{'YEARLY PERFORMANCE':^{W}}")
print("=" * W)
print(f"  {'Year':<6} {'Agg Trades':>10} {'Agg P/L':>10} {'Agg WR':>7} {'Con Trades':>11} {'Con P/L':>10} {'Con WR':>7} {'Agg Edge':>10}")
print("  " + "-" * 75)

if len(tdf) > 0:
    con_all_trades = []
    for t in TICKERS:
        for trade in con_results[t].trade_log:
            tc = trade.copy()
            tc['ticker'] = t
            con_all_trades.append(tc)
    cdf = pd.DataFrame(con_all_trades)

    for year in range(2016, 2027):
        ayr = tdf[tdf['date'].dt.year == year]
        cyr = cdf[cdf['date'].dt.year == year]
        if len(ayr) == 0 and len(cyr) == 0:
            continue
        a_pnl = ayr['pnl'].sum() if len(ayr) > 0 else 0
        c_pnl = cyr['pnl'].sum() if len(cyr) > 0 else 0
        a_wr = ayr['winner'].mean() if len(ayr) > 0 else 0
        c_wr = cyr['winner'].mean() if len(cyr) > 0 else 0

        print(f"  {year:<6} {len(ayr):>9} {a_pnl:>+10,.0f} {a_wr:>6.0%}"
              f"  {len(cyr):>10} {c_pnl:>+10,.0f} {c_wr:>6.0%}"
              f"  {a_pnl - c_pnl:>+10,.0f}")

    print("  " + "-" * 75)
    print(f"  {'TOTAL':<6} {len(tdf):>9} {tdf['pnl'].sum():>+10,.0f} {tdf['winner'].mean():>6.0%}"
          f"  {len(cdf):>10} {cdf['pnl'].sum():>+10,.0f} {cdf['winner'].mean():>6.0%}"
          f"  {tdf['pnl'].sum() - cdf['pnl'].sum():>+10,.0f}")

# =====================================================
# BEAR MARKET STRESS TEST
# =====================================================
print()
print("=" * W)
print(f"{'BEAR MARKET STRESS TEST':^{W}}")
print("=" * W)

bear_periods = {
    '2018 Q4 (Oil+Fed)':     lambda t: (t['date'].dt.year == 2018) & (t['date'].dt.month >= 10),
    '2020 COVID (Feb-Apr)':  lambda t: (t['date'].dt.year == 2020) & (t['date'].dt.month.isin([2, 3, 4])),
    '2022 H1 (Russia/Infl)': lambda t: (t['date'].dt.year == 2022) & (t['date'].dt.month <= 6),
    '2025 Q1 (Energy/Tar)':  lambda t: (t['date'].dt.year == 2025) & (t['date'].dt.month <= 3),
}

if len(tdf) > 0:
    for period_name, mask_fn in bear_periods.items():
        a_mask = mask_fn(tdf)
        c_mask = mask_fn(cdf)
        a_period = tdf[a_mask]
        c_period = cdf[c_mask]

        print(f"\n  {period_name}:")
        print(f"  {'Tkr':<6} {'Agg WR':>7} {'Agg P/L':>10} {'Con WR':>7} {'Con P/L':>10} {'Edge':>8}")
        print(f"  {'-' * 55}")

        for t in TICKERS:
            at = a_period[a_period['ticker'] == t]
            ct = c_period[c_period['ticker'] == t]
            if len(at) == 0 and len(ct) == 0:
                continue
            a_wr = at['winner'].mean() if len(at) > 0 else 0
            a_pnl = at['pnl'].sum() if len(at) > 0 else 0
            c_wr = ct['winner'].mean() if len(ct) > 0 else 0
            c_pnl = ct['pnl'].sum() if len(ct) > 0 else 0
            print(f"  {t:<6} {a_wr:>6.0%} {a_pnl:>+10,.0f} {c_wr:>6.0%} {c_pnl:>+10,.0f} {a_pnl - c_pnl:>+8,.0f}")

        a_total = a_period['pnl'].sum() if len(a_period) > 0 else 0
        c_total = c_period['pnl'].sum() if len(c_period) > 0 else 0
        print(f"  {'TOTAL':<6} {'':>7} {a_total:>+10,.0f} {'':>7} {c_total:>+10,.0f} {a_total - c_total:>+8,.0f}")

# =====================================================
# LOSS DISTRIBUTION
# =====================================================
print()
print("=" * W)
print(f"{'LOSS DISTRIBUTION: AGGRESSIVE vs CONSERVATIVE':^{W}}")
print("=" * W)

if len(tdf) > 0:
    a_losses = tdf[~tdf['winner']]['pnl']
    c_losses = cdf[~cdf['winner']]['pnl']

    print(f"\n  {'Metric':<25} {'Aggressive':>15} {'Conservative':>15}")
    print(f"  {'-' * 58}")
    print(f"  {'Total losses':<25} {len(a_losses):>15} {len(c_losses):>15}")
    if len(a_losses) > 0:
        print(f"  {'Avg loss':<25} ${a_losses.mean():>+14,.0f} ${c_losses.mean():>+14,.0f}")
        print(f"  {'Median loss':<25} ${a_losses.median():>+14,.0f} ${c_losses.median():>+14,.0f}")
        print(f"  {'Worst loss':<25} ${a_losses.min():>+14,.0f} ${c_losses.min():>+14,.0f}")
        print(f"  {'95th pct loss':<25} ${a_losses.quantile(0.05):>+14,.0f} ${c_losses.quantile(0.05):>+14,.0f}")

# =====================================================
# ACCOUNT GROWTH SIMULATION
# =====================================================
print()
print("=" * W)
print(f"{'ACCOUNT GROWTH SIMULATION':^{W}}")
print(f"{'$10K starting | 25% buying power per position | max 8 concurrent':^{W}}")
print("=" * W)

STARTING_CAPITAL = 10_000
BUYING_POWER_PCT = 0.25
MAX_POSITIONS = 8

if len(tdf) > 0:
    # Simulate account growth with compounding
    sorted_trades = tdf.sort_values('date').reset_index(drop=True)

    for label, trades_df, max_pos in [
        ('Aggressive', tdf, 8),
        ('Conservative', cdf, 6),
    ]:
        sorted_t = trades_df.sort_values('date').reset_index(drop=True)
        balance = STARTING_CAPITAL
        peak_bal = STARTING_CAPITAL
        max_dd_pct = 0.0
        yearly = {}

        for _, trade in sorted_t.iterrows():
            year = trade['date'].year
            if year not in yearly:
                yearly[year] = {'start': balance, 'pnl': 0}

            # Scale P/L by account size relative to baseline
            scale = balance / STARTING_CAPITAL
            scaled_pnl = trade['pnl'] * min(scale, 5.0)  # cap at 5x
            balance += scaled_pnl
            balance = max(balance, 100)  # floor
            yearly[year]['pnl'] += scaled_pnl

            peak_bal = max(peak_bal, balance)
            dd = (balance - peak_bal) / peak_bal
            max_dd_pct = min(max_dd_pct, dd)

        print(f"\n  {label} ($10K start):")
        print(f"    Final balance:  ${balance:>12,.0f}")
        print(f"    Total return:   {(balance - STARTING_CAPITAL) / STARTING_CAPITAL:>11.0%}")
        cagr_years = 10.25
        cagr = ((balance / STARTING_CAPITAL) ** (1 / cagr_years) - 1) if balance > 0 else -1
        print(f"    CAGR:           {cagr:>11.1%}")
        print(f"    Max drawdown:   {max_dd_pct:>11.1%}")

        print(f"\n    {'Year':<6} {'Start':>10} {'P/L':>10} {'End':>10} {'Return':>8}")
        print(f"    {'-' * 50}")
        for year in sorted(yearly.keys()):
            y = yearly[year]
            end = y['start'] + y['pnl']
            ret = y['pnl'] / y['start'] if y['start'] > 0 else 0
            print(f"    {year:<6} ${y['start']:>9,.0f} ${y['pnl']:>+9,.0f} ${end:>9,.0f} {ret:>+7.1%}")

# =====================================================
# SENSITIVITY ANALYSIS
# =====================================================
print()
print("=" * W)
print(f"{'SENSITIVITY: BUFFER & SPREAD WIDTH':^{W}}")
print("=" * W)

configs = [
    ('5% buf / 3% width', 0.05, 0.03),
    ('7% buf / 3% width', 0.07, 0.03),
    ('7% buf / 4% width', 0.07, 0.04),
    ('10% buf / 3% width', 0.10, 0.03),
    ('5% buf / 5% width', 0.05, 0.05),
]

print(f"\n  {'Config':<22} {'Trades':>7} {'WR':>6} {'Total P/L':>12} {'$/Trade':>9} {'MaxDD%':>8}")
print(f"  {'-' * 70}")

for label, buf, sw in configs:
    strat = AggressivePullbackStrategy(
        buffer=buf,
        spread_pct=sw,
        pullback_threshold=0.03,
        deep_pullback=0.05,
        streak_bonus_threshold=2,
        streak_buffer=buf - 0.01,
    )
    total_trades = 0
    total_pnl = 0.0
    total_winners = 0
    dd_list = []

    for t in TICKERS:
        r = strat.run(DATA[t])
        total_trades += r.total_trades
        total_pnl += r.total_pnl
        total_winners += r.winners
        dd_list.append(r.max_drawdown_pct)

    wr = total_winners / total_trades if total_trades > 0 else 0
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    avg_dd = np.mean(dd_list)

    print(f"  {label:<22} {total_trades:>6} {wr:>5.1%} {total_pnl:>+12,.0f} {avg_pnl:>+9.2f} {avg_dd:>7.1%}")

# =====================================================
# RISK WARNINGS
# =====================================================
print()
print("=" * W)
print(f"{'RISK ASSESSMENT FOR INDIVIDUAL ACCOUNT':^{W}}")
print("=" * W)

if len(tdf) > 0:
    # Max consecutive losses
    for t in TICKERS:
        trades = [tr for tr in all_agg_trades if tr['ticker'] == t]
        if not trades:
            continue
        max_consec = 0
        current = 0
        for tr in trades:
            if not tr['winner']:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 0

    print(f"""
  RISK PROFILE:
    Buffer:         7% (vs 10% conservative) -- 30% less room for drops
    Spread width:   3% (vs 2%) -- 50% higher max loss per contract
    Stacking:       2x position on 5%+ pullbacks -- doubling down on weakness
    Streak bonus:   6% buffer after 2 wins -- pressing luck

  WHAT COULD GO WRONG:
    1. COVID-type crash: 30%+ drop in 3 weeks blows through 7% buffer easily
       Conservative loses ~$500/contract, aggressive loses ~$900/contract
       Stacking doubles that to ~$1,800 in a deep pullback scenario

    2. Streak bonus whipsaw: 2 wins -> tighten to 6% -> sudden reversal
       The tighter buffer means the next loss is larger

    3. Correlation risk: all 8 tickers drop simultaneously (systemic event)
       With 8 max positions, worst case is 8 x $900 = $7,200 on $10K account

  WHO THIS IS FOR:
    - Account size: $10K-$50K (enough to absorb $5K+ drawdowns)
    - Risk tolerance: Can stomach -30%+ drawdowns without panic closing
    - Time horizon: 3+ years (needs time to compound past drawdowns)
    - Experience: Understands options mechanics and has live-traded spreads

  WHO THIS IS NOT FOR:
    - Accounts under $5K (one bad month could wipe 50%+)
    - Anyone who will close positions early during drawdowns
    - Anyone expecting smooth equity curves
""")

print("=" * W)
print(f"{'END OF REPORT':^{W}}")
print("=" * W)
