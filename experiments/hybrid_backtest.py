"""Hybrid strategy backtest: sweep close thresholds and emergency stops."""

import warnings
warnings.filterwarnings("ignore")

from tradelab.pipeline import DataPipeline
from tradelab.strategies.hybrid_spreads import HybridSpreadStrategy
import pandas as pd
import numpy as np

pipe = DataPipeline()

TICKERS = ['QQQ', 'AVGO', 'CAT', 'MSFT', 'NVDA', 'AAPL', 'SPY', 'META', 'AMD', 'GOOG']

data = {}
for t in TICKERS:
    data[t] = pipe.fetch_stock(t, start='2016-01-01', end='2026-04-04')

W = 115

# =====================================================
# 1. COMPARE: Baseline B, Baseline C, Hybrid variants
# =====================================================

configs = {
    'B: 30d hold':           {'close_threshold': 9.99, 'emergency_stop': 9.99},  # never triggers
    'C: 30d->14d always':    {'close_threshold': 0.00, 'emergency_stop': 9.99},  # always close at checkpoint
    'H1: 3% / 7% stop':     {'close_threshold': 0.03, 'emergency_stop': 0.07},
    'H2: 3% / 10% stop':    {'close_threshold': 0.03, 'emergency_stop': 0.10},
    'H3: 5% / 7% stop':     {'close_threshold': 0.05, 'emergency_stop': 0.07},
    'H4: 5% / 10% stop':    {'close_threshold': 0.05, 'emergency_stop': 0.10},
    'H5: 3% / no stop':     {'close_threshold': 0.03, 'emergency_stop': 9.99},
    'H6: 5% / no stop':     {'close_threshold': 0.05, 'emergency_stop': 9.99},
    'H7: 7% / 10% stop':    {'close_threshold': 0.07, 'emergency_stop': 0.10},
}

all_results = []

print("Running hybrid configs...")
for config_name, params in configs.items():
    for ticker in TICKERS:
        strat = HybridSpreadStrategy(
            buffer=0.10,
            spread_pct=0.02,
            dte_open=30,
            dte_checkpoint=14,
            close_threshold=params['close_threshold'],
            emergency_stop=params['emergency_stop'],
        )
        result = strat.run(data[ticker])
        if result.total_trades == 0:
            continue

        trades = pd.DataFrame(result.trade_log)
        avg_loss = abs(trades[~trades['winner']]['pnl'].mean()) if result.losers > 0 else 0

        all_results.append({
            'config': config_name,
            'ticker': ticker,
            'trades': result.total_trades,
            'win_rate': result.win_rate,
            'total_pnl': result.total_pnl,
            'avg_pnl': result.total_pnl / result.total_trades,
            'max_dd': result.max_drawdown,
            'avg_loss': avg_loss,
            'held': result.held_to_expiry,
            'checkpoint': result.closed_at_checkpoint,
            'emergency': result.emergency_closed,
            'winners': result.winners,
            'losers': result.losers,
        })

df_results = pd.DataFrame(all_results)

# =====================================================
# 2. AGGREGATE BY CONFIG
# =====================================================
print()
print("=" * W)
print(f"{'HYBRID STRATEGY: CONFIG COMPARISON (aggregated across 10 tickers)':^{W}}")
print("=" * W)
print(f"{'Config':<22} {'Trades':>6} {'WR':>6} {'Tot P/L':>10} {'$/Trade':>8} {'AvgDD':>8} {'AvgLoss':>8} {'Hold':>5} {'Chkpt':>5} {'Emrg':>5} {'Prof':>5}")
print("-" * W)

for config_name in configs:
    sub = df_results[df_results['config'] == config_name]
    prof = len(sub[sub['total_pnl'] > 0])
    print(f"{config_name:<22} "
          f"{sub['trades'].sum():>6} "
          f"{sub['winners'].sum() / sub['trades'].sum():>5.1%} "
          f"${sub['total_pnl'].sum():>+9.0f} "
          f"${sub['avg_pnl'].mean():>+7.2f} "
          f"${sub['max_dd'].mean():>+7.0f} "
          f"${sub['avg_loss'].mean():>7.0f} "
          f"{sub['held'].sum():>5} "
          f"{sub['checkpoint'].sum():>5} "
          f"{sub['emergency'].sum():>5} "
          f"{prof:>3}/10")

# =====================================================
# 3. BEST CONFIG PER TICKER
# =====================================================
print()
print("=" * W)
print(f"{'BEST CONFIG PER TICKER':^{W}}")
print("=" * W)
print(f"{'Ticker':<6} {'Best Config':<22} {'P/L':>10} {'WR':>6} {'MaxDD':>8} {'2nd Best':<22} {'P/L':>10}")
print("-" * W)

for ticker in TICKERS:
    sub = df_results[df_results['ticker'] == ticker].sort_values('total_pnl', ascending=False)
    best = sub.iloc[0]
    second = sub.iloc[1] if len(sub) > 1 else best
    print(f"{ticker:<6} {best['config']:<22} ${best['total_pnl']:>+9.0f} {best['win_rate']:>5.1%} ${best['max_dd']:>+7.0f} "
          f"{second['config']:<22} ${second['total_pnl']:>+9.0f}")

# =====================================================
# 4. DEEP DIVE: BEST HYBRID vs B vs C
# =====================================================
# Find overall best hybrid config
hybrid_configs = [c for c in configs if c.startswith('H')]
hybrid_agg = df_results[df_results['config'].isin(hybrid_configs)].groupby('config').agg({
    'total_pnl': 'sum',
    'avg_pnl': 'mean',
    'max_dd': 'mean',
}).sort_values('total_pnl', ascending=False)

best_hybrid = hybrid_agg.index[0]
print()
print("=" * W)
print(f"  BEST HYBRID CONFIG: {best_hybrid}")
print(f"  Combined P/L: ${hybrid_agg.loc[best_hybrid, 'total_pnl']:+,.0f}  "
      f"Avg $/trade: ${hybrid_agg.loc[best_hybrid, 'avg_pnl']:+.2f}  "
      f"Avg MaxDD: ${hybrid_agg.loc[best_hybrid, 'max_dd']:+,.0f}")
print("=" * W)

# Per-ticker comparison: B vs best hybrid
print()
print(f"{'Tkr':<6} {'--- B: 30d hold ---':>22} {'--- ' + best_hybrid + ' ---':>28} {'Improvement':>12}")
print(f"{'':.<6} {'WR':>6} {'P/L':>9} {'MaxDD':>8} {'WR':>7} {'P/L':>9} {'MaxDD':>8} {'P/L diff':>12}")
print("-" * W)

for ticker in TICKERS:
    b = df_results[(df_results['config'] == 'B: 30d hold') & (df_results['ticker'] == ticker)]
    h = df_results[(df_results['config'] == best_hybrid) & (df_results['ticker'] == ticker)]
    if b.empty or h.empty:
        continue
    b = b.iloc[0]
    h = h.iloc[0]
    diff = h['total_pnl'] - b['total_pnl']
    print(f"{ticker:<6} {b['win_rate']:>5.1%} ${b['total_pnl']:>+8.0f} ${b['max_dd']:>+7.0f}"
          f"  {h['win_rate']:>6.1%} ${h['total_pnl']:>+8.0f} ${h['max_dd']:>+7.0f}"
          f"  ${diff:>+11.0f}")

# Totals
b_total = df_results[df_results['config'] == 'B: 30d hold']['total_pnl'].sum()
h_total = df_results[df_results['config'] == best_hybrid]['total_pnl'].sum()
b_dd = df_results[df_results['config'] == 'B: 30d hold']['max_dd'].mean()
h_dd = df_results[df_results['config'] == best_hybrid]['max_dd'].mean()
print("-" * W)
print(f"{'TOTAL':<6} {'':>5} ${b_total:>+8.0f} ${b_dd:>+7.0f}"
      f"  {'':>6} ${h_total:>+8.0f} ${h_dd:>+7.0f}"
      f"  ${h_total - b_total:>+11.0f}")

# =====================================================
# 5. EXIT TYPE ANALYSIS FOR BEST HYBRID
# =====================================================
print()
print("=" * W)
print(f"{'EXIT TYPE ANALYSIS: ' + best_hybrid:^{W}}")
print("=" * W)

for ticker in TICKERS:
    h_data = df_results[(df_results['config'] == best_hybrid) & (df_results['ticker'] == ticker)]
    if h_data.empty:
        continue
    h = h_data.iloc[0]
    total = h['trades']
    print(f"  {ticker:<5}  Held: {h['held']:>3} ({h['held']/total:>4.0%})  "
          f"Checkpoint: {h['checkpoint']:>3} ({h['checkpoint']/total:>4.0%})  "
          f"Emergency: {h['emergency']:>3} ({h['emergency']/total:>4.0%})")

# =====================================================
# 6. BEAR MARKET DRILL: BEST HYBRID
# =====================================================
print()
print("=" * W)
print(f"{'BEAR MARKET DRILL: ' + best_hybrid:^{W}}")
print("=" * W)

best_params = configs[best_hybrid]

bear_windows = {
    '2018 Q4':    ('2018-09-15', '2018-12-31'),
    '2020 COVID': ('2020-02-01', '2020-05-01'),
    '2022 H1':    ('2022-01-01', '2022-07-01'),
    '2025 Q1':    ('2025-01-01', '2025-04-01'),
}

for period_name, (start, end) in bear_windows.items():
    print(f"\n  {period_name} ({start} to {end}):")
    print(f"  {'Tkr':<6} {'Trades':>6} {'WR':>6} {'P/L':>9} {'Hold':>5} {'Chkpt':>5} {'Emrg':>5} {'AvgLoss':>8}")
    print(f"  {'-'*60}")

    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())

    period_pnls = []
    for ticker in TICKERS[:6]:
        # Run the hybrid on just this window
        period_df = data[ticker][(data[ticker].index >= start_ts) & (data[ticker].index <= end_ts)]
        if len(period_df) < 30:
            continue

        strat = HybridSpreadStrategy(
            buffer=0.10, spread_pct=0.02, dte_open=30, dte_checkpoint=14,
            close_threshold=best_params['close_threshold'],
            emergency_stop=best_params['emergency_stop'],
        )
        result = strat.run(period_df)
        if result.total_trades == 0:
            continue

        trades = pd.DataFrame(result.trade_log)
        avg_loss = abs(trades[~trades['winner']]['pnl'].mean()) if result.losers > 0 else 0

        print(f"  {ticker:<6} {result.total_trades:>6} {result.win_rate:>5.0%} "
              f"${result.total_pnl:>+8.0f} {result.held_to_expiry:>5} "
              f"{result.closed_at_checkpoint:>5} {result.emergency_closed:>5} "
              f"${avg_loss:>7.0f}")
        period_pnls.append(result.total_pnl)

    if period_pnls:
        survived = sum(1 for p in period_pnls if p >= 0)
        print(f"  {'':.<6} {'':>6} {'':>6} ${sum(period_pnls):>+8.0f}  ({survived}/{len(period_pnls)} survived)")

# =====================================================
# 7. SUMMARY
# =====================================================
print()
print("=" * W)
print(f"{'FINAL SUMMARY':^{W}}")
print("=" * W)

# Rank all configs
config_rank = df_results.groupby('config').agg({
    'total_pnl': 'sum',
    'avg_pnl': 'mean',
    'max_dd': 'mean',
}).sort_values('total_pnl', ascending=False)

prof_by_config = df_results.groupby('config').apply(lambda x: (x['total_pnl'] > 0).sum()).to_dict()

print(f"\n  {'Rank':<5} {'Config':<22} {'Combined P/L':>13} {'Avg $/Trade':>12} {'Avg MaxDD':>10} {'Profitable':>11}")
print("  " + "-" * 80)
for rank, (config_name, row) in enumerate(config_rank.iterrows(), 1):
    prof = prof_by_config.get(config_name, 0)
    print(f"  {rank:<5} {config_name:<22} ${row['total_pnl']:>+12.0f} ${row['avg_pnl']:>+11.2f} ${row['max_dd']:>+9.0f} {prof:>7}/10")
