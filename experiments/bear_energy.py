"""Bear market analysis: energy-cost-driven selloffs and credit spread performance."""

import warnings
warnings.filterwarnings("ignore")

from tradelab.pipeline import DataPipeline
from tradelab.options import historical_volatility, put_credit_spread_price
import pandas as pd
import numpy as np

pipe = DataPipeline()

def load(ticker):
    return pipe.fetch_stock(ticker, start='2007-01-01', end='2026-04-04')

spy = load('SPY')
qqq = load('QQQ')
xle = load('XLE')
nvda = load('NVDA')
aapl = load('AAPL')

def ts(date_str):
    return int(pd.Timestamp(date_str).timestamp())

def price_at(df, date_str):
    t = ts(date_str)
    idx = np.searchsorted(df.index.values, t)
    idx = min(idx, len(df) - 1)
    return df['close'].iloc[idx]

def pct_change_between(df, d1, d2):
    p1, p2 = price_at(df, d1), price_at(df, d2)
    return (p2 - p1) / p1 * 100

# =====================================================
# 1. IDENTIFY ANALOGOUS PERIODS
# =====================================================
# Energy-cost-driven selloffs where XLE surged while SPY dropped
periods = {
    '2008 Oil Spike': {
        'desc': 'Oil hit $147/bbl, preceded financial crisis',
        'energy_run': ('2007-12-01', '2008-07-01'),  # XLE rallied
        'market_pain': ('2008-05-01', '2008-11-20'),  # SPY crashed
        'worst_month': ('2008-09-01', '2008-10-10'),
    },
    '2018 Q4 Selloff': {
        'desc': 'Oil spike + Fed tightening + trade war',
        'energy_run': ('2018-06-01', '2018-10-01'),
        'market_pain': ('2018-09-20', '2018-12-24'),
        'worst_month': ('2018-11-08', '2018-12-24'),
    },
    '2022 Energy Crisis': {
        'desc': 'Russia/Ukraine, oil $130+, inflation spiral',
        'energy_run': ('2022-01-01', '2022-06-08'),  # XLE +60%
        'market_pain': ('2022-01-03', '2022-06-17'),  # SPY -24%
        'worst_month': ('2022-04-01', '2022-05-20'),
    },
    '2025-26 Current': {
        'desc': 'Rising energy costs, tariff escalation',
        'energy_run': ('2025-12-01', '2026-04-04'),
        'market_pain': ('2026-02-01', '2026-04-04'),
        'worst_month': ('2026-03-01', '2026-04-04'),
    },
}

W = 100
print("=" * W)
print(f"{'ENERGY-COST-DRIVEN SELLOFFS: HISTORICAL COMPARISON':^{W}}")
print("=" * W)
print()

# Header
print(f"{'Period':<22} {'SPY':>8} {'QQQ':>8} {'XLE':>8} {'NVDA':>8} {'AAPL':>8}  Description")
print("-" * W)

for name, p in periods.items():
    d1, d2 = p['market_pain']
    try:
        vals = [pct_change_between(d, d1, d2) for d in [spy, qqq, xle, nvda, aapl]]
        print(f"{name:<22}", end="")
        for v in vals:
            print(f" {v:>+7.1f}%", end="")
        print(f"  {p['desc']}")
    except:
        print(f"{name:<22} (data not available)")

# Worst month detail
print()
print(f"{'Worst Month Drawdown':<22} {'SPY':>8} {'QQQ':>8} {'XLE':>8} {'NVDA':>8} {'AAPL':>8}")
print("-" * W)
for name, p in periods.items():
    d1, d2 = p['worst_month']
    try:
        vals = [pct_change_between(d, d1, d2) for d in [spy, qqq, xle, nvda, aapl]]
        print(f"{name:<22}", end="")
        for v in vals:
            print(f" {v:>+7.1f}%", end="")
        print()
    except:
        print(f"{name:<22} (data not available)")

# XLE during energy run
print()
print(f"{'Energy Sector (XLE) During Run-Up':}")
print("-" * 60)
for name, p in periods.items():
    d1, d2 = p['energy_run']
    try:
        xle_chg = pct_change_between(xle, d1, d2)
        spy_chg = pct_change_between(spy, d1, d2)
        print(f"  {name:<22} XLE: {xle_chg:>+7.1f}%   SPY: {spy_chg:>+7.1f}%   Divergence: {xle_chg - spy_chg:>+7.1f}%")
    except:
        pass

# =====================================================
# 2. CURRENT MARKET CONTEXT (last 60 days)
# =====================================================
print()
print("=" * W)
print(f"{'CURRENT MARKET CONTEXT (last 60 trading days)':^{W}}")
print("=" * W)

for name, df in [('SPY', spy), ('QQQ', qqq), ('XLE', xle), ('NVDA', nvda), ('AAPL', aapl)]:
    recent = df.iloc[-60:]
    vol = historical_volatility(df['close'], window=30)
    current_vol = vol.iloc[-1]
    high = recent['close'].max()
    low = recent['close'].min()
    last = recent['close'].iloc[-1]
    dd = (last - high) / high * 100
    chg_60d = (last - recent['close'].iloc[0]) / recent['close'].iloc[0] * 100
    print(f"  {name:<5} ${last:>8.2f}  60d: {chg_60d:>+6.1f}%  HV30: {current_vol:>5.1%}  Drawdown from 60d high: {dd:>+6.1f}%")

# =====================================================
# 3. CREDIT SPREAD PERFORMANCE DURING EACH PERIOD
# =====================================================
print()
print("=" * W)
print(f"{'PUT CREDIT SPREAD PERFORMANCE DURING SELLOFF PERIODS':^{W}}")
print("=" * W)

def run_period(df, start, end, buffer, dte, spread_pct):
    t_start = ts(start)
    t_end = ts(end)
    period_df = df[(df.index >= t_start) & (df.index <= t_end)]
    if len(period_df) < 30:
        return None

    close_prices = period_df['close'].values
    timestamps = period_df.index.values

    # Use vol from full history for context
    full_vol = historical_volatility(df['close'], window=30)
    offset = max(1, int(dte * 21 / 30))

    results = []
    i = 0
    while i < len(period_df) - offset:
        price = close_prices[i]
        # Find vol for this timestamp
        vol_idx = df.index.get_indexer([timestamps[i]], method='nearest')[0]
        vol_val = full_vol.iloc[vol_idx]
        if np.isnan(vol_val) or vol_val <= 0:
            i += 1
            continue

        spread_width = price * spread_pct
        short_strike = price * (1 - buffer)
        long_strike = short_strike - spread_width

        expiry_idx = i + offset
        if expiry_idx >= len(close_prices):
            break
        expiry_price = close_prices[expiry_idx]

        T = dte / 365.0
        sp = put_credit_spread_price(price, short_strike, long_strike, T, 0.05, vol_val)
        net_credit = sp['net_credit_dollar']
        max_loss = sp['max_loss']
        if max_loss <= 0 or net_credit <= 0:
            i += offset
            continue

        winner = expiry_price > short_strike
        pnl = net_credit if winner else -max_loss

        results.append({
            'date': pd.Timestamp(timestamps[i], unit='s'),
            'price': price,
            'expiry_price': expiry_price,
            'drop_pct': (expiry_price - price) / price * 100,
            'pnl': pnl,
            'credit': net_credit,
            'max_loss': max_loss,
            'winner': winner,
            'sigma': vol_val,
        })
        i += offset

    if not results:
        return None
    return pd.DataFrame(results)


# Test each period with different configs
configs = [
    ('Aggressive',    0.05, 14, 0.03),
    ('Moderate',      0.07, 14, 0.03),
    ('Conservative',  0.10, 14, 0.02),
    ('Your Best (NV)', 0.10, 14, 0.05),
]

for ticker_name, ticker_df in [('NVDA', nvda), ('AAPL', aapl), ('SPY', spy)]:
    print()
    print(f"\n{'-' * W}")
    print(f" {ticker_name}")
    print(f"{'-' * W}")

    for period_name, p in periods.items():
        d1, d2 = p['market_pain']
        mkt_chg = pct_change_between(ticker_df, d1, d2)
        print(f"\n  {period_name} ({d1} to {d2})  |  {ticker_name} moved {mkt_chg:+.1f}%")
        print(f"  {'Config':<18} {'Trades':>6} {'Win%':>6} {'W/L':>6} {'AvgCr':>7} {'AvgLoss':>8} {'P/L':>9} {'$/Trd':>8}")
        print(f"  {'-'*75}")

        for config_name, buf, dte, sprd in configs:
            trades = run_period(ticker_df, d1, d2, buf, dte, sprd)
            if trades is None or len(trades) == 0:
                print(f"  {config_name:<18} {'--':>6}")
                continue
            w = trades['winner'].sum()
            l = len(trades) - w
            wr = w / len(trades)
            ac = trades[trades['winner']]['credit'].mean() if w > 0 else 0
            al = abs(trades[~trades['winner']]['pnl'].mean()) if l > 0 else 0
            tp = trades['pnl'].sum()
            ap = trades['pnl'].mean()
            print(f"  {config_name:<18} {len(trades):>6} {wr:>5.0%} {w:>2}/{l:<2}  ${ac:>6.1f} ${al:>7.1f} ${tp:>+8.0f} ${ap:>+7.2f}")

        # Show individual trades for conservative config during worst periods
        trades = run_period(ticker_df, d1, d2, 0.10, 14, 0.02)
        if trades is not None and len(trades) > 0:
            losers = trades[~trades['winner']]
            if len(losers) > 0:
                print(f"\n  Conservative losses:")
                for _, r in losers.iterrows():
                    print(f"    {r['date']:%Y-%m-%d}  ${r['price']:>7.2f} -> ${r['expiry_price']:>7.2f}  ({r['drop_pct']:>+5.1f}%)  HV={r['sigma']:.0%}  -${abs(r['pnl']):.0f}")

# =====================================================
# 4. PATTERN ANALYSIS: what do energy selloffs have in common?
# =====================================================
print()
print("=" * W)
print(f"{'PATTERN ANALYSIS: COMMON TRAITS OF ENERGY-DRIVEN SELLOFFS':^{W}}")
print("=" * W)

print("""
Key patterns across 2008, 2018, 2022, and current:

1. ENERGY LEADS, EQUITIES LAG
   - XLE/energy sector rallies 1-3 months BEFORE equity pain peaks
   - Rising energy costs compress margins -> earnings misses -> selloff

2. VOLATILITY SPIKE TIMING
   - HV30 typically doubles from pre-selloff levels
   - The spike happens DURING the selloff, not before
   - By the time vol filter would trigger, damage is done

3. MAGNITUDE AND DURATION""")

for name, p in periods.items():
    d1, d2 = p['market_pain']
    spy_chg = pct_change_between(spy, d1, d2)
    days = (pd.Timestamp(d2) - pd.Timestamp(d1)).days
    print(f"   {name:<22} SPY {spy_chg:>+7.1f}% over {days:>3} days")

print("""
4. TECH IS HIT HARDEST
   - NVDA/QQQ consistently drop more than SPY
   - High-beta names get punished disproportionately

5. RECOVERY PATTERN
   - Energy selloffs tend to have 2-3 waves (not a single crash)
   - The "dead cat bounce" between waves is where credit spreads
     get killed -- you open after the first drop thinking the worst
     is over, then wave 2 hits.
""")

# =====================================================
# 5. WHAT TO EXPECT MID-APRIL 2026
# =====================================================
print("=" * W)
print(f"{'CURRENT POSITIONING: MID-APRIL 2026 OUTLOOK':^{W}}")
print("=" * W)

# Current readings
spy_vol = historical_volatility(spy['close'], window=30).iloc[-1]
nvda_vol = historical_volatility(nvda['close'], window=30).iloc[-1]
aapl_vol = historical_volatility(aapl['close'], window=30).iloc[-1]
xle_30d = pct_change_between(xle, '2026-03-01', '2026-04-02')
spy_30d = pct_change_between(spy, '2026-03-01', '2026-04-02')

print(f"""
  Current volatility:   SPY {spy_vol:.1%}   NVDA {nvda_vol:.1%}   AAPL {aapl_vol:.1%}
  XLE vs SPY (30d):     XLE {xle_30d:+.1f}%   SPY {spy_30d:+.1f}%   Divergence {xle_30d-spy_30d:+.1f}%
""")

# Compare current vol to historical selloff entry points
print("  Volatility comparison to selloff entry points:")
for name, p in periods.items():
    d1, _ = p['market_pain']
    try:
        vol_idx = spy.index.get_indexer([ts(d1)], method='nearest')[0]
        entry_vol = historical_volatility(spy['close'], window=30).iloc[vol_idx]
        print(f"    {name:<22} Entry HV30: {entry_vol:.1%}   (current: {spy_vol:.1%})")
    except:
        pass

print(f"""
  IMPLICATIONS FOR CREDIT SPREADS:
  ---------------------------------
  If energy costs accelerate in mid-April, historical analogs suggest:

  - SPY could see another 5-15% downside over 1-3 months
  - NVDA/QQQ likely hit harder (1.5-2x SPY drawdown)
  - Volatility will spike further, inflating premiums BUT also losses

  DEFENSIVE POSITIONING:
  1. WIDEN BUFFERS to 10%+ (the conservative config survived best)
  2. SHORTEN DTE to 7-14 days (less time for wave 2 to hit)
  3. NARROW SPREADS to 2% of underlying (cap max loss)
  4. FAVOR SPY over single stocks (diversification dampens tail risk)
  5. CONSIDER PAUSING when HV30 > 35-40% (vol filter)
  6. SIZE DOWN -- reduce contracts, not strategy
""")
