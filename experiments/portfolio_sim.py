"""Multi-security portfolio simulation with selection strategies.

Models a single brokerage account trading put credit spreads across
multiple tickers simultaneously, with:
- Shared buying power pool
- Per-trade friction (commissions + slippage)
- Multiple selection/allocation strategies
- Proper position tracking and risk limits
"""

import warnings
warnings.filterwarnings("ignore")

from tradelab.pipeline import DataPipeline
from tradelab.options import (
    bs_put_price,
    put_credit_spread_price,
    historical_volatility,
)
import pandas as pd
import numpy as np

pipe = DataPipeline()

# =====================================================
# UNIVERSE
# =====================================================
UNIVERSE = {
    'QQQ':  'ETF',
    'AVGO': 'Semi',
    'CAT':  'Industrial',
    'MSFT': 'Tech',
    'NVDA': 'Semi',
    'META': 'Tech',
    'AMD':  'Semi',
    'GOOG': 'Tech',
    'AAPL': 'Tech',
    'SPY':  'ETF',
    'IWM':  'ETF',
    'JPM':  'Financial',
    'WMT':  'Consumer',
    'HD':   'Consumer',
    'XLE':  'Energy',
}

# Strategy params
BUFFER = 0.10
SPREAD_PCT = 0.02
DTE_OPEN = 30
DTE_CLOSE = 14
R = 0.05

# Friction
COMMISSION_PER_CONTRACT = 0.65  # typical options commission per leg
SLIPPAGE_PCT = 0.02             # 2% of credit lost to bid-ask slippage

# Load all data
print("Loading data...")
ALL_DATA = {}
ALL_VOL = {}
for ticker in UNIVERSE:
    df = pipe.fetch_stock(ticker, start='2016-01-01', end='2026-04-04')
    ALL_DATA[ticker] = df
    ALL_VOL[ticker] = historical_volatility(df['close'], window=30)
print(f"Loaded {len(ALL_DATA)} tickers\n")

# Build a unified date index (trading days present in ALL tickers)
common_idx = ALL_DATA['SPY'].index  # SPY has all trading days


def get_price(ticker, idx):
    df = ALL_DATA[ticker]
    pos = np.searchsorted(df.index.values, common_idx[idx])
    pos = min(pos, len(df) - 1)
    return df['close'].iloc[pos]


def get_vol(ticker, idx):
    vol = ALL_VOL[ticker]
    df = ALL_DATA[ticker]
    pos = np.searchsorted(df.index.values, common_idx[idx])
    pos = min(pos, len(vol) - 1)
    v = vol.iloc[pos]
    return v if not np.isnan(v) and v > 0 else None


def get_sma(ticker, idx, window=50):
    df = ALL_DATA[ticker]
    pos = np.searchsorted(df.index.values, common_idx[idx])
    if pos < window:
        return None
    return df['close'].iloc[pos - window:pos].mean()


# =====================================================
# SELECTION STRATEGIES
# =====================================================

def select_equal_weight(candidates, account_state, idx):
    """Trade all candidates equally."""
    return {t: 1.0 for t in candidates}


def select_top_premium(candidates, account_state, idx, top_n=5):
    """Pick the N tickers offering the richest credit potential."""
    scored = []
    for t in candidates:
        price = get_price(t, idx)
        vol = get_vol(t, idx)
        if vol is None or price <= 0:
            continue
        sw = price * SPREAD_PCT
        sk = price * (1 - BUFFER)
        lk = sk - sw
        if lk <= 0:
            continue
        sp = put_credit_spread_price(price, sk, lk, DTE_OPEN / 365, R, vol)
        if sp['max_loss'] > 0 and sp['net_credit_dollar'] > 0:
            scored.append((t, sp['credit_potential']))
    scored.sort(key=lambda x: x[1], reverse=True)
    return {t: 1.0 for t, _ in scored[:top_n]}


def select_momentum_filter(candidates, account_state, idx):
    """Only trade tickers above their 50-day SMA (uptrend)."""
    selected = {}
    for t in candidates:
        price = get_price(t, idx)
        sma = get_sma(t, idx, 50)
        if sma is not None and price > sma:
            selected[t] = 1.0
    return selected


def select_vol_weighted(candidates, account_state, idx):
    """Weight allocation by volatility (higher vol = more allocation)."""
    vols = {}
    for t in candidates:
        v = get_vol(t, idx)
        if v is not None and v > 0:
            vols[t] = v
    if not vols:
        return {}
    total = sum(vols.values())
    return {t: v / total * len(vols) for t, v in vols.items()}


def select_diversified(candidates, account_state, idx, max_per_sector=2):
    """Limit exposure per sector, pick best premium within each."""
    scored = []
    for t in candidates:
        price = get_price(t, idx)
        vol = get_vol(t, idx)
        if vol is None or price <= 0:
            continue
        sw = price * SPREAD_PCT
        sk = price * (1 - BUFFER)
        lk = sk - sw
        if lk <= 0:
            continue
        sp = put_credit_spread_price(price, sk, lk, DTE_OPEN / 365, R, vol)
        if sp['max_loss'] > 0 and sp['net_credit_dollar'] > 0:
            scored.append((t, UNIVERSE[t], sp['credit_potential']))

    scored.sort(key=lambda x: x[2], reverse=True)
    selected = {}
    sector_count = {}
    for t, sector, _ in scored:
        if sector_count.get(sector, 0) >= max_per_sector:
            continue
        selected[t] = 1.0
        sector_count[sector] = sector_count.get(sector, 0) + 1
    return selected


def select_momentum_top_premium(candidates, account_state, idx, top_n=5):
    """Momentum filter + top premium: only uptrending, pick richest."""
    uptrend = []
    for t in candidates:
        price = get_price(t, idx)
        sma = get_sma(t, idx, 50)
        if sma is not None and price > sma:
            uptrend.append(t)
    return select_top_premium(uptrend, account_state, idx, top_n)


# =====================================================
# PORTFOLIO SIMULATOR
# =====================================================

def run_portfolio(
    selector_fn,
    starting_capital=25000,
    max_contracts_per_ticker=10,
    max_total_positions=6,
    max_pct_per_position=0.15,
    rebalance_interval=21,  # trading days between selection rounds
):
    offset_close = max(1, int((DTE_OPEN - DTE_CLOSE) * 21 / 30))

    balance = float(starting_capital)
    locked = 0.0
    positions = []  # list of open position dicts
    trade_log = []
    equity_curve = []

    n = len(common_idx)
    i = 50  # skip SMA warmup

    last_selection_day = 0

    while i < n:
        ts = common_idx[i]

        # --- CLOSE positions at checkpoint ---
        still_open = []
        for pos in positions:
            if i >= pos['close_idx']:
                exit_price = get_price(pos['ticker'], min(pos['close_idx'], n - 1))
                exit_vol = get_vol(pos['ticker'], min(pos['close_idx'], n - 1))
                if exit_vol is None:
                    exit_vol = pos['entry_vol']

                close_cost = (
                    bs_put_price(exit_price, pos['short_strike'], DTE_CLOSE / 365, R, exit_vol)
                    - bs_put_price(exit_price, pos['long_strike'], DTE_CLOSE / 365, R, exit_vol)
                ) * 100 * pos['contracts']

                # Friction on close
                close_commission = COMMISSION_PER_CONTRACT * 2 * pos['contracts']

                pnl = pos['credit_received'] - close_cost - pos['open_commission'] - close_commission
                balance += pos['collateral'] + pnl
                locked -= pos['collateral']

                trade_log.append({
                    'open_date': pd.Timestamp(pos['open_ts'], unit='s'),
                    'close_date': pd.Timestamp(ts, unit='s'),
                    'ticker': pos['ticker'],
                    'sector': UNIVERSE[pos['ticker']],
                    'contracts': pos['contracts'],
                    'collateral': pos['collateral'],
                    'credit': pos['credit_received'],
                    'pnl': pnl,
                    'friction': pos['open_commission'] + close_commission + pos['slippage'],
                    'winner': pnl > 0,
                    'year': pd.Timestamp(pos['open_ts'], unit='s').year,
                })
            else:
                still_open.append(pos)
        positions = still_open

        # --- SELECT and OPEN new positions ---
        if i - last_selection_day >= rebalance_interval and len(positions) < max_total_positions:
            # Which tickers don't already have open positions?
            open_tickers = {p['ticker'] for p in positions}
            candidates = [t for t in UNIVERSE if t not in open_tickers]

            selection = selector_fn(candidates, {'balance': balance, 'locked': locked}, i)

            if selection:
                last_selection_day = i
                slots = max_total_positions - len(positions)
                selected_tickers = list(selection.keys())[:slots]

                for ticker in selected_tickers:
                    price = get_price(ticker, i)
                    vol = get_vol(ticker, i)
                    if vol is None or price <= 0:
                        continue

                    sw = price * SPREAD_PCT
                    sk = price * (1 - BUFFER)
                    lk = sk - sw
                    if lk <= 0:
                        continue

                    sp = put_credit_spread_price(price, sk, lk, DTE_OPEN / 365, R, vol)
                    cr_per = sp['net_credit_dollar']
                    col_per = sw * 100

                    if cr_per <= 0 or col_per <= 0:
                        continue

                    weight = selection.get(ticker, 1.0)
                    max_alloc = balance * max_pct_per_position * weight
                    contracts = min(max_contracts_per_ticker, max(1, int(max_alloc / col_per)))
                    total_col = col_per * contracts
                    total_cr = cr_per * contracts

                    # Friction on open
                    open_commission = COMMISSION_PER_CONTRACT * 2 * contracts
                    slippage = total_cr * SLIPPAGE_PCT

                    if total_col + open_commission > balance:
                        continue

                    balance -= total_col
                    balance += total_cr - slippage  # receive credit minus slippage
                    locked += total_col

                    positions.append({
                        'ticker': ticker,
                        'open_ts': ts,
                        'close_idx': i + offset_close,
                        'entry_price': price,
                        'entry_vol': vol,
                        'short_strike': sk,
                        'long_strike': lk,
                        'contracts': contracts,
                        'collateral': total_col,
                        'credit_received': total_cr,
                        'open_commission': open_commission,
                        'slippage': slippage,
                    })

        # Equity snapshot
        if i % 5 == 0:
            equity_curve.append({
                'date': pd.Timestamp(ts, unit='s'),
                'equity': balance + locked,
                'positions': len(positions),
            })

        i += 1

    trades = pd.DataFrame(trade_log)
    eq = pd.DataFrame(equity_curve)
    final = eq['equity'].iloc[-1] if len(eq) > 0 else starting_capital
    years = (eq['date'].iloc[-1] - eq['date'].iloc[0]).days / 365 if len(eq) > 1 else 1
    cagr = (final / starting_capital) ** (1 / years) - 1 if years > 0 else 0

    peak = eq['equity'].cummax()
    max_dd_pct = ((eq['equity'] - peak) / peak).min() if len(eq) > 1 else 0

    return {
        'trades': trades,
        'equity': eq,
        'final': final,
        'cagr': cagr,
        'max_dd_pct': max_dd_pct,
        'total_friction': trades['friction'].sum() if len(trades) > 0 else 0,
    }


# =====================================================
# RUN ALL STRATEGIES
# =====================================================

W = 115
strategies = {
    'Equal Weight (all 15)':       lambda c, a, i: select_equal_weight(c, a, i),
    'Top 5 Premium':               lambda c, a, i: select_top_premium(c, a, i, 5),
    'Top 8 Premium':               lambda c, a, i: select_top_premium(c, a, i, 8),
    'Momentum Filter':             lambda c, a, i: select_momentum_filter(c, a, i),
    'Momentum + Top 5':            lambda c, a, i: select_momentum_top_premium(c, a, i, 5),
    'Vol-Weighted':                lambda c, a, i: select_vol_weighted(c, a, i),
    'Sector Diversified (2/sect)': lambda c, a, i: select_diversified(c, a, i, 2),
    'Sector Diversified (1/sect)': lambda c, a, i: select_diversified(c, a, i, 1),
}

print("=" * W)
print(f"{'PORTFOLIO SIMULATION: $25,000 ACCOUNT, STRATEGY C, WITH FRICTION':^{W}}")
print(f"{'Max 6 concurrent positions, max 10 contracts/ticker, 15% per position':^{W}}")
print(f"{'Friction: $0.65/contract/leg + 2% bid-ask slippage':^{W}}")
print("=" * W)

results = {}
for name, fn in strategies.items():
    print(f"  Running: {name}...", end="", flush=True)
    r = run_portfolio(fn, starting_capital=25000)
    results[name] = r
    trades = r['trades']
    print(f" {len(trades)} trades, ${r['final']:,.0f}")

# =====================================================
# SUMMARY TABLE
# =====================================================
print()
print("=" * W)
print(f"{'STRATEGY COMPARISON':^{W}}")
print("=" * W)
print(f"{'Strategy':<30} {'Trades':>6} {'WR':>6} {'Final':>11} {'Return':>8} {'CAGR':>6} {'MaxDD':>7} {'Friction':>9} {'Tickers':>8}")
print("-" * W)

for name in strategies:
    r = results[name]
    trades = r['trades']
    if len(trades) == 0:
        continue
    wr = trades['winner'].mean()
    total_ret = (r['final'] / 25000 - 1) * 100
    unique_tickers = trades['ticker'].nunique()
    print(f"{name:<30} {len(trades):>6} {wr:>5.1%} ${r['final']:>10,.0f} {total_ret:>+7.0f}% {r['cagr']:>5.1%} {r['max_dd_pct']:>6.1%} ${r['total_friction']:>8,.0f} {unique_tickers:>8}")

# =====================================================
# DETAILED: BEST STRATEGY YEARLY BREAKDOWN
# =====================================================
best_name = max(results, key=lambda n: results[n]['final'])
r = results[best_name]
trades = r['trades']
eq = r['equity']

print()
print("=" * W)
print(f"  BEST STRATEGY: {best_name}")
print(f"  ${25000:,} -> ${r['final']:,.0f} ({r['cagr']:.1%} CAGR, {r['max_dd_pct']:.1%} max DD)")
print("=" * W)

print(f"\n  {'Year':<6} {'Equity':>10} {'YTD':>8} {'Trades':>7} {'WR':>6} {'Friction':>9} {'Avg Pos':>8}")
print(f"  {'-'*60}")
for year in range(2016, 2027):
    yr_eq = eq[eq['date'].dt.year == year]
    yr_trades = trades[trades['year'] == year]
    if len(yr_eq) == 0:
        continue
    end_eq = yr_eq['equity'].iloc[-1]
    start_eq = yr_eq['equity'].iloc[0]
    ytd = (end_eq - start_eq) / start_eq if start_eq > 0 else 0
    avg_pos = yr_eq['positions'].mean()
    friction = yr_trades['friction'].sum() if len(yr_trades) > 0 else 0
    wr = yr_trades['winner'].mean() if len(yr_trades) > 0 else 0
    print(f"  {year:<6} ${end_eq:>9,.0f} {ytd:>+7.1%} {len(yr_trades):>7} {wr:>5.0%} ${friction:>8,.0f} {avg_pos:>8.1f}")

# Ticker distribution
print(f"\n  Ticker distribution:")
ticker_counts = trades['ticker'].value_counts()
for t, count in ticker_counts.items():
    t_trades = trades[trades['ticker'] == t]
    t_pnl = t_trades['pnl'].sum()
    t_wr = t_trades['winner'].mean()
    print(f"    {t:<5} {count:>4} trades  WR {t_wr:>5.0%}  P/L ${t_pnl:>+8,.0f}")

# Sector distribution
print(f"\n  Sector distribution:")
sector_counts = trades['sector'].value_counts()
for s, count in sector_counts.items():
    s_trades = trades[trades['sector'] == s]
    s_pnl = s_trades['pnl'].sum()
    print(f"    {s:<12} {count:>4} trades  P/L ${s_pnl:>+8,.0f}")

# =====================================================
# FRICTION IMPACT
# =====================================================
print()
print("=" * W)
print(f"{'FRICTION IMPACT ANALYSIS':^{W}}")
print("=" * W)

# Re-run best strategy without friction for comparison
print(f"\n  {best_name}:")
r_with = results[best_name]
# Quick estimate: add friction back to P/L
gross_pnl = trades['pnl'].sum() + r_with['total_friction']
net_pnl = trades['pnl'].sum()
print(f"    Gross P/L (no friction):  ${25000 + gross_pnl:>10,.0f}")
print(f"    Total friction:           ${r_with['total_friction']:>10,.0f}")
print(f"    Net P/L (with friction):  ${r_with['final']:>10,.0f}")
print(f"    Friction as % of gross:   {r_with['total_friction'] / gross_pnl * 100:.1f}%" if gross_pnl > 0 else "")
print(f"    Friction per trade:       ${r_with['total_friction'] / len(trades):>10.2f}")

# =====================================================
# RISK ANALYSIS: WORST DRAWDOWNS
# =====================================================
print()
print("=" * W)
print(f"{'WORST DRAWDOWN PERIODS: ' + best_name:^{W}}")
print("=" * W)

eq_series = eq.set_index('date')['equity']
peak = eq_series.cummax()
dd = (eq_series - peak) / peak

# Find drawdown periods
in_dd = False
dd_start = None
worst_dds = []
for date, val in dd.items():
    if val < -0.01 and not in_dd:
        in_dd = True
        dd_start = date
    elif val >= 0 and in_dd:
        in_dd = False
        dd_bottom = dd[dd_start:date].min()
        worst_dds.append((dd_start, date, dd_bottom))

worst_dds.sort(key=lambda x: x[2])
print(f"\n  {'Start':<12} {'End':<12} {'Depth':>8} {'Duration':>10}")
print(f"  {'-'*45}")
for start, end, depth in worst_dds[:8]:
    days = (end - start).days
    print(f"  {start:%Y-%m-%d}  {end:%Y-%m-%d}  {depth:>+7.1%}  {days:>7} days")

# =====================================================
# STRATEGY RECOMMENDATION
# =====================================================
print()
print("=" * W)
print(f"{'STRATEGY RECOMMENDATIONS':^{W}}")
print("=" * W)

# Rank by risk-adjusted return (CAGR / |max_dd|)
ranked = []
for name, r in results.items():
    if r['max_dd_pct'] != 0:
        sharpe_proxy = r['cagr'] / abs(r['max_dd_pct'])
    else:
        sharpe_proxy = r['cagr']
    ranked.append((name, r['cagr'], r['max_dd_pct'], sharpe_proxy, r['final']))

ranked.sort(key=lambda x: x[3], reverse=True)

print(f"\n  Ranked by risk-adjusted return (CAGR / MaxDD):")
print(f"  {'Rank':<5} {'Strategy':<30} {'CAGR':>6} {'MaxDD':>7} {'CAGR/DD':>8} {'Final':>11}")
print(f"  {'-'*75}")
for rank, (name, cagr, dd, ratio, final) in enumerate(ranked, 1):
    print(f"  {rank:<5} {name:<30} {cagr:>5.1%} {dd:>6.1%} {ratio:>7.2f}x ${final:>10,.0f}")
