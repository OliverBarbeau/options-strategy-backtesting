"""Advanced portfolio construction strategies.

Builds on the portfolio_sim framework with new selection ideas:
- Pullback entry (contrarian: enter when RSI low / stock dipped from highs)
- Regime adaptive (widen buffer in high-vol, narrow in low-vol)
- Risk parity (equal risk contribution per position)
- Performance chasing (overweight recent winners)
- Portfolio heat (throttle when total risk is high)
- Combined best-of (layer multiple filters)
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

UNIVERSE = {
    'QQQ': 'ETF', 'AVGO': 'Semi', 'CAT': 'Industrial', 'MSFT': 'Tech',
    'NVDA': 'Semi', 'META': 'Tech', 'AMD': 'Semi', 'GOOG': 'Tech',
    'AAPL': 'Tech', 'SPY': 'ETF', 'IWM': 'ETF', 'JPM': 'Financial',
    'WMT': 'Consumer', 'HD': 'Consumer', 'XLE': 'Energy',
}

SPREAD_PCT = 0.02
DTE_OPEN = 30
DTE_CLOSE = 14
R = 0.05
COMMISSION_PER_CONTRACT = 0.65
SLIPPAGE_PCT = 0.02

print("Loading data...")
ALL_DATA = {}
ALL_VOL = {}
for ticker in UNIVERSE:
    df = pipe.fetch_stock(ticker, start='2016-01-01', end='2026-04-04')
    ALL_DATA[ticker] = df
    ALL_VOL[ticker] = historical_volatility(df['close'], window=30)

common_idx = ALL_DATA['SPY'].index
print(f"Loaded {len(ALL_DATA)} tickers, {len(common_idx)} trading days\n")


def get_price(ticker, idx):
    df = ALL_DATA[ticker]
    pos = min(np.searchsorted(df.index.values, common_idx[idx]), len(df) - 1)
    return df['close'].iloc[pos]

def get_vol(ticker, idx):
    vol = ALL_VOL[ticker]
    df = ALL_DATA[ticker]
    pos = min(np.searchsorted(df.index.values, common_idx[idx]), len(vol) - 1)
    v = vol.iloc[pos]
    return v if not np.isnan(v) and v > 0 else None

def get_rsi(ticker, idx, period=14):
    df = ALL_DATA[ticker]
    pos = min(np.searchsorted(df.index.values, common_idx[idx]), len(df) - 1)
    if pos < period + 1:
        return None
    prices = df['close'].iloc[pos - period:pos + 1]
    delta = prices.diff().dropna()
    gain = delta.where(delta > 0, 0).mean()
    loss = (-delta.where(delta < 0, 0)).mean()
    if loss == 0:
        return 100
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_drawdown_from_high(ticker, idx, lookback=20):
    df = ALL_DATA[ticker]
    pos = min(np.searchsorted(df.index.values, common_idx[idx]), len(df) - 1)
    if pos < lookback:
        return 0
    recent_high = df['close'].iloc[pos - lookback:pos + 1].max()
    current = df['close'].iloc[pos]
    return (current - recent_high) / recent_high

def get_credit_potential(ticker, idx, buffer=0.10):
    price = get_price(ticker, idx)
    vol = get_vol(ticker, idx)
    if vol is None or price <= 0:
        return None, None, None
    sw = price * SPREAD_PCT
    sk = price * (1 - buffer)
    lk = sk - sw
    if lk <= 0:
        return None, None, None
    sp = put_credit_spread_price(price, sk, lk, DTE_OPEN / 365, R, vol)
    if sp['max_loss'] <= 0 or sp['net_credit_dollar'] <= 0:
        return None, None, None
    return sp['credit_potential'], sp['net_credit_dollar'], sp['max_loss']

def spy_vol(idx):
    return get_vol('SPY', idx)


# =====================================================
# STRATEGY FUNCTIONS
# =====================================================
# Each returns: dict of {ticker: (weight, buffer_override)} or {ticker: weight}

def strat_top8_baseline(candidates, state, idx):
    """Baseline: Top 8 by premium (from previous experiment winner)."""
    scored = []
    for t in candidates:
        cp, _, _ = get_credit_potential(t, idx)
        if cp is not None:
            scored.append((t, cp))
    scored.sort(key=lambda x: x[1], reverse=True)
    return {t: {'weight': 1.0, 'buffer': 0.10} for t, _ in scored[:8]}


def strat_pullback_entry(candidates, state, idx):
    """Contrarian: only enter tickers that have pulled back >3% from 20d high.
    These have elevated vol = richer premiums, and are more likely to mean-revert."""
    scored = []
    for t in candidates:
        dd = get_drawdown_from_high(t, idx, 20)
        if dd < -0.03:  # stock has pulled back 3%+ from recent high
            cp, _, _ = get_credit_potential(t, idx)
            if cp is not None:
                scored.append((t, cp, dd))
    scored.sort(key=lambda x: x[1], reverse=True)
    return {t: {'weight': 1.0, 'buffer': 0.10} for t, _, _ in scored[:8]}


def strat_rsi_oversold(candidates, state, idx):
    """Enter when RSI < 45 (mild oversold). Pick top premium among those."""
    scored = []
    for t in candidates:
        rsi = get_rsi(t, idx)
        if rsi is not None and rsi < 45:
            cp, _, _ = get_credit_potential(t, idx)
            if cp is not None:
                scored.append((t, cp, rsi))
    scored.sort(key=lambda x: x[1], reverse=True)
    return {t: {'weight': 1.0, 'buffer': 0.10} for t, _, _ in scored[:8]}


def strat_regime_adaptive(candidates, state, idx):
    """Adapt buffer to market regime:
    - Low vol (SPY HV30 < 15%): tighter buffer (7%), more trades
    - Normal vol (15-25%): standard buffer (10%)
    - High vol (>25%): wide buffer (13%), fewer but safer trades
    """
    sv = spy_vol(idx)
    if sv is None:
        return {}
    if sv < 0.15:
        buffer = 0.07
    elif sv < 0.25:
        buffer = 0.10
    else:
        buffer = 0.13

    scored = []
    for t in candidates:
        cp, _, _ = get_credit_potential(t, idx, buffer)
        if cp is not None:
            scored.append((t, cp))
    scored.sort(key=lambda x: x[1], reverse=True)
    return {t: {'weight': 1.0, 'buffer': buffer} for t, _ in scored[:8]}


def strat_risk_parity(candidates, state, idx):
    """Size each position so max loss is equal across all positions.
    High-priced tickers get fewer contracts, low-priced get more."""
    scored = []
    for t in candidates:
        cp, credit, maxloss = get_credit_potential(t, idx)
        if cp is not None:
            scored.append((t, cp, maxloss))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:8]
    if not top:
        return {}
    # Invert max_loss for weighting (lower max_loss = higher weight)
    max_losses = [m for _, _, m in top]
    avg_ml = np.mean(max_losses)
    return {t: {'weight': avg_ml / ml if ml > 0 else 1.0, 'buffer': 0.10}
            for t, _, ml in top}


def strat_performance_chase(candidates, state, idx):
    """Overweight tickers that have been profitable in recent trades.
    Uses trailing P/L from the trade log."""
    recent_pnl = state.get('recent_pnl', {})
    scored = []
    for t in candidates:
        cp, _, _ = get_credit_potential(t, idx)
        if cp is not None:
            # Bonus for recent winners
            trail = recent_pnl.get(t, 0)
            adj_score = cp * (1 + max(0, trail / 100))
            scored.append((t, adj_score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return {t: {'weight': 1.0, 'buffer': 0.10} for t, _ in scored[:8]}


def strat_heat_control(candidates, state, idx):
    """Reduce position count when portfolio is in drawdown.
    Normal: up to 6 positions. In drawdown >5%: max 3. >10%: max 1."""
    eq = state.get('equity', 25000)
    peak = state.get('peak_equity', 25000)
    dd = (eq - peak) / peak if peak > 0 else 0

    if dd < -0.10:
        max_pos = 1
    elif dd < -0.05:
        max_pos = 3
    else:
        max_pos = 6

    current_open = state.get('open_count', 0)
    if current_open >= max_pos:
        return {}

    scored = []
    for t in candidates:
        cp, _, _ = get_credit_potential(t, idx)
        if cp is not None:
            scored.append((t, cp))
    scored.sort(key=lambda x: x[1], reverse=True)
    slots = max_pos - current_open
    return {t: {'weight': 1.0, 'buffer': 0.10} for t, _ in scored[:slots]}


def strat_combined_best(candidates, state, idx):
    """Best-of: regime-adaptive buffer + pullback preference + top premium.
    Layer filters intelligently."""
    sv = spy_vol(idx)
    if sv is None:
        return {}
    if sv < 0.15:
        buffer = 0.07
    elif sv < 0.25:
        buffer = 0.10
    else:
        buffer = 0.13

    scored = []
    for t in candidates:
        cp, _, _ = get_credit_potential(t, idx, buffer)
        if cp is None:
            continue
        dd = get_drawdown_from_high(t, idx, 20)
        # Boost score for pullbacks (more premium + mean-reversion tailwind)
        pullback_bonus = 1.0 + min(0.5, abs(dd) * 5) if dd < -0.02 else 1.0
        scored.append((t, cp * pullback_bonus, buffer))
    scored.sort(key=lambda x: x[1], reverse=True)
    return {t: {'weight': 1.0, 'buffer': buf} for t, _, buf in scored[:8]}


# =====================================================
# PORTFOLIO SIMULATOR (enhanced with per-ticker buffer)
# =====================================================

def run_portfolio(
    selector_fn,
    starting_capital=25000,
    max_contracts_per_ticker=10,
    max_total_positions=6,
    max_pct_per_position=0.15,
    rebalance_interval=21,
):
    offset_close = max(1, int((DTE_OPEN - DTE_CLOSE) * 21 / 30))

    balance = float(starting_capital)
    locked = 0.0
    positions = []
    trade_log = []
    equity_curve = []
    recent_pnl = {}  # trailing P/L per ticker
    peak_equity = starting_capital

    n = len(common_idx)
    i = 50
    last_selection_day = 0

    while i < n:
        ts = common_idx[i]
        total_equity = balance + locked

        # Track peak
        if total_equity > peak_equity:
            peak_equity = total_equity

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

                close_commission = COMMISSION_PER_CONTRACT * 2 * pos['contracts']
                pnl = pos['credit_received'] - close_cost - pos['open_commission'] - close_commission
                balance += pos['collateral'] + pnl
                locked -= pos['collateral']

                # Track recent P/L per ticker (exponential decay)
                old = recent_pnl.get(pos['ticker'], 0)
                recent_pnl[pos['ticker']] = old * 0.7 + pnl

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
                    'buffer_used': pos.get('buffer', 0.10),
                })
            else:
                still_open.append(pos)
        positions = still_open

        # --- SELECT and OPEN ---
        if i - last_selection_day >= rebalance_interval and len(positions) < max_total_positions:
            open_tickers = {p['ticker'] for p in positions}
            candidates = [t for t in UNIVERSE if t not in open_tickers]

            state = {
                'balance': balance, 'locked': locked,
                'equity': balance + locked, 'peak_equity': peak_equity,
                'recent_pnl': recent_pnl,
                'open_count': len(positions),
            }
            selection = selector_fn(candidates, state, i)

            if selection:
                last_selection_day = i
                slots = max_total_positions - len(positions)
                selected_tickers = list(selection.keys())[:slots]

                for ticker in selected_tickers:
                    info = selection[ticker]
                    if isinstance(info, dict):
                        weight = info.get('weight', 1.0)
                        buffer = info.get('buffer', 0.10)
                    else:
                        weight = info
                        buffer = 0.10

                    price = get_price(ticker, i)
                    vol = get_vol(ticker, i)
                    if vol is None or price <= 0:
                        continue

                    sw = price * SPREAD_PCT
                    sk = price * (1 - buffer)
                    lk = sk - sw
                    if lk <= 0:
                        continue

                    sp = put_credit_spread_price(price, sk, lk, DTE_OPEN / 365, R, vol)
                    cr_per = sp['net_credit_dollar']
                    col_per = sw * 100
                    if cr_per <= 0 or col_per <= 0:
                        continue

                    max_alloc = balance * max_pct_per_position * min(weight, 2.0)
                    contracts = min(max_contracts_per_ticker, max(1, int(max_alloc / col_per)))
                    total_col = col_per * contracts
                    total_cr = cr_per * contracts

                    open_commission = COMMISSION_PER_CONTRACT * 2 * contracts
                    slippage = total_cr * SLIPPAGE_PCT

                    if total_col + open_commission > balance:
                        continue

                    balance -= total_col
                    balance += total_cr - slippage
                    locked += total_col

                    positions.append({
                        'ticker': ticker, 'open_ts': ts,
                        'close_idx': i + offset_close,
                        'entry_price': price, 'entry_vol': vol,
                        'short_strike': sk, 'long_strike': lk,
                        'contracts': contracts, 'collateral': total_col,
                        'credit_received': total_cr,
                        'open_commission': open_commission,
                        'slippage': slippage,
                        'buffer': buffer,
                    })

        if i % 5 == 0:
            equity_curve.append({
                'date': pd.Timestamp(ts, unit='s'),
                'equity': balance + locked,
                'positions': len(positions),
            })

        i += 1

    trades = pd.DataFrame(trade_log)
    eq = pd.DataFrame(equity_curve)
    if eq.empty:
        return {'trades': trades, 'equity': eq, 'final': starting_capital,
                'cagr': 0, 'max_dd_pct': 0, 'total_friction': 0}

    final = eq['equity'].iloc[-1]
    years = (eq['date'].iloc[-1] - eq['date'].iloc[0]).days / 365
    cagr = (final / starting_capital) ** (1 / years) - 1 if years > 0 else 0
    peak = eq['equity'].cummax()
    max_dd_pct = ((eq['equity'] - peak) / peak).min()

    return {
        'trades': trades, 'equity': eq, 'final': final,
        'cagr': cagr, 'max_dd_pct': max_dd_pct,
        'total_friction': trades['friction'].sum() if len(trades) > 0 else 0,
    }


# =====================================================
# RUN ALL STRATEGIES
# =====================================================

strategies = {
    'Top 8 Premium (baseline)':  strat_top8_baseline,
    'Pullback Entry (3%+ dip)':  strat_pullback_entry,
    'RSI Oversold (<45)':        strat_rsi_oversold,
    'Regime Adaptive Buffer':    strat_regime_adaptive,
    'Risk Parity Sizing':        strat_risk_parity,
    'Performance Chase':         strat_performance_chase,
    'Heat Control (DD sizing)':  strat_heat_control,
    'Combined Best-Of':          strat_combined_best,
}

W = 115
print("=" * W)
print(f"{'ADVANCED PORTFOLIO STRATEGIES: $25K, 10 YEARS, WITH FRICTION':^{W}}")
print("=" * W)

results = {}
for name, fn in strategies.items():
    print(f"  Running: {name}...", end="", flush=True)
    r = run_portfolio(fn, starting_capital=25000)
    results[name] = r
    trades = r['trades']
    wr = trades['winner'].mean() if len(trades) > 0 else 0
    print(f" {len(trades)} trades, WR {wr:.0%}, ${r['final']:,.0f}")

# =====================================================
# COMPARISON TABLE
# =====================================================
print()
print("=" * W)
print(f"{'STRATEGY COMPARISON (sorted by risk-adjusted return)':^{W}}")
print("=" * W)
print(f"{'Strategy':<30} {'Trades':>6} {'WR':>6} {'Final':>11} {'CAGR':>6} {'MaxDD':>7} {'CAGR/DD':>8} {'Friction':>9}")
print("-" * W)

ranked = []
for name, r in results.items():
    trades = r['trades']
    if len(trades) == 0:
        continue
    ratio = r['cagr'] / abs(r['max_dd_pct']) if r['max_dd_pct'] != 0 else 0
    ranked.append((name, r, ratio))

ranked.sort(key=lambda x: x[2], reverse=True)

for name, r, ratio in ranked:
    trades = r['trades']
    wr = trades['winner'].mean()
    unique = trades['ticker'].nunique()
    print(f"{name:<30} {len(trades):>6} {wr:>5.1%} ${r['final']:>10,.0f} {r['cagr']:>5.1%} {r['max_dd_pct']:>6.1%} {ratio:>7.2f}x ${r['total_friction']:>8,.0f}")

# =====================================================
# YEARLY COMPARISON: TOP 3
# =====================================================
top3_names = [name for name, _, _ in ranked[:3]]
print()
print("=" * W)
print(f"{'YEARLY EQUITY: TOP 3 STRATEGIES':^{W}}")
print("=" * W)
print(f"{'Year':<6}", end="")
for name in top3_names:
    print(f"  {name[:25]:>28}", end="")
print()
print("-" * (6 + 30 * len(top3_names)))

for year in range(2016, 2027):
    print(f"{year:<6}", end="")
    for name in top3_names:
        eq = results[name]['equity']
        yr = eq[eq['date'].dt.year == year]
        if len(yr) < 2:
            print(f"{'--':>30}", end="")
        else:
            end_eq = yr['equity'].iloc[-1]
            start_eq = yr['equity'].iloc[0]
            ytd = (end_eq - start_eq) / start_eq
            print(f"  ${end_eq:>12,.0f} ({ytd:>+5.1%})", end="")
    print()

# =====================================================
# DEEP DIVE: BEST STRATEGY
# =====================================================
best_name = ranked[0][0]
r = results[best_name]
trades = r['trades']

print()
print("=" * W)
print(f"  BEST STRATEGY: {best_name}")
print(f"  $25,000 -> ${r['final']:,.0f} ({r['cagr']:.1%} CAGR, {r['max_dd_pct']:.1%} max DD)")
print("=" * W)

# Ticker breakdown
print(f"\n  Ticker breakdown:")
print(f"  {'Ticker':<6} {'Trades':>6} {'WR':>6} {'P/L':>9} {'Avg P/L':>8} {'Sector':<12}")
print(f"  {'-'*55}")
for t in sorted(trades['ticker'].unique()):
    tt = trades[trades['ticker'] == t]
    print(f"  {t:<6} {len(tt):>6} {tt['winner'].mean():>5.0%} ${tt['pnl'].sum():>+8,.0f} ${tt['pnl'].mean():>+7.2f} {UNIVERSE[t]:<12}")

# Buffer usage (for regime-adaptive strategies)
if 'buffer_used' in trades.columns:
    buffers = trades['buffer_used'].value_counts()
    if len(buffers) > 1:
        print(f"\n  Buffer usage:")
        for buf, count in buffers.items():
            buf_trades = trades[trades['buffer_used'] == buf]
            print(f"    {buf:.0%} buffer: {count} trades, WR {buf_trades['winner'].mean():.0%}, P/L ${buf_trades['pnl'].sum():+,.0f}")

# =====================================================
# BEAR MARKET COMPARISON: TOP 3
# =====================================================
print()
print("=" * W)
print(f"{'BEAR MARKET PERFORMANCE: TOP 3':^{W}}")
print("=" * W)

bear_periods = {
    '2018 Q4': (2018, [10, 11, 12]),
    '2020 COVID': (2020, [2, 3, 4]),
    '2022 H1': (2022, [1, 2, 3, 4, 5, 6]),
    '2025 Q1': (2025, [1, 2, 3]),
}

for period, (year, months) in bear_periods.items():
    print(f"\n  {period}:")
    print(f"  {'Strategy':<30} {'Trades':>6} {'WR':>6} {'P/L':>9}")
    print(f"  {'-'*55}")
    for name in top3_names:
        trades = results[name]['trades']
        mask = (trades['year'] == year) & (trades['close_date'].dt.month.isin(months))
        period_trades = trades[mask]
        if len(period_trades) == 0:
            print(f"  {name:<30} {'--':>6}")
        else:
            print(f"  {name:<30} {len(period_trades):>6} {period_trades['winner'].mean():>5.0%} ${period_trades['pnl'].sum():>+8,.0f}")

# =====================================================
# SUMMARY
# =====================================================
print()
print("=" * W)
print(f"{'FINDINGS':^{W}}")
print("=" * W)

best = ranked[0]
second = ranked[1]
third = ranked[2]

print(f"""
  1. BEST RISK-ADJUSTED: {best[0]}
     CAGR: {best[1]['cagr']:.1%}  MaxDD: {best[1]['max_dd_pct']:.1%}  Ratio: {best[2]:.2f}x

  2. RUNNER UP: {second[0]}
     CAGR: {second[1]['cagr']:.1%}  MaxDD: {second[1]['max_dd_pct']:.1%}  Ratio: {second[2]:.2f}x

  3. THIRD: {third[0]}
     CAGR: {third[1]['cagr']:.1%}  MaxDD: {third[1]['max_dd_pct']:.1%}  Ratio: {third[2]:.2f}x
""")
