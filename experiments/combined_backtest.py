"""Combined backtest: iron condors + calendars + pullback + regime adaptive.

Tests each spread type independently, then as portfolio additions.
"""

import warnings
warnings.filterwarnings("ignore")

from tradelab.pipeline import DataPipeline
from tradelab.options import (
    bs_put_price, bs_call_price,
    put_credit_spread_price, iron_condor_price, calendar_spread_price,
    historical_volatility,
)
from tradelab.strategies.pullback_entry import PullbackEntryStrategy
from tradelab.strategies.regime_adaptive import RegimeAdaptiveStrategy
import pandas as pd
import numpy as np

pipe = DataPipeline()

TICKERS = ['QQQ', 'AVGO', 'CAT', 'MSFT', 'NVDA', 'META', 'AMD', 'GOOG']
R = 0.05
DTE_OPEN = 30
DTE_CLOSE = 14
COMMISSION = 0.65
SLIPPAGE = 0.02

print("Loading data...")
DATA = {}
VOL = {}
for t in TICKERS + ['SPY']:
    DATA[t] = pipe.fetch_stock(t, start='2016-01-01', end='2026-04-04')
    VOL[t] = historical_volatility(DATA[t]['close'], window=30)

SPY_VOL = VOL['SPY']
print(f"Loaded {len(DATA)} tickers\n")


# =====================================================
# IRON CONDOR BACKTESTER
# =====================================================

def backtest_iron_condor(ticker, buffer=0.10, spread_pct=0.02, max_contracts=10):
    df = DATA[ticker]
    close = df['close'].values
    timestamps = df.index.values
    vol = VOL[ticker]
    offset_close = max(1, int((DTE_OPEN - DTE_CLOSE) * 21 / 30))

    trades = []
    i = 30
    while i < len(df) - int(DTE_OPEN * 21 / 30):
        price = close[i]
        v = vol.iloc[i]
        if np.isnan(v) or v <= 0:
            i += 1
            continue

        sw = price * spread_pct
        put_sk, put_lk = price * (1 - buffer), price * (1 - buffer) - sw
        call_sk, call_lk = price * (1 + buffer), price * (1 + buffer) + sw

        if put_lk <= 0:
            i += offset_close
            continue

        ic = iron_condor_price(price, put_sk, put_lk, call_sk, call_lk, DTE_OPEN / 365, R, v)
        credit = ic['total_credit_dollar']
        max_loss = ic['max_loss']
        if credit <= 0 or max_loss <= 0:
            i += offset_close
            continue

        # Friction
        legs = 4
        open_comm = COMMISSION * legs * max_contracts
        slip = credit * max_contracts * SLIPPAGE
        net_credit = credit * max_contracts - slip

        close_idx = min(i + offset_close, len(close) - 1)
        exit_price = close[close_idx]
        exit_vol = vol.iloc[close_idx] if close_idx < len(vol) else v
        if np.isnan(exit_vol) or exit_vol <= 0:
            exit_vol = v

        T_rem = DTE_CLOSE / 365
        put_close = (bs_put_price(exit_price, put_sk, T_rem, R, exit_vol)
                     - bs_put_price(exit_price, put_lk, T_rem, R, exit_vol)) * 100
        call_close = (bs_call_price(exit_price, call_sk, T_rem, R, exit_vol)
                      - bs_call_price(exit_price, call_lk, T_rem, R, exit_vol)) * 100
        close_cost = (put_close + call_close) * max_contracts
        close_comm = COMMISSION * legs * max_contracts

        pnl = net_credit - close_cost - open_comm - close_comm

        trades.append({
            'date': pd.Timestamp(timestamps[i], unit='s'),
            'ticker': ticker, 'type': 'iron_condor',
            'entry_price': price, 'exit_price': exit_price,
            'credit': net_credit, 'pnl': pnl, 'winner': pnl > 0,
            'sigma': v, 'contracts': max_contracts,
            'friction': open_comm + close_comm + slip,
        })
        i += offset_close

    return pd.DataFrame(trades)


# =====================================================
# CALENDAR SPREAD BACKTESTER
# =====================================================

def backtest_calendar(ticker, strike_pct=0.95, max_contracts=10):
    """Sell 14 DTE put, buy 45 DTE put at same strike. Close at near expiry."""
    df = DATA[ticker]
    close = df['close'].values
    timestamps = df.index.values
    vol = VOL[ticker]
    offset_near = max(1, int(14 * 21 / 30))  # ~10 trading days

    trades = []
    i = 30
    while i < len(df) - int(45 * 21 / 30):
        price = close[i]
        v = vol.iloc[i]
        if np.isnan(v) or v <= 0:
            i += 1
            continue

        strike = price * strike_pct
        cs = calendar_spread_price(price, strike, 14 / 365, 45 / 365, R, v, "put")
        net_debit = cs['net_debit']  # we pay this
        if net_debit <= 0:
            i += offset_near
            continue

        # Friction (2 legs open, 2 legs close)
        open_comm = COMMISSION * 2 * max_contracts
        total_debit = net_debit * max_contracts

        # At near expiry (14 DTE), reprice the far leg
        close_idx = min(i + offset_near, len(close) - 1)
        exit_price = close[close_idx]
        exit_vol = vol.iloc[close_idx] if close_idx < len(vol) else v
        if np.isnan(exit_vol) or exit_vol <= 0:
            exit_vol = v

        # Near leg expires/is nearly worthless, far leg still has value
        near_at_expiry = bs_put_price(exit_price, strike, 1 / 365, R, exit_vol) * 100
        far_remaining = bs_put_price(exit_price, strike, 31 / 365, R, exit_vol) * 100
        # We close: buy back near (cheap), sell the far
        recovery = (far_remaining - near_at_expiry) * max_contracts
        close_comm = COMMISSION * 2 * max_contracts

        pnl = recovery - total_debit - open_comm - close_comm

        trades.append({
            'date': pd.Timestamp(timestamps[i], unit='s'),
            'ticker': ticker, 'type': 'calendar',
            'entry_price': price, 'exit_price': exit_price,
            'credit': -total_debit, 'pnl': pnl, 'winner': pnl > 0,
            'sigma': v, 'contracts': max_contracts,
            'friction': open_comm + close_comm,
        })
        i += offset_near

    return pd.DataFrame(trades)


W = 110

# =====================================================
# 1. INDIVIDUAL SPREAD TYPE COMPARISON
# =====================================================
print("=" * W)
print(f"{'SPREAD TYPE COMPARISON: 10 YEARS, PER-CONTRACT, WITH FRICTION':^{W}}")
print("=" * W)

all_type_results = {}

for spread_type, backtest_fn in [
    ('Put Credit Spread', lambda t: backtest_iron_condor(t, buffer=0.10, spread_pct=0.02)),  # just the put side
    ('Iron Condor', lambda t: backtest_iron_condor(t, buffer=0.10, spread_pct=0.02)),
    ('Calendar Spread', lambda t: backtest_calendar(t, strike_pct=0.95)),
]:
    type_results = []
    for ticker in TICKERS:
        if spread_type == 'Put Credit Spread':
            # Use the pullback strategy for PCS
            strat = PullbackEntryStrategy()
            result = strat.run(DATA[ticker], max_contracts=10)
            trades_df = pd.DataFrame(result.trade_log)
            if len(trades_df) > 0:
                trades_df['type'] = 'pcs'
                trades_df['ticker'] = ticker
                trades_df['friction'] = 0  # already included
                type_results.append(trades_df)
        else:
            trades_df = backtest_fn(ticker)
            if len(trades_df) > 0:
                type_results.append(trades_df)

    if type_results:
        combined = pd.concat(type_results, ignore_index=True)
        all_type_results[spread_type] = combined

print(f"\n{'Spread Type':<22} {'Trades':>6} {'WR':>6} {'Total P/L':>11} {'$/Trade':>9} {'AvgFriction':>12} {'Tickers':>8}")
print("-" * W)

for spread_type, trades in all_type_results.items():
    wr = trades['winner'].mean()
    total = trades['pnl'].sum()
    avg = trades['pnl'].mean()
    avg_fric = trades['friction'].mean() if 'friction' in trades.columns else 0
    unique = trades['ticker'].nunique()
    print(f"{spread_type:<22} {len(trades):>6} {wr:>5.1%} ${total:>+10,.0f} ${avg:>+8.2f} ${avg_fric:>11.2f} {unique:>8}")

# =====================================================
# 2. IRON CONDOR DETAIL BY TICKER
# =====================================================
print()
print("=" * W)
print(f"{'IRON CONDOR RESULTS BY TICKER':^{W}}")
print("=" * W)
print(f"{'Ticker':<6} {'Trades':>6} {'WR':>6} {'P/L':>10} {'$/Trade':>8} {'AvgCredit':>10} {'Friction':>9}")
print("-" * 65)

ic_trades = all_type_results.get('Iron Condor', pd.DataFrame())
if len(ic_trades) > 0:
    for t in TICKERS:
        tt = ic_trades[ic_trades['ticker'] == t]
        if len(tt) == 0:
            continue
        print(f"{t:<6} {len(tt):>6} {tt['winner'].mean():>5.1%} ${tt['pnl'].sum():>+9,.0f} "
              f"${tt['pnl'].mean():>+7.2f} ${tt['credit'].mean():>9.1f} ${tt['friction'].sum():>8,.0f}")

# =====================================================
# 3. CALENDAR SPREAD DETAIL BY TICKER
# =====================================================
print()
print("=" * W)
print(f"{'CALENDAR SPREAD RESULTS BY TICKER':^{W}}")
print("=" * W)
print(f"{'Ticker':<6} {'Trades':>6} {'WR':>6} {'P/L':>10} {'$/Trade':>8} {'Friction':>9}")
print("-" * 55)

cal_trades = all_type_results.get('Calendar Spread', pd.DataFrame())
if len(cal_trades) > 0:
    for t in TICKERS:
        tt = cal_trades[cal_trades['ticker'] == t]
        if len(tt) == 0:
            continue
        print(f"{t:<6} {len(tt):>6} {tt['winner'].mean():>5.1%} ${tt['pnl'].sum():>+9,.0f} "
              f"${tt['pnl'].mean():>+7.2f} ${tt['friction'].sum():>8,.0f}")

# =====================================================
# 4. PRODUCTION STRATEGIES: PULLBACK + REGIME ADAPTIVE
# =====================================================
print()
print("=" * W)
print(f"{'PRODUCTION STRATEGIES: PULLBACK ENTRY + REGIME ADAPTIVE':^{W}}")
print("=" * W)

for strat_name, strat_class in [
    ('Pullback Entry', PullbackEntryStrategy),
    ('Regime Adaptive', RegimeAdaptiveStrategy),
]:
    print(f"\n  {strat_name}:")
    print(f"  {'Ticker':<6} {'Trades':>6} {'WR':>6} {'P/L':>10} {'$/Trade':>8}")
    print(f"  {'-'*42}")

    total_pnl = 0
    total_trades = 0
    total_wins = 0

    for ticker in TICKERS:
        if strat_name == 'Pullback Entry':
            strat = PullbackEntryStrategy()
            result = strat.run(DATA[ticker], max_contracts=10)
        else:
            strat = RegimeAdaptiveStrategy()
            result = strat.run(DATA[ticker], market_vol_series=SPY_VOL, max_contracts=10)

        if result.total_trades == 0:
            continue
        avg = result.total_pnl / result.total_trades
        print(f"  {ticker:<6} {result.total_trades:>6} {result.win_rate:>5.1%} "
              f"${result.total_pnl:>+9,.0f} ${avg:>+7.2f}")
        total_pnl += result.total_pnl
        total_trades += result.total_trades
        total_wins += result.winners

    if total_trades > 0:
        print(f"  {'TOTAL':<6} {total_trades:>6} {total_wins/total_trades:>5.1%} "
              f"${total_pnl:>+9,.0f} ${total_pnl/total_trades:>+7.2f}")

# =====================================================
# 5. COMBINED PORTFOLIO: MIX SPREAD TYPES
# =====================================================
print()
print("=" * W)
print(f"{'COMBINED PORTFOLIO: MIXING SPREAD TYPES':^{W}}")
print("=" * W)

# Combine: PCS (pullback) + Iron Condors on high-vol names + Calendars
print("""
  Portfolio allocation:
  - Put Credit Spreads (pullback entry): primary, all 8 tickers
  - Iron Condors: on tickers with HV30 > 35% (captures both sides)
  - Calendar Spreads: on tickers with HV30 > median (exploit decay)
""")

# Aggregate all trades with type labels
all_portfolio_trades = []

for ticker in TICKERS:
    # PCS trades (pullback)
    strat = PullbackEntryStrategy()
    result = strat.run(DATA[ticker], max_contracts=5)  # half-size for diversification
    for t in result.trade_log:
        t['type'] = 'PCS'
        t['ticker'] = ticker
    all_portfolio_trades.extend(result.trade_log)

    # Iron condors on high-vol names
    current_vol = VOL[ticker].dropna().median()
    if current_vol > 0.30:
        ic = backtest_iron_condor(ticker, max_contracts=3)
        for _, row in ic.iterrows():
            all_portfolio_trades.append(row.to_dict())

    # Calendars on moderate+ vol names
    if current_vol > 0.20:
        cal = backtest_calendar(ticker, max_contracts=3)
        for _, row in cal.iterrows():
            all_portfolio_trades.append(row.to_dict())

portfolio_df = pd.DataFrame(all_portfolio_trades)

print(f"{'Type':<18} {'Trades':>6} {'WR':>6} {'Total P/L':>11} {'$/Trade':>9}")
print("-" * 55)
for stype in portfolio_df['type'].unique():
    sub = portfolio_df[portfolio_df['type'] == stype]
    print(f"{stype:<18} {len(sub):>6} {sub['winner'].mean():>5.1%} "
          f"${sub['pnl'].sum():>+10,.0f} ${sub['pnl'].mean():>+8.2f}")

print("-" * 55)
print(f"{'COMBINED':<18} {len(portfolio_df):>6} {portfolio_df['winner'].mean():>5.1%} "
      f"${portfolio_df['pnl'].sum():>+10,.0f} ${portfolio_df['pnl'].mean():>+8.2f}")

# Yearly breakdown
print(f"\n  Yearly P/L by spread type:")
portfolio_df['year'] = pd.to_datetime(portfolio_df['date']).dt.year
yearly = portfolio_df.groupby(['year', 'type'])['pnl'].sum().unstack(fill_value=0)
yearly['TOTAL'] = yearly.sum(axis=1)
print(yearly.to_string())

# =====================================================
# 6. FINAL COMPARISON
# =====================================================
print()
print("=" * W)
print(f"{'FINAL COMPARISON: WHICH APPROACH WINS?':^{W}}")
print("=" * W)

approaches = {
    'PCS only (pullback)': portfolio_df[portfolio_df['type'] == 'PCS'],
    'Iron Condors only': portfolio_df[portfolio_df['type'] == 'iron_condor'],
    'Calendars only': portfolio_df[portfolio_df['type'] == 'calendar'],
    'Combined portfolio': portfolio_df,
}

print(f"\n{'Approach':<25} {'Trades':>6} {'WR':>6} {'Total P/L':>11} {'$/Trade':>9} {'P/L/Year':>10}")
print("-" * 75)
for name, df in approaches.items():
    if len(df) == 0:
        continue
    years = 10.3  # approximate
    print(f"{name:<25} {len(df):>6} {df['winner'].mean():>5.1%} "
          f"${df['pnl'].sum():>+10,.0f} ${df['pnl'].mean():>+8.2f} "
          f"${df['pnl'].sum()/years:>+9,.0f}")
