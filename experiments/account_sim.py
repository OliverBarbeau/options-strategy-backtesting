"""Account simulation: Strategy C with proper position sizing and buying power.

Models a real brokerage account:
- Starting capital
- Collateral locked per trade (spread width * 100 * num_contracts)
- Position sizing: allocate X% of available buying power per trade
- Track: balance, buying power, locked collateral, P&L over time
- Compound: re-invest gains into larger positions
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

TICKERS = ['QQQ', 'AVGO', 'CAT', 'MSFT', 'NVDA', 'META', 'AMD', 'GOOG']

# Strategy params
BUFFER = 0.10
SPREAD_PCT = 0.02
DTE_OPEN = 30
DTE_CLOSE = 14
R = 0.05


def simulate_account(
    ticker,
    starting_capital=10000,
    max_position_pct=0.20,    # max 20% of buying power per trade
    max_concurrent=3,          # max 3 overlapping positions
):
    df = pipe.fetch_stock(ticker, start='2016-01-01', end='2026-04-04')
    if len(df) < 200:
        return None

    close = df['close'].values
    timestamps = df.index.values
    vol = historical_volatility(df['close'], window=30)

    offset_open = max(1, int(DTE_OPEN * 21 / 30))
    offset_close = max(1, int((DTE_OPEN - DTE_CLOSE) * 21 / 30))

    # Account state
    balance = starting_capital
    locked = 0.0  # collateral locked in open positions
    open_positions = []  # list of dicts
    trade_log = []
    equity_curve = []

    peak_balance = starting_capital
    max_drawdown = 0.0
    total_collateral_deployed = 0.0

    i = 30
    while i < len(df):
        price = close[i]
        ts = timestamps[i]
        vol_val = vol.iloc[i] if i < len(vol) else np.nan

        # --- CHECK: close any positions at their 14 DTE mark ---
        still_open = []
        for pos in open_positions:
            if i >= pos['close_idx']:
                # Close this position
                exit_price = close[min(pos['close_idx'], len(close) - 1)]
                exit_vol = vol.iloc[min(pos['close_idx'], len(vol) - 1)]
                if np.isnan(exit_vol) or exit_vol <= 0:
                    exit_vol = pos['entry_vol']

                T_remaining = DTE_CLOSE / 365.0
                close_cost = (
                    bs_put_price(exit_price, pos['short_strike'], T_remaining, R, exit_vol)
                    - bs_put_price(exit_price, pos['long_strike'], T_remaining, R, exit_vol)
                ) * 100 * pos['contracts']

                pnl = pos['credit_received'] - close_cost
                balance += pos['collateral'] + pnl  # return collateral + P&L
                locked -= pos['collateral']

                trade_log.append({
                    'date': pd.Timestamp(pos['entry_ts'], unit='s'),
                    'exit_date': pd.Timestamp(timestamps[min(pos['close_idx'], len(timestamps) - 1)], unit='s'),
                    'ticker': ticker,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'contracts': pos['contracts'],
                    'collateral': pos['collateral'],
                    'credit': pos['credit_received'],
                    'pnl': pnl,
                    'winner': pnl > 0,
                    'return_on_collateral': pnl / pos['collateral'] if pos['collateral'] > 0 else 0,
                    'balance_after': balance + locked,
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        # --- CHECK: open new position? ---
        if (
            len(open_positions) < max_concurrent
            and not np.isnan(vol_val)
            and vol_val > 0
            and i + offset_open < len(df)
        ):
            spread_width = price * SPREAD_PCT
            short_strike = price * (1 - BUFFER)
            long_strike = short_strike - spread_width

            if long_strike > 0:
                sp = put_credit_spread_price(price, short_strike, long_strike, DTE_OPEN / 365, R, vol_val)
                credit_per_contract = sp['net_credit_dollar']
                collateral_per_contract = spread_width * 100

                if credit_per_contract > 0 and collateral_per_contract > 0:
                    buying_power = balance  # what's not locked
                    max_alloc = buying_power * max_position_pct
                    contracts = max(1, int(max_alloc / collateral_per_contract))
                    total_collateral = collateral_per_contract * contracts
                    total_credit = credit_per_contract * contracts

                    if total_collateral <= balance:
                        balance -= total_collateral  # lock collateral
                        balance += total_credit       # receive credit upfront
                        locked += total_collateral
                        total_collateral_deployed += total_collateral

                        open_positions.append({
                            'entry_ts': ts,
                            'entry_idx': i,
                            'close_idx': i + offset_close,
                            'entry_price': price,
                            'entry_vol': vol_val,
                            'short_strike': short_strike,
                            'long_strike': long_strike,
                            'contracts': contracts,
                            'collateral': total_collateral,
                            'credit_received': total_credit,
                        })

        # Track equity curve
        total_equity = balance + locked
        equity_curve.append({
            'date': pd.Timestamp(ts, unit='s'),
            'balance': balance,
            'locked': locked,
            'equity': total_equity,
            'open_positions': len(open_positions),
        })

        peak_balance = max(peak_balance, total_equity)
        dd = total_equity - peak_balance
        max_drawdown = min(max_drawdown, dd)

        i += 1

    if not trade_log:
        return None

    trades_df = pd.DataFrame(trade_log)
    equity_df = pd.DataFrame(equity_curve)

    final_equity = equity_df['equity'].iloc[-1]

    return {
        'ticker': ticker,
        'trades': trades_df,
        'equity': equity_df,
        'starting_capital': starting_capital,
        'final_equity': final_equity,
        'total_return': (final_equity - starting_capital) / starting_capital,
        'total_pnl': final_equity - starting_capital,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown / peak_balance if peak_balance > 0 else 0,
        'total_collateral_deployed': total_collateral_deployed,
        'num_trades': len(trades_df),
        'win_rate': trades_df['winner'].mean(),
        'years': (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days / 365,
    }


W = 110

# =====================================================
# RUN SIMULATIONS
# =====================================================
print("=" * W)
print(f"{'ACCOUNT SIMULATION: STRATEGY C WITH REAL POSITION SIZING':^{W}}")
print("=" * W)

for starting_cap in [5000, 10000, 25000]:
    print(f"\n{'=' * W}")
    print(f"  STARTING CAPITAL: ${starting_cap:,}")
    print(f"  Position sizing: 20% of buying power per trade, max 3 concurrent")
    print(f"{'=' * W}")

    results = []
    for ticker in TICKERS:
        sim = simulate_account(ticker, starting_capital=starting_cap)
        if sim:
            results.append(sim)

    print(f"\n  {'Ticker':<6} {'Trades':>6} {'WR':>6} {'Final Equity':>13} {'Total Return':>13} {'CAGR':>7} {'Max DD':>9} {'DD%':>6} {'Collateral':>12}")
    print(f"  {'-'*85}")

    for r in sorted(results, key=lambda x: x['total_return'], reverse=True):
        cagr = (r['final_equity'] / r['starting_capital']) ** (1 / r['years']) - 1 if r['years'] > 0 else 0
        print(f"  {r['ticker']:<6} {r['num_trades']:>6} {r['win_rate']:>5.1%} "
              f"${r['final_equity']:>12,.0f} {r['total_return']:>+12.1%} "
              f"{cagr:>6.1%} ${r['max_drawdown']:>+8.0f} {r['max_drawdown_pct']:>5.1%} "
              f"${r['total_collateral_deployed']:>11,.0f}")

    # Summary
    avg_return = np.mean([r['total_return'] for r in results])
    avg_cagr = np.mean([(r['final_equity'] / r['starting_capital']) ** (1 / r['years']) - 1 for r in results])
    avg_dd = np.mean([r['max_drawdown_pct'] for r in results])
    total_final = sum(r['final_equity'] for r in results)
    total_start = starting_cap * len(results)
    portfolio_return = (total_final - total_start) / total_start

    print(f"\n  Portfolio (equal-weight all {len(results)}):")
    print(f"    Total invested:  ${total_start:>12,}")
    print(f"    Final value:     ${total_final:>12,.0f}")
    print(f"    Portfolio return: {portfolio_return:>+12.1%}")
    print(f"    Avg CAGR:        {avg_cagr:>12.1%}")
    print(f"    Avg max DD%:     {avg_dd:>12.1%}")

# =====================================================
# DETAILED YEARLY EQUITY FOR $10K on best ticker
# =====================================================
print(f"\n{'=' * W}")
print(f"  DETAILED: $10,000 ACCOUNT EQUITY OVER TIME")
print(f"{'=' * W}")

for ticker in ['QQQ', 'AVGO', 'NVDA']:
    sim = simulate_account(ticker, starting_capital=10000)
    if not sim:
        continue

    eq = sim['equity']
    trades = sim['trades']
    print(f"\n  {ticker}: ${sim['starting_capital']:,} -> ${sim['final_equity']:,.0f} ({sim['total_return']:+.1%})")

    # Yearly snapshots
    print(f"  {'Year':<6} {'Equity':>10} {'YTD Return':>11} {'Trades':>7} {'WR':>6} {'Avg Contracts':>14}")
    print(f"  {'-'*60}")
    for year in range(2016, 2027):
        yr_eq = eq[eq['date'].dt.year == year]
        yr_trades = trades[trades['date'].dt.year == year]
        if len(yr_eq) == 0:
            continue
        start_eq = yr_eq['equity'].iloc[0]
        end_eq = yr_eq['equity'].iloc[-1]
        ytd = (end_eq - start_eq) / start_eq if start_eq > 0 else 0
        avg_contracts = yr_trades['contracts'].mean() if len(yr_trades) > 0 else 0
        wr = yr_trades['winner'].mean() if len(yr_trades) > 0 else 0
        print(f"  {year:<6} ${end_eq:>9,.0f} {ytd:>+10.1%} {len(yr_trades):>7} {wr:>5.0%} {avg_contracts:>14.1f}")

    # Trade sizing over time
    if len(trades) > 0:
        first5 = trades.head(5)
        last5 = trades.tail(5)
        print(f"\n  First 5 trades: avg {first5['contracts'].mean():.1f} contracts, avg collateral ${first5['collateral'].mean():,.0f}")
        print(f"  Last 5 trades:  avg {last5['contracts'].mean():.1f} contracts, avg collateral ${last5['collateral'].mean():,.0f}")
