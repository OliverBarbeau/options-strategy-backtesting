"""Parameter sweep: buffer, DTE, spread width, vol filter across NVDA/AAPL/SPY."""

import warnings
warnings.filterwarnings("ignore")

from tradelab.pipeline import DataPipeline
from tradelab.options import historical_volatility, put_credit_spread_price
import pandas as pd
import numpy as np

pipe = DataPipeline()

# Load data once
data = {}
for t in ['NVDA', 'AAPL', 'SPY']:
    data[t] = pipe.fetch_stock(t, start='2020-01-01', end='2026-04-01')
    print(f"Loaded {t}: {len(data[t])} rows")


def run_sweep(ticker, buffer, dte, spread_pct, vol_cap):
    df = data[ticker]
    close_prices = df['close'].values
    timestamps = df.index.values
    offset = max(1, int(dte * 21 / 30))
    vol = historical_volatility(df['close'], window=30)

    results = []
    i = 30  # skip vol window
    while i < len(df) - offset:
        price = close_prices[i]
        vol_val = vol.iloc[i]
        if np.isnan(vol_val) or vol_val <= 0:
            i += 1
            continue
        if vol_cap is not None and vol_val > vol_cap:
            i += offset  # skip this window entirely
            continue

        spread_width = price * spread_pct
        short_strike = price * (1 - buffer)
        long_strike = short_strike - spread_width
        if long_strike <= 0:
            i += offset
            continue

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
            'pnl': pnl,
            'credit': net_credit,
            'max_loss': max_loss,
            'winner': winner,
            'sigma': vol_val,
        })
        i += offset

    if not results or len(results) < 5:
        return None
    return pd.DataFrame(results)


# =====================================================
# SWEEP
# =====================================================
buffers = [0.05, 0.07, 0.10]
dtes = [7, 14, 21, 30]
spread_pcts = [0.02, 0.03, 0.05]
vol_caps = [None, 0.40, 0.30]
tickers = ['NVDA', 'AAPL', 'SPY']

all_results = []
total = len(tickers) * len(buffers) * len(dtes) * len(spread_pcts) * len(vol_caps)
count = 0

for ticker in tickers:
    for buffer in buffers:
        for dte in dtes:
            for spread_pct in spread_pcts:
                for vol_cap in vol_caps:
                    count += 1
                    trades = run_sweep(ticker, buffer, dte, spread_pct, vol_cap)
                    if trades is None:
                        continue

                    wins = trades['winner'].sum()
                    losses = len(trades) - wins
                    win_rate = wins / len(trades)
                    total_pnl = trades['pnl'].sum()
                    avg_pnl = trades['pnl'].mean()
                    avg_credit = trades[trades['winner']]['credit'].mean() if wins > 0 else 0
                    avg_loss = abs(trades[~trades['winner']]['pnl'].mean()) if losses > 0 else 0
                    breakeven = avg_loss / (avg_credit + avg_loss) if (avg_credit + avg_loss) > 0 else 1

                    all_results.append({
                        'ticker': ticker,
                        'buffer': buffer,
                        'dte': dte,
                        'spread_pct': spread_pct,
                        'vol_cap': vol_cap if vol_cap else 'none',
                        'trades': len(trades),
                        'win_rate': win_rate,
                        'total_pnl': total_pnl,
                        'avg_pnl': avg_pnl,
                        'avg_credit': avg_credit,
                        'avg_loss': avg_loss,
                        'breakeven_wr': breakeven,
                        'edge': win_rate - breakeven,
                    })

sweep = pd.DataFrame(all_results)
print(f"\nTotal combos tested: {len(sweep)} / {total}")
print()

# =====================================================
# TOP 15 OVERALL
# =====================================================
W = 115
print("=" * W)
print(f"{'TOP 15 PARAMETER COMBOS BY TOTAL P/L':^{W}}")
print("=" * W)
top = sweep.nlargest(15, 'total_pnl')
print(f"{'Tkr':<5} {'Buf':>4} {'DTE':>4} {'Sprd%':>5} {'VolCap':>7} {'Trades':>6} {'WinRate':>8} {'BkevnWR':>8} {'Edge':>7} {'AvgCr':>8} {'AvgLoss':>8} {'P/L':>10} {'$/Trade':>9}")
print("-" * W)
for _, r in top.iterrows():
    vc = f"{r['vol_cap']:.0%}" if r['vol_cap'] != 'none' else 'none'
    print(f"{r['ticker']:<5} {r['buffer']:>4.0%} {r['dte']:>4.0f} {r['spread_pct']:>5.0%}   {vc:>5} {r['trades']:>6.0f} {r['win_rate']:>7.1%} {r['breakeven_wr']:>7.1%} {r['edge']:>+6.1%} {r['avg_credit']:>8.1f} {r['avg_loss']:>8.1f} {r['total_pnl']:>+10.0f} {r['avg_pnl']:>+9.2f}")

# =====================================================
# TOP 5 PER TICKER
# =====================================================
for ticker in tickers:
    print()
    print("=" * 100)
    print(f" {ticker} - TOP 5 BY TOTAL P/L")
    print("=" * 100)
    t5 = sweep[sweep['ticker'] == ticker].nlargest(5, 'total_pnl')
    print(f"{'Buf':>4} {'DTE':>4} {'Sprd%':>5} {'VolCap':>7} {'Trades':>6} {'WinRate':>8} {'BkevnWR':>8} {'Edge':>7} {'AvgCr':>8} {'AvgLoss':>8} {'P/L':>10} {'$/Trade':>9}")
    print("-" * 100)
    for _, r in t5.iterrows():
        vc = f"{r['vol_cap']:.0%}" if r['vol_cap'] != 'none' else 'none'
        print(f"{r['buffer']:>4.0%} {r['dte']:>4.0f} {r['spread_pct']:>5.0%}   {vc:>5} {r['trades']:>6.0f} {r['win_rate']:>7.1%} {r['breakeven_wr']:>7.1%} {r['edge']:>+6.1%} {r['avg_credit']:>8.1f} {r['avg_loss']:>8.1f} {r['total_pnl']:>+10.0f} {r['avg_pnl']:>+9.2f}")

# =====================================================
# WORST 5 PER TICKER
# =====================================================
for ticker in tickers:
    print()
    print(f" {ticker} - WORST 5")
    t5 = sweep[sweep['ticker'] == ticker].nsmallest(5, 'total_pnl')
    print(f"{'Buf':>4} {'DTE':>4} {'Sprd%':>5} {'VolCap':>7} {'Trades':>6} {'WinRate':>8} {'Edge':>7} {'P/L':>10}")
    for _, r in t5.iterrows():
        vc = f"{r['vol_cap']:.0%}" if r['vol_cap'] != 'none' else 'none'
        print(f"{r['buffer']:>4.0%} {r['dte']:>4.0f} {r['spread_pct']:>5.0%}   {vc:>5} {r['trades']:>6.0f} {r['win_rate']:>7.1%} {r['edge']:>+6.1%} {r['total_pnl']:>+10.0f}")

# =====================================================
# DIMENSION ANALYSIS
# =====================================================
print()
print("=" * W)
print(f"{'WHICH LEVER MATTERS MOST? (avg P/L per trade)':^{W}}")
print("=" * W)

print("\n--- By Buffer ---")
for b in buffers:
    sub = sweep[sweep['buffer'] == b]
    prof = len(sub[sub['total_pnl'] > 0])
    print(f"  {b:.0%} buffer:  avg $/trade {sub['avg_pnl'].mean():>+8.2f}  avg WR {sub['win_rate'].mean():.1%}  edge {sub['edge'].mean():>+5.2%}  profitable {prof}/{len(sub)}")

print("\n--- By DTE ---")
for d in dtes:
    sub = sweep[sweep['dte'] == d]
    prof = len(sub[sub['total_pnl'] > 0])
    print(f"  {d:>2}d DTE:    avg $/trade {sub['avg_pnl'].mean():>+8.2f}  avg WR {sub['win_rate'].mean():.1%}  edge {sub['edge'].mean():>+5.2%}  profitable {prof}/{len(sub)}")

print("\n--- By Spread Width ---")
for s in spread_pcts:
    sub = sweep[sweep['spread_pct'] == s]
    prof = len(sub[sub['total_pnl'] > 0])
    print(f"  {s:.0%} width:  avg $/trade {sub['avg_pnl'].mean():>+8.2f}  avg WR {sub['win_rate'].mean():.1%}  edge {sub['edge'].mean():>+5.2%}  profitable {prof}/{len(sub)}")

print("\n--- By Vol Filter ---")
for v in vol_caps:
    sub = sweep[sweep['vol_cap'] == (v if v else 'none')]
    prof = len(sub[sub['total_pnl'] > 0])
    label = f"{v:.0%} cap" if v else "no cap"
    print(f"  {label:>8}:  avg $/trade {sub['avg_pnl'].mean():>+8.2f}  avg WR {sub['win_rate'].mean():.1%}  edge {sub['edge'].mean():>+5.2%}  profitable {prof}/{len(sub)}")

print("\n--- By Ticker ---")
for t in tickers:
    sub = sweep[sweep['ticker'] == t]
    prof = len(sub[sub['total_pnl'] > 0])
    print(f"  {t:>4}:      avg $/trade {sub['avg_pnl'].mean():>+8.2f}  avg WR {sub['win_rate'].mean():.1%}  edge {sub['edge'].mean():>+5.2%}  profitable {prof}/{len(sub)}")

# =====================================================
# SUMMARY
# =====================================================
profitable = sweep[sweep['total_pnl'] > 0]
print()
print("=" * W)
print(f"SUMMARY: {len(profitable)}/{len(sweep)} combos profitable ({len(profitable)/len(sweep):.0%})")
print(f"  NVDA: {len(profitable[profitable['ticker']=='NVDA'])}/{len(sweep[sweep['ticker']=='NVDA'])}")
print(f"  AAPL: {len(profitable[profitable['ticker']=='AAPL'])}/{len(sweep[sweep['ticker']=='AAPL'])}")
print(f"  SPY:  {len(profitable[profitable['ticker']=='SPY'])}/{len(sweep[sweep['ticker']=='SPY'])}")

if len(profitable) > 0:
    print()
    print("Common traits of profitable combos:")
    print(f"  Avg buffer:     {profitable['buffer'].mean():.1%}  (vs {sweep['buffer'].mean():.1%} overall)")
    print(f"  Avg DTE:        {profitable['dte'].mean():.0f}d  (vs {sweep['dte'].mean():.0f}d overall)")
    print(f"  Avg spread:     {profitable['spread_pct'].mean():.1%}  (vs {sweep['spread_pct'].mean():.1%} overall)")
    vc_none = len(profitable[profitable['vol_cap'] == 'none'])
    vc_40 = len(profitable[profitable['vol_cap'] == 0.40])
    vc_30 = len(profitable[profitable['vol_cap'] == 0.30])
    print(f"  Vol cap:        none={vc_none}  40%={vc_40}  30%={vc_30}")
print("=" * W)
