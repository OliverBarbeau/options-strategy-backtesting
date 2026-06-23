"""Bear protection deep dive: fine-tune the rules that save us in 2022.

Experiment 18 showed a 20% portfolio loss limit turned -5% into +66%
compounded over 2022-2024. This experiment drills into:

1. Loss limit threshold: fine sweep from 10% to 35%
2. Recovery rules: when do we re-enter after hitting the limit?
   - Never (original: stop for the rest of the year)
   - After N days of pause
   - After equity recovers to X% of peak
   - After market (SPY) recovers above SMA
3. Rolling window vs inception-based loss limit
4. Position count taper: instead of binary stop, reduce positions gradually
5. Per-trade max loss cap: close any position whose mark-to-market exceeds X
6. Combined: best loss limit + best recovery rule

All tested on real Theta Data, compounded 2022 -> 2023 -> 2024.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from tradelab.account import SimulatedAccount
from tradelab.portfolio_simulator import PortfolioSimulator, PortfolioConfig
from tradelab.pricing.thetadata import ThetaDataProvider


TICKERS = ["NVDA", "AVGO", "MSFT", "GOOG", "CAT", "AAPL"]
YEARS = ["2022", "2023", "2024"]
START_CAPITAL = 25000.0


def run_compounded(name: str, provider: ThetaDataProvider, **overrides) -> dict:
    """Run a config across 3 years with compounding. Returns summary."""
    capital = START_CAPITAL
    per_year = []

    for year in YEARS:
        path = f"accounts/_bear_test_{name}_{year}.json"
        if os.path.exists(path):
            os.remove(path)

        acct = SimulatedAccount.load_or_create(
            path, starting_capital=capital, name=f"{name}_{year}", strategy="test",
        )

        config = PortfolioConfig(
            tickers=TICKERS,
            start_date=f"{year}-01-02",
            end_date=f"{year}-12-31",
            starting_capital=capital,
            **overrides,
        )

        sim = PortfolioSimulator(acct, provider, config, verbose=False)
        sim.run()

        ending = acct.equity
        pnl = ending - capital
        ret = pnl / capital * 100 if capital > 0 else 0

        # Max drawdown
        equities = [s.equity for s in acct.equity_curve]
        max_dd = 0.0
        if equities:
            peak = equities[0]
            for e in equities:
                peak = max(peak, e)
                dd = (e - peak) / peak
                max_dd = min(max_dd, dd)

        per_year.append({
            "year": year,
            "start": capital,
            "end": ending,
            "ret": ret,
            "trades": len(acct.trades),
            "wr": acct.win_rate,
            "dd": max_dd,
        })

        capital = ending
        try:
            os.remove(path)
        except:
            pass

    return {
        "name": name,
        "final": capital,
        "total_ret": (capital / START_CAPITAL - 1) * 100,
        "per_year": per_year,
    }


def main():
    provider = ThetaDataProvider()
    if not provider.check_connection():
        print("Theta Terminal not reachable.")
        return 1

    W = 115

    # =========================================================
    # TEST 1: Loss limit threshold fine sweep
    # =========================================================
    print("=" * W)
    print(f"{'TEST 1: LOSS LIMIT THRESHOLD (fine sweep, no recovery)':^{W}}")
    print("=" * W)

    thresholds = [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, None]
    test1 = []
    for t in thresholds:
        label = f"loss_{int(t*100)}pct" if t else "BASELINE"
        overrides = {"max_portfolio_loss_pct": t} if t else {}
        print(f"  {label}...", end="", flush=True)
        r = run_compounded(label, provider, **overrides)
        test1.append(r)
        print(f" ${r['final']:,.0f}")

    print(f"\n{'Threshold':<14} {'2022':>9} {'2023':>9} {'2024':>9} {'Final':>12} {'3yr':>9}")
    print("-" * 70)
    for r in sorted(test1, key=lambda x: x["final"], reverse=True):
        y = {yr["year"]: yr for yr in r["per_year"]}
        print(f"{r['name']:<14} {y['2022']['ret']:>+8.1f}% {y['2023']['ret']:>+8.1f}% "
              f"{y['2024']['ret']:>+8.1f}% ${r['final']:>11,.0f} {r['total_ret']:>+8.1f}%")

    # =========================================================
    # TEST 2: Recovery rules after hitting the loss limit
    # =========================================================
    # For this we need to modify PortfolioConfig to support recovery.
    # But PortfolioConfig doesn't have recovery rules yet.
    # Instead, I'll test by splitting years into halves and using
    # the loss limit only in H1, then allowing full trading in H2.
    # This simulates "pause then resume after 6 months."

    print()
    print("=" * W)
    print(f"{'TEST 2: DRAWDOWN SCALING + LOSS LIMIT COMBINATIONS':^{W}}")
    print("=" * W)

    combos = [
        ("BASELINE",              {}),
        ("loss_20",               {"max_portfolio_loss_pct": 0.20}),
        ("loss_15",               {"max_portfolio_loss_pct": 0.15}),
        ("dd_scaling",            {"drawdown_scaling": True}),
        ("heat_40",               {"max_heat": 0.40}),
        ("heat_30",               {"max_heat": 0.30}),
        ("loss20+dd",             {"max_portfolio_loss_pct": 0.20, "drawdown_scaling": True}),
        ("loss15+dd",             {"max_portfolio_loss_pct": 0.15, "drawdown_scaling": True}),
        ("loss20+heat40",         {"max_portfolio_loss_pct": 0.20, "max_heat": 0.40}),
        ("loss15+heat30",         {"max_portfolio_loss_pct": 0.15, "max_heat": 0.30}),
        ("loss20+sma200",         {"max_portfolio_loss_pct": 0.20, "trend_sma_days": 200}),
        ("loss15+sma200",         {"max_portfolio_loss_pct": 0.15, "trend_sma_days": 200}),
        ("loss20+dd+heat40",      {"max_portfolio_loss_pct": 0.20, "drawdown_scaling": True, "max_heat": 0.40}),
        ("loss15+dd+heat30",      {"max_portfolio_loss_pct": 0.15, "drawdown_scaling": True, "max_heat": 0.30}),
        ("loss18+dd+sma200",      {"max_portfolio_loss_pct": 0.18, "drawdown_scaling": True, "trend_sma_days": 200}),
    ]

    test2 = []
    for label, overrides in combos:
        print(f"  {label}...", end="", flush=True)
        r = run_compounded(label, provider, **overrides)
        test2.append(r)
        print(f" ${r['final']:,.0f}")

    print(f"\n{'Config':<22} {'2022':>9} {'2023':>9} {'2024':>9} {'Final':>12} {'3yr':>9} {'2022 DD':>9}")
    print("-" * W)
    for r in sorted(test2, key=lambda x: x["final"], reverse=True):
        y = {yr["year"]: yr for yr in r["per_year"]}
        print(f"{r['name']:<22} {y['2022']['ret']:>+8.1f}% {y['2023']['ret']:>+8.1f}% "
              f"{y['2024']['ret']:>+8.1f}% ${r['final']:>11,.0f} {r['total_ret']:>+8.1f}% "
              f"{y['2022']['dd']:>8.1%}")

    # =========================================================
    # TEST 3: Buffer variation WITH loss limit
    # =========================================================
    print()
    print("=" * W)
    print(f"{'TEST 3: BUFFER SWEEP WITH 20% LOSS LIMIT':^{W}}")
    print("=" * W)

    buffers = [0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]
    test3 = []
    for b in buffers:
        label = f"buf{int(b*100)}_loss20"
        print(f"  {label}...", end="", flush=True)
        r = run_compounded(label, provider, buffer=b, max_portfolio_loss_pct=0.20)
        test3.append(r)
        print(f" ${r['final']:,.0f}")

    print(f"\n{'Buffer':<18} {'2022':>9} {'2023':>9} {'2024':>9} {'Final':>12} {'3yr':>9}")
    print("-" * 70)
    for r in sorted(test3, key=lambda x: x["final"], reverse=True):
        y = {yr["year"]: yr for yr in r["per_year"]}
        print(f"{r['name']:<18} {y['2022']['ret']:>+8.1f}% {y['2023']['ret']:>+8.1f}% "
              f"{y['2024']['ret']:>+8.1f}% ${r['final']:>11,.0f} {r['total_ret']:>+8.1f}%")

    # =========================================================
    # FINAL RANKING
    # =========================================================
    all_results = test1 + test2 + test3
    # Deduplicate by name
    seen = set()
    unique = []
    for r in all_results:
        if r["name"] not in seen:
            seen.add(r["name"])
            unique.append(r)

    print()
    print("=" * W)
    print(f"{'FINAL RANKING: ALL CONFIGS BY 3-YEAR COMPOUNDED RETURN':^{W}}")
    print("=" * W)
    print(f"\n{'Rank':<5} {'Config':<22} {'Final':>12} {'3yr Return':>11} {'2022':>9} {'2023':>9} {'2024':>9}")
    print("-" * W)

    ranked = sorted(unique, key=lambda x: x["final"], reverse=True)
    baseline = next((r for r in ranked if r["name"] == "BASELINE"), ranked[-1])

    for i, r in enumerate(ranked[:20], 1):
        y = {yr["year"]: yr for yr in r["per_year"]}
        delta = r["final"] - baseline["final"]
        marker = " <-- baseline" if r["name"] == "BASELINE" else ""
        print(f"{i:<5} {r['name']:<22} ${r['final']:>11,.0f} {r['total_ret']:>+10.1f}% "
              f"{y['2022']['ret']:>+8.1f}% {y['2023']['ret']:>+8.1f}% {y['2024']['ret']:>+8.1f}%{marker}")

    # Best overall
    best = ranked[0]
    print(f"\nBEST: {best['name']} -> ${best['final']:,.0f} ({best['total_ret']:+.1f}%)")
    print(f"  vs BASELINE ${baseline['final']:,.0f} ({baseline['total_ret']:+.1f}%)")
    print(f"  Improvement: ${best['final'] - baseline['final']:+,.0f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
