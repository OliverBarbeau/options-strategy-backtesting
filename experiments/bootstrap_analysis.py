"""Bootstrap resampling: confidence intervals for our strategy.

Uses real trade outcomes from Theta Data portfolio simulations (2022-2024)
to generate 10,000 possible equity paths and answer:

1. What's the probability the strategy is profitable over 1/2/3 years?
2. What's the 5th percentile worst case?
3. How much does the 20% loss limit improve the distribution?
4. Is 7% buffer + loss limit genuinely better, or did it get lucky?
5. Block bootstrap: does preserving loss streaks change the picture?
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from tradelab.bootstrap import BootstrapAnalyzer


def main():
    W = 100

    # Load real trade outcomes from all 3 years
    account_files = [
        "accounts/portfolio_theta_2022.json",
        "accounts/portfolio_theta_2023.json",
        "accounts/portfolio_theta_2024.json",
    ]

    print("Loading real trade data...")
    analyzer = BootstrapAnalyzer.from_accounts(account_files)
    print(f"  Loaded {len(analyzer.trades)} trades from 3 years of real Theta Data")

    pnls = [t[0] for t in analyzer.trades]
    winners = sum(1 for p in pnls if p > 0)
    print(f"  Win rate: {winners/len(pnls):.1%}")
    print(f"  Avg winner: ${np.mean([p for p in pnls if p > 0]):+.2f}")
    print(f"  Avg loser: ${np.mean([p for p in pnls if p <= 0]):+.2f}")
    print()

    N_PATHS = 10000

    # =========================================================
    # TEST 1: Baseline distribution (no loss limit)
    # =========================================================
    print("=" * W)
    print(f"{'TEST 1: BASELINE BOOTSTRAP (10,000 paths, no loss limit)':^{W}}")
    print("=" * W)

    # 1-year paths (same number of trades as one year ~80-90)
    for label, path_len in [("1 year (~90 trades)", 90), ("2 years (~180)", 180), ("3 years (~270)", 270)]:
        result = analyzer.run(n_paths=N_PATHS, path_length=path_len, starting_capital=25000)
        print(f"\n  {label}:")
        print(f"    Mean return:    {result.mean_return:+.1f}%")
        print(f"    Median return:  {result.median_return:+.1f}%")
        print(f"    5th pct:        {result.pct_5:+.1f}%  (worst 5%)")
        print(f"    95th pct:       {result.pct_95:+.1f}%  (best 5%)")
        print(f"    Profitable:     {result.pct_profitable:.1f}%")
        print(f"    Mean max DD:    {result.mean_max_dd:.1f}%")
        print(f"    Worst max DD:   {result.worst_max_dd:.1f}%")

    # =========================================================
    # TEST 2: With 20% loss limit
    # =========================================================
    print()
    print("=" * W)
    print(f"{'TEST 2: WITH 20% LOSS LIMIT':^{W}}")
    print("=" * W)

    for label, path_len in [("1 year (~90 trades)", 90), ("2 years (~180)", 180), ("3 years (~270)", 270)]:
        result = analyzer.run(
            n_paths=N_PATHS, path_length=path_len, starting_capital=25000,
            max_portfolio_loss_pct=0.20,
        )
        print(f"\n  {label}:")
        print(f"    Mean return:    {result.mean_return:+.1f}%")
        print(f"    Median return:  {result.median_return:+.1f}%")
        print(f"    5th pct:        {result.pct_5:+.1f}%  (worst 5%)")
        print(f"    95th pct:       {result.pct_95:+.1f}%  (best 5%)")
        print(f"    Profitable:     {result.pct_profitable:.1f}%")
        print(f"    Mean max DD:    {result.mean_max_dd:.1f}%")
        print(f"    Worst max DD:   {result.worst_max_dd:.1f}%")

    # =========================================================
    # TEST 3: Head-to-head comparison
    # =========================================================
    print()
    print("=" * W)
    print(f"{'TEST 3: HEAD-TO-HEAD (3-year paths)':^{W}}")
    print("=" * W)

    configs = {
        "Baseline (no protection)": {},
        "Loss limit 15%": {"max_portfolio_loss_pct": 0.15},
        "Loss limit 20%": {"max_portfolio_loss_pct": 0.20},
        "Loss limit 25%": {"max_portfolio_loss_pct": 0.25},
    }

    results = analyzer.compare_configs(configs, n_paths=N_PATHS, path_length=270)

    print(f"\n{'Config':<30} {'Mean':>8} {'Median':>8} {'5th':>8} {'95th':>8} {'Profit%':>8} {'MeanDD':>8}")
    print("-" * W)
    for name, r in sorted(results.items(), key=lambda x: x[1].median_return, reverse=True):
        print(f"{name:<30} {r.mean_return:>+7.1f}% {r.median_return:>+7.1f}% "
              f"{r.pct_5:>+7.1f}% {r.pct_95:>+7.1f}% {r.pct_profitable:>7.1f}% "
              f"{r.mean_max_dd:>7.1f}%")

    # =========================================================
    # TEST 4: Block bootstrap (preserve loss streaks)
    # =========================================================
    print()
    print("=" * W)
    print(f"{'TEST 4: BLOCK BOOTSTRAP (preserve serial correlation)':^{W}}")
    print("=" * W)
    print("  Blocks of 5 consecutive trades — preserves loss-streak patterns")

    for label, loss_limit in [("Baseline", None), ("Loss limit 20%", 0.20)]:
        result_iid = analyzer.run(
            n_paths=N_PATHS, path_length=270, starting_capital=25000,
            max_portfolio_loss_pct=loss_limit, block_size=1,
        )
        result_block = analyzer.run(
            n_paths=N_PATHS, path_length=270, starting_capital=25000,
            max_portfolio_loss_pct=loss_limit, block_size=5,
        )
        print(f"\n  {label}:")
        print(f"    {'':20} {'IID':>12} {'Block(5)':>12} {'Diff':>10}")
        print(f"    {'Mean return:':<20} {result_iid.mean_return:>+11.1f}% {result_block.mean_return:>+11.1f}% {result_block.mean_return - result_iid.mean_return:>+9.1f}%")
        print(f"    {'5th pct:':<20} {result_iid.pct_5:>+11.1f}% {result_block.pct_5:>+11.1f}% {result_block.pct_5 - result_iid.pct_5:>+9.1f}%")
        print(f"    {'Profitable:':<20} {result_iid.pct_profitable:>11.1f}% {result_block.pct_profitable:>11.1f}% {result_block.pct_profitable - result_iid.pct_profitable:>+9.1f}%")
        print(f"    {'Worst DD:':<20} {result_iid.worst_max_dd:>11.1f}% {result_block.worst_max_dd:>11.1f}% {result_block.worst_max_dd - result_iid.worst_max_dd:>+9.1f}%")

    # =========================================================
    # TEST 5: What fraction of paths lose more than X%?
    # =========================================================
    print()
    print("=" * W)
    print(f"{'TEST 5: TAIL RISK (3-year paths, 10K simulations)':^{W}}")
    print("=" * W)

    for label, loss_limit in [("Baseline", None), ("Loss limit 20%", 0.20)]:
        result = analyzer.run(
            n_paths=N_PATHS, path_length=270, starting_capital=25000,
            max_portfolio_loss_pct=loss_limit,
        )
        returns = (result.final_equities / 25000 - 1) * 100

        print(f"\n  {label}:")
        for threshold in [-50, -30, -20, -10, 0, 10, 25, 50, 100]:
            pct_below = np.mean(returns < threshold) * 100
            pct_above = np.mean(returns >= threshold) * 100
            if threshold <= 0:
                print(f"    Paths losing >{abs(threshold)}%:  {pct_below:>6.1f}%")
            else:
                print(f"    Paths gaining >{threshold}%: {pct_above:>6.1f}%")

    # =========================================================
    # SUMMARY
    # =========================================================
    print()
    print("=" * W)
    print(f"{'SUMMARY':^{W}}")
    print("=" * W)

    # Run the two key configs one more time for the summary
    baseline = analyzer.run(n_paths=N_PATHS, path_length=270, starting_capital=25000)
    protected = analyzer.run(n_paths=N_PATHS, path_length=270, starting_capital=25000, max_portfolio_loss_pct=0.20)

    print(f"""
  Based on {len(analyzer.trades)} real trades from Theta Data (2022-2024),
  resampled into {N_PATHS:,} possible 3-year paths:

  BASELINE (no protection):
    Median 3yr return: {baseline.median_return:+.1f}%
    Probability of profit: {baseline.pct_profitable:.0f}%
    5th percentile (worst case): {baseline.pct_5:+.1f}%
    95th percentile (best case): {baseline.pct_95:+.1f}%

  WITH 20% LOSS LIMIT:
    Median 3yr return: {protected.median_return:+.1f}%
    Probability of profit: {protected.pct_profitable:.0f}%
    5th percentile (worst case): {protected.pct_5:+.1f}%
    95th percentile (best case): {protected.pct_95:+.1f}%

  The loss limit:
    - Improves median return by {protected.median_return - baseline.median_return:+.1f} percentage points
    - Raises profit probability by {protected.pct_profitable - baseline.pct_profitable:+.1f} percentage points
    - Raises worst-case floor by {protected.pct_5 - baseline.pct_5:+.1f} percentage points
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
