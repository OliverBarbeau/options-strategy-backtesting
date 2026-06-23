"""Bootstrap resampling for strategy confidence intervals.

Takes the actual trade outcomes from real Theta Data backtests and resamples
them to generate thousands of possible equity paths. This answers:

- "How likely is it that the 7%+20% config just got lucky?"
- "What's the 5th percentile worst-case 3-year return?"
- "How often does the strategy lose money over 3 years?"

The method is well-established in quantitative finance: we don't fabricate
new market data, we just shuffle the order of real trades to see how
sensitive results are to specific sequencing.

Two resampling modes:
1. **Trade-level bootstrap**: resample individual trades with replacement.
   Assumes trades are independent (approximately true for our strategy
   since each trade is a separate 16-day window).
2. **Block bootstrap**: resample blocks of N consecutive trades to preserve
   any serial correlation (e.g., loss streaks from sustained bear markets).
   More conservative — preserves the clustering effect.

Usage::

    from tradelab.bootstrap import BootstrapAnalyzer

    # Load real trade outcomes from a completed simulation
    analyzer = BootstrapAnalyzer.from_account("accounts/portfolio_theta_2024.json")

    # Or combine trades from multiple years
    analyzer = BootstrapAnalyzer.from_accounts([
        "accounts/portfolio_theta_2022.json",
        "accounts/portfolio_theta_2023.json",
        "accounts/portfolio_theta_2024.json",
    ])

    # Run 10,000 bootstrap paths
    result = analyzer.run(n_paths=10000, path_length=250, starting_capital=25000)
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tradelab.account import SimulatedAccount


@dataclass
class BootstrapResult:
    """Results of a bootstrap resampling analysis."""
    n_paths: int
    path_length: int  # trades per path
    starting_capital: float
    source_trades: int  # number of real trades in the sample pool

    # Distribution of final equity across all paths
    final_equities: np.ndarray = field(repr=False)

    # Distribution of max drawdown across all paths
    max_drawdowns: np.ndarray = field(repr=False)

    # Distribution of win rates across all paths
    win_rates: np.ndarray = field(repr=False)

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.final_equities) / self.starting_capital - 1) * 100

    @property
    def median_return(self) -> float:
        return float(np.median(self.final_equities) / self.starting_capital - 1) * 100

    @property
    def pct_profitable(self) -> float:
        return float(np.mean(self.final_equities > self.starting_capital)) * 100

    @property
    def pct_5(self) -> float:
        """5th percentile return (worst 5% of paths)."""
        return float(np.percentile(self.final_equities, 5) / self.starting_capital - 1) * 100

    @property
    def pct_25(self) -> float:
        return float(np.percentile(self.final_equities, 25) / self.starting_capital - 1) * 100

    @property
    def pct_75(self) -> float:
        return float(np.percentile(self.final_equities, 75) / self.starting_capital - 1) * 100

    @property
    def pct_95(self) -> float:
        """95th percentile return (best 5% of paths)."""
        return float(np.percentile(self.final_equities, 95) / self.starting_capital - 1) * 100

    @property
    def mean_max_dd(self) -> float:
        return float(np.mean(self.max_drawdowns)) * 100

    @property
    def worst_max_dd(self) -> float:
        return float(np.min(self.max_drawdowns)) * 100

    def summary(self) -> str:
        return (
            f"Bootstrap Analysis ({self.n_paths:,} paths, {self.path_length} trades each)\n"
            f"  Source: {self.source_trades} real trades\n"
            f"  Starting capital: ${self.starting_capital:,.0f}\n"
            f"\n"
            f"  Return distribution:\n"
            f"    Mean:     {self.mean_return:+.1f}%\n"
            f"    Median:   {self.median_return:+.1f}%\n"
            f"    5th pct:  {self.pct_5:+.1f}%  (worst 5% of paths)\n"
            f"    25th pct: {self.pct_25:+.1f}%\n"
            f"    75th pct: {self.pct_75:+.1f}%\n"
            f"    95th pct: {self.pct_95:+.1f}%  (best 5% of paths)\n"
            f"\n"
            f"  Profitable paths: {self.pct_profitable:.1f}%\n"
            f"\n"
            f"  Drawdown distribution:\n"
            f"    Mean max DD:  {self.mean_max_dd:.1f}%\n"
            f"    Worst max DD: {self.worst_max_dd:.1f}%\n"
            f"    Median DD:    {float(np.median(self.max_drawdowns)) * 100:.1f}%"
        )


class BootstrapAnalyzer:
    """Bootstrap resampling engine for strategy confidence intervals.

    Args:
        trades: List of (pnl, collateral) tuples from real backtests.
            Each represents one completed trade's dollar P/L and the
            collateral that was locked for that trade.
        seed: Random seed for reproducibility.
    """

    def __init__(self, trades: list[tuple[float, float]], seed: int = 42):
        self.trades = trades
        self.rng = np.random.default_rng(seed)

    @classmethod
    def from_account(cls, path: str, **kwargs) -> "BootstrapAnalyzer":
        """Load trades from a single SimulatedAccount JSON file."""
        acct = SimulatedAccount.load(path)
        trades = [(t.pnl, t.collateral) for t in acct.trades]
        return cls(trades, **kwargs)

    @classmethod
    def from_accounts(cls, paths: list[str], **kwargs) -> "BootstrapAnalyzer":
        """Load and combine trades from multiple account files."""
        all_trades = []
        for path in paths:
            try:
                acct = SimulatedAccount.load(path)
                all_trades.extend((t.pnl, t.collateral) for t in acct.trades)
            except Exception as e:
                print(f"Warning: failed to load {path}: {e}")
        return cls(all_trades, **kwargs)

    @classmethod
    def from_trade_list(cls, pnls: list[float], collaterals: list[float] | None = None, **kwargs):
        """Create from raw P/L and collateral lists."""
        if collaterals is None:
            collaterals = [500.0] * len(pnls)  # default estimate
        return cls(list(zip(pnls, collaterals)), **kwargs)

    def run(
        self,
        n_paths: int = 10000,
        path_length: int | None = None,
        starting_capital: float = 25000.0,
        max_portfolio_loss_pct: float | None = None,
        block_size: int = 1,
    ) -> BootstrapResult:
        """Run bootstrap resampling.

        Args:
            n_paths: Number of simulated equity paths.
            path_length: Trades per path. None = same as source.
            starting_capital: Starting equity for each path.
            max_portfolio_loss_pct: If set, stop trading in a path when
                cumulative loss exceeds this fraction. Simulates the
                loss-limit rule within the bootstrap.
            block_size: For block bootstrap, resample in blocks of this
                size to preserve serial correlation. 1 = standard iid.
        """
        if not self.trades:
            raise ValueError("No trades to resample")

        pnls = np.array([t[0] for t in self.trades])
        collaterals = np.array([t[1] for t in self.trades])
        n_source = len(pnls)

        if path_length is None:
            path_length = n_source

        final_equities = np.zeros(n_paths)
        max_drawdowns = np.zeros(n_paths)
        win_rates = np.zeros(n_paths)

        for p in range(n_paths):
            # Resample trade indices
            if block_size <= 1:
                # Standard iid bootstrap
                indices = self.rng.integers(0, n_source, size=path_length)
            else:
                # Block bootstrap: sample starting points, take consecutive blocks
                n_blocks = (path_length + block_size - 1) // block_size
                block_starts = self.rng.integers(0, max(1, n_source - block_size + 1), size=n_blocks)
                indices = np.concatenate([
                    np.arange(s, min(s + block_size, n_source))
                    for s in block_starts
                ])[:path_length]

            # Simulate equity path
            equity = starting_capital
            peak = equity
            max_dd = 0.0
            wins = 0
            trades_taken = 0
            stopped = False

            for idx in indices:
                if stopped:
                    break

                trade_pnl = pnls[idx]
                trade_col = collaterals[idx]

                # Check if we can afford this trade
                if trade_col > equity * 0.5:  # rough affordability check
                    continue

                equity += trade_pnl
                trades_taken += 1

                if trade_pnl > 0:
                    wins += 1

                peak = max(peak, equity)
                dd = (equity - peak) / peak if peak > 0 else 0
                max_dd = min(max_dd, dd)

                # Loss limit check
                if max_portfolio_loss_pct is not None:
                    loss_pct = (equity - starting_capital) / starting_capital
                    if loss_pct <= -max_portfolio_loss_pct:
                        stopped = True

            final_equities[p] = equity
            max_drawdowns[p] = max_dd
            win_rates[p] = wins / trades_taken if trades_taken > 0 else 0

        return BootstrapResult(
            n_paths=n_paths,
            path_length=path_length,
            starting_capital=starting_capital,
            source_trades=n_source,
            final_equities=final_equities,
            max_drawdowns=max_drawdowns,
            win_rates=win_rates,
        )

    def compare_configs(
        self,
        configs: dict[str, dict],
        n_paths: int = 10000,
        path_length: int | None = None,
        starting_capital: float = 25000.0,
    ) -> dict[str, BootstrapResult]:
        """Run bootstrap with multiple config variants and compare.

        Args:
            configs: Dict mapping config name to kwargs for run().
                Example: {"baseline": {}, "loss_20": {"max_portfolio_loss_pct": 0.20}}
        """
        results = {}
        for name, kwargs in configs.items():
            results[name] = self.run(
                n_paths=n_paths,
                path_length=path_length,
                starting_capital=starting_capital,
                **kwargs,
            )
        return results
