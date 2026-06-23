"""A/B test framework for comparing strategy variants.

Runs multiple configurations of the PortfolioSimulator side-by-side and
reports statistical significance of differences. Primary use case:
comparing pullback-filtered entries vs always-enter baseline to validate
that the pullback filter adds genuine edge.

Usage::

    from tradelab.ab_test import run_comparison

    results = run_comparison(
        base_config=config,
        provider=provider,
        variants={"pullback": 0.03, "always_enter": 0.0},
    )
    print(results.summary())
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field

from tradelab.account import SimulatedAccount
from tradelab.portfolio_simulator import (
    PortfolioConfig,
    PortfolioReport,
    PortfolioSimulator,
    _wilson_ci,
)
from tradelab.pricing.thetadata import ThetaDataProvider


@dataclass
class VariantResult:
    """Results for a single A/B test variant."""
    name: str
    config: PortfolioConfig
    report: PortfolioReport


@dataclass
class ABTestResult:
    """Comparison of multiple strategy variants."""
    variants: list[VariantResult]

    def _fisher_exact_pvalue(self, a_wins: int, a_total: int,
                              b_wins: int, b_total: int) -> float:
        """Approximate Fisher's exact test p-value for 2x2 table.

        Uses the normal approximation to the hypergeometric distribution.
        Returns two-sided p-value.
        """
        if a_total == 0 or b_total == 0:
            return 1.0
        p_a = a_wins / a_total
        p_b = b_wins / b_total
        p_pool = (a_wins + b_wins) / (a_total + b_total)
        if p_pool == 0 or p_pool == 1:
            return 1.0
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / a_total + 1 / b_total))
        if se == 0:
            return 1.0
        z = abs(p_a - p_b) / se
        # Two-sided p-value via normal approximation
        p = 2 * (1 - _norm_cdf(z))
        return p

    def summary(self) -> str:
        lines = [
            "=" * 80,
            "A/B TEST COMPARISON",
            "=" * 80,
            "",
            f"  {'Variant':<20} {'Return':>10} {'Sharpe':>8} {'Win Rate':>10} {'Trades':>8} {'Max DD':>10}",
            "-" * 80,
        ]
        for v in self.variants:
            r = v.report
            wr_lo, wr_hi = r.win_rate_ci_lower, r.win_rate_ci_upper
            lines.append(
                f"  {v.name:<20} {r.total_return_pct:>+9.1f}% "
                f"{r.sharpe_ratio:>8.2f} "
                f"{r.win_rate:>8.1%} [{wr_lo:.0%}-{wr_hi:.0%}] "
                f"{r.total_trades:>5} "
                f"{r.max_drawdown_pct:>9.1%}"
            )
        lines.append("")

        # Statistical significance between first two variants
        if len(self.variants) >= 2:
            a, b = self.variants[0], self.variants[1]
            p_val = self._fisher_exact_pvalue(
                a.report.winners, a.report.total_trades,
                b.report.winners, b.report.total_trades,
            )
            lines.append(f"  Win rate difference: {a.name} vs {b.name}")
            lines.append(f"    p-value (two-sided): {p_val:.4f}")
            if p_val < 0.05:
                lines.append("    ** Statistically significant at 95% confidence **")
            elif p_val < 0.10:
                lines.append("    * Marginally significant (p < 0.10) *")
            else:
                lines.append("    Not statistically significant")

        lines.append("=" * 80)
        return "\n".join(lines)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def run_comparison(
    base_config: PortfolioConfig,
    provider: ThetaDataProvider,
    variants: dict[str, float] | None = None,
    verbose: bool = False,
) -> ABTestResult:
    """Run the strategy with different pullback thresholds and compare.

    Args:
        base_config: The base configuration to vary.
        provider: ThetaData pricing provider.
        variants: Dict of {name: pullback_threshold}. Defaults to
            pullback (0.03) vs always-enter (0.0).
        verbose: Print progress.

    Returns:
        ABTestResult with per-variant reports and significance tests.
    """
    if variants is None:
        variants = {
            "pullback_3pct": base_config.pullback_threshold,
            "always_enter": 0.0,
        }

    import tempfile
    import os

    results = []
    for name, threshold in variants.items():
        cfg = deepcopy(base_config)
        cfg.pullback_threshold = threshold

        # Use a temp file for the account
        tmp = tempfile.NamedTemporaryFile(
            suffix=".json", prefix=f"ab_{name}_", delete=False
        )
        tmp_path = tmp.name
        tmp.close()
        os.unlink(tmp_path)  # remove so load_or_create creates fresh
        account = SimulatedAccount.load_or_create(
            tmp_path,
            starting_capital=cfg.starting_capital,
            name=f"ab_{name}",
        )

        sim = PortfolioSimulator(account, provider, cfg, verbose=verbose)
        sim.run()
        report = sim.report()

        results.append(VariantResult(name=name, config=cfg, report=report))

        if verbose:
            print(f"\n--- {name} (threshold={threshold}) ---")
            print(report.summary())

    return ABTestResult(variants=results)
