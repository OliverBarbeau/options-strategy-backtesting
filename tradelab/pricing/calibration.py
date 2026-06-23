"""Calibration tool: compare pricing providers to measure systematic bias.

Runs the same spread quotes through two providers (typically B-S vs real
data) and computes the discrepancy. Builds a calibration dataset over time
that can be used to apply correction factors to B-S backtests.

Usage::

    from tradelab.pricing import BlackScholesProvider, ThetaDataProvider
    from tradelab.pricing.calibration import Calibrator

    bs = BlackScholesProvider()
    theta = ThetaDataProvider()

    cal = Calibrator(baseline=bs, reference=theta)
    report = cal.compare_spread(
        ticker="AAPL",
        short_strike=170,
        long_strike=165,
        expiry="2024-07-19",
        date="2024-06-14",
    )
    print(report.summary())

    # Build a calibration dataset over many quotes
    cal.calibrate_ticker("AAPL", start="2024-01-01", end="2024-06-30")
    print(cal.summary_by_ticker())
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tradelab.pricing.base import PricingProvider, SpreadQuote, PricingError

logger = logging.getLogger(__name__)

CALIBRATION_LOG_PATH = "data/calibration_log.jsonl"


@dataclass
class CalibrationEntry:
    """A single comparison between two providers."""
    timestamp: str
    ticker: str
    date: str
    expiry: str
    dte: int
    short_strike: float
    long_strike: float
    underlying_price: float

    # Baseline (usually B-S)
    baseline_source: str
    baseline_net_credit: float
    baseline_short_mid: float
    baseline_long_mid: float

    # Reference (usually Theta / real data)
    reference_source: str
    reference_net_credit: float
    reference_short_bid: float
    reference_short_ask: float
    reference_long_bid: float
    reference_long_ask: float

    # Discrepancy (reference - baseline) / baseline
    credit_bias_pct: float
    short_bias_pct: float
    long_bias_pct: float

    # Bid-ask spread characteristics (only meaningful for real data)
    reference_short_spread_pct: float
    reference_long_spread_pct: float


class CalibrationReport:
    """Analysis of a set of calibration entries."""

    def __init__(self, entries: list[CalibrationEntry]):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def mean_credit_bias(self) -> float:
        if not self.entries:
            return 0.0
        return float(np.mean([e.credit_bias_pct for e in self.entries]))

    def median_credit_bias(self) -> float:
        if not self.entries:
            return 0.0
        return float(np.median([e.credit_bias_pct for e in self.entries]))

    def std_credit_bias(self) -> float:
        if not self.entries:
            return 0.0
        return float(np.std([e.credit_bias_pct for e in self.entries]))

    def by_ticker(self) -> dict[str, "CalibrationReport"]:
        """Split entries by ticker."""
        groups: dict[str, list[CalibrationEntry]] = {}
        for e in self.entries:
            groups.setdefault(e.ticker, []).append(e)
        return {t: CalibrationReport(es) for t, es in groups.items()}

    def correction_factor(self) -> float:
        """Multiplier to apply to baseline credits to match reference.

        If baseline systematically overestimates credit by 15%, correction
        factor is 1/1.15 = 0.87 (reduce baseline credits by 13%).
        """
        bias = self.median_credit_bias()
        return 1.0 / (1.0 + bias) if bias > -1 else 1.0

    def summary(self) -> str:
        if not self.entries:
            return "No calibration data."

        lines = [
            f"Calibration Report ({len(self.entries)} samples)",
            f"  Baseline:  {self.entries[0].baseline_source}",
            f"  Reference: {self.entries[0].reference_source}",
            f"",
            f"  Credit bias (reference vs baseline):",
            f"    Mean:   {self.mean_credit_bias():+.2%}",
            f"    Median: {self.median_credit_bias():+.2%}",
            f"    Stddev: {self.std_credit_bias():.2%}",
            f"",
            f"  Correction factor for baseline: {self.correction_factor():.4f}",
            f"  (multiply B-S credits by this to match real market)",
        ]

        # By ticker
        by_ticker = self.by_ticker()
        if len(by_ticker) > 1:
            lines.append("")
            lines.append("  By ticker:")
            for ticker in sorted(by_ticker.keys()):
                r = by_ticker[ticker]
                lines.append(
                    f"    {ticker:<6} n={len(r):>3}  "
                    f"median={r.median_credit_bias():+.2%}  "
                    f"correction={r.correction_factor():.4f}"
                )

        return "\n".join(lines)


class Calibrator:
    """Compare two pricing providers across many spread quotes.

    Args:
        baseline: The provider we want to calibrate (e.g., BlackScholesProvider).
        reference: The ground-truth provider (e.g., ThetaDataProvider).
        log_path: Where to append calibration entries.
    """

    def __init__(
        self,
        baseline: PricingProvider,
        reference: PricingProvider,
        log_path: str = CALIBRATION_LOG_PATH,
    ):
        self.baseline = baseline
        self.reference = reference
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def compare_spread(
        self,
        ticker: str,
        short_strike: float,
        long_strike: float,
        expiry: str,
        date: str,
        underlying_price: float | None = None,
    ) -> CalibrationEntry | None:
        """Compare a single spread between the two providers."""
        try:
            bs_q = self.baseline.get_spread_quote(
                ticker=ticker,
                short_strike=short_strike,
                long_strike=long_strike,
                expiry=expiry,
                date=date,
                underlying_price=underlying_price,
            )
            ref_q = self.reference.get_spread_quote(
                ticker=ticker,
                short_strike=short_strike,
                long_strike=long_strike,
                expiry=expiry,
                date=date,
                underlying_price=bs_q.underlying_price,  # share the underlying
            )
        except PricingError as e:
            logger.debug(f"Skipping {ticker} {date}: {e}")
            return None

        # Credit bias
        if bs_q.net_credit_mid > 0:
            credit_bias = (ref_q.net_credit_mid - bs_q.net_credit_mid) / bs_q.net_credit_mid
        else:
            credit_bias = 0.0

        if bs_q.short_quote.mid > 0:
            short_bias = (ref_q.short_quote.mid - bs_q.short_quote.mid) / bs_q.short_quote.mid
        else:
            short_bias = 0.0

        if bs_q.long_quote.mid > 0:
            long_bias = (ref_q.long_quote.mid - bs_q.long_quote.mid) / bs_q.long_quote.mid
        else:
            long_bias = 0.0

        short_spread = ref_q.short_quote.spread_pct
        long_spread = ref_q.long_quote.spread_pct

        entry = CalibrationEntry(
            timestamp=datetime.now().isoformat(),
            ticker=ticker,
            date=date,
            expiry=expiry,
            dte=bs_q.dte,
            short_strike=short_strike,
            long_strike=long_strike,
            underlying_price=bs_q.underlying_price,
            baseline_source=self.baseline.name,
            baseline_net_credit=bs_q.net_credit_mid,
            baseline_short_mid=bs_q.short_quote.mid,
            baseline_long_mid=bs_q.long_quote.mid,
            reference_source=self.reference.name,
            reference_net_credit=ref_q.net_credit_mid,
            reference_short_bid=ref_q.short_quote.bid,
            reference_short_ask=ref_q.short_quote.ask,
            reference_long_bid=ref_q.long_quote.bid,
            reference_long_ask=ref_q.long_quote.ask,
            credit_bias_pct=credit_bias,
            short_bias_pct=short_bias,
            long_bias_pct=long_bias,
            reference_short_spread_pct=short_spread,
            reference_long_spread_pct=long_spread,
        )

        self._log_entry(entry)
        return entry

    def calibrate_ticker(
        self,
        ticker: str,
        start: str,
        end: str,
        buffer: float = 0.10,
        spread_pct: float = 0.02,
        dte_target: int = 30,
        sample_every_n_days: int = 5,
        verbose: bool = False,
    ) -> CalibrationReport:
        """Run calibration across many dates for a ticker.

        If the reference provider exposes list_expirations/list_strikes
        (like ThetaDataProvider), uses them to pick real listed strikes.
        Otherwise falls back to computed strikes.
        """
        from tradelab.pipeline import DataPipeline

        pipe = DataPipeline()
        df = pipe.fetch_stock(ticker, start=start, end=end)

        # Try to get actual listed expirations and strikes if the reference supports it
        available_expirations: list[str] = []
        has_chain_metadata = hasattr(self.reference, "list_expirations") and hasattr(self.reference, "list_strikes")
        if has_chain_metadata:
            try:
                available_expirations = self.reference.list_expirations(ticker)
                if verbose:
                    print(f"Found {len(available_expirations)} expirations for {ticker}")
            except Exception as e:
                if verbose:
                    print(f"Could not list expirations: {e}")
                has_chain_metadata = False

        # Check if reference has bulk chain capability (Theta Data does)
        has_bulk_chain = hasattr(self.reference, "get_bulk_chain")

        def find_listed_expiry(target_date_str: str) -> str:
            """Find the actual listed expiry nearest to target DTE."""
            if not available_expirations:
                dt = datetime.fromisoformat(target_date_str) + timedelta(days=dte_target)
                return dt.strftime("%Y-%m-%d")
            target = datetime.fromisoformat(target_date_str) + timedelta(days=dte_target)
            best = min(
                available_expirations,
                key=lambda e: abs((datetime.fromisoformat(e) - target).days),
            )
            return best

        def find_tradeable_strikes(
            expiry: str, date_str: str, target_short: float, target_long: float
        ) -> tuple[float, float] | None:
            """Find strikes that actually have EOD data on the target date.

            Uses bulk chain to get all strikes with data, then picks the
            closest match to target short/long strikes.
            """
            if not has_bulk_chain:
                return target_short, target_long
            try:
                chain = self.reference.get_bulk_chain(ticker, expiry, date_str, put_call="put")
            except Exception:
                return None
            if chain is None or chain.empty or "strike" not in chain.columns:
                return None
            # Only keep strikes with non-zero bid (have real NBBO data)
            valid = chain[chain["bid"] > 0] if "bid" in chain.columns else chain
            if valid.empty:
                return None
            available = sorted(valid["strike"].astype(float).unique())
            # Short: nearest strike <= target (conservative)
            valid_shorts = [s for s in available if s <= target_short + 0.01]
            if not valid_shorts:
                return None
            short = max(valid_shorts)
            # Long: nearest strike < short
            valid_longs = [s for s in available if s < short - 0.01]
            if not valid_longs:
                return None
            long = min(valid_longs, key=lambda s: abs(s - target_long))
            return short, long

        entries = []
        n = len(df)
        i = 0
        attempts = 0
        failures = 0
        while i < n:
            attempts += 1
            date_ts = df.index[i]
            date_str = pd.Timestamp(date_ts, unit="s").strftime("%Y-%m-%d")
            price = float(df["close"].iloc[i])

            target_short = price * (1 - buffer)
            target_long = target_short - price * spread_pct

            expiry = find_listed_expiry(date_str)
            result = find_tradeable_strikes(expiry, date_str, target_short, target_long)
            if result is None:
                if verbose:
                    print(f"  {date_str}: no valid strikes")
                failures += 1
                i += sample_every_n_days
                continue

            short_strike, long_strike = result

            entry = self.compare_spread(
                ticker=ticker,
                short_strike=short_strike,
                long_strike=long_strike,
                expiry=expiry,
                date=date_str,
                underlying_price=price,
            )
            if entry is not None:
                entries.append(entry)
                if verbose:
                    print(
                        f"  {date_str}: ${price:.2f} -> {short_strike:.0f}/{long_strike:.0f} "
                        f"exp {expiry}  bias={entry.credit_bias_pct:+.1%}"
                    )
            else:
                failures += 1

            i += sample_every_n_days

        if verbose:
            print(f"\nCalibration: {len(entries)}/{attempts} successful, {failures} failures")

        return CalibrationReport(entries)

    def _log_entry(self, entry: CalibrationEntry):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def load_log(self) -> CalibrationReport:
        """Load all historical calibration entries."""
        if not Path(self.log_path).exists():
            return CalibrationReport([])
        entries = []
        with open(self.log_path) as f:
            for line in f:
                if line.strip():
                    entries.append(CalibrationEntry(**json.loads(line)))
        return CalibrationReport(entries)
