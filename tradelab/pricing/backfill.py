"""Proactive Theta Data cache backfill utility.

Warms the parquet cache for a known universe over a date range, so that
subsequent experiments run entirely from cache (no API calls, instant).

Usage::

    from tradelab.pricing.backfill import backfill_tickers
    from tradelab.pricing.thetadata import ThetaDataProvider

    theta = ThetaDataProvider()
    backfill_tickers(
        theta,
        tickers=["NVDA", "AVGO", "MSFT"],
        start="2024-01-01",
        end="2024-12-31",
        sample_every_n_days=5,  # only sample every 5th trading day
    )

The backfill strategy:
  1. For each ticker, get its stock EOD curve from Theta (small)
  2. For each sampled trading day:
     a. Find the listed expiration nearest to day + 30 DTE
     b. Fetch the full put chain for that expiry on that day (one API call)
     c. Also fetch the put chain for day + ~14 trading days (exit date)
  3. All fetches are cached to parquet automatically

This front-loads the API work so backtests and parameter sweeps after the
backfill run entirely from cache.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd

from tradelab.pipeline import DataPipeline
from tradelab.pricing.base import PricingError
from tradelab.pricing.thetadata import ThetaDataProvider

logger = logging.getLogger(__name__)


def backfill_tickers(
    provider: ThetaDataProvider,
    tickers: list[str],
    start: str,
    end: str,
    sample_every_n_days: int = 5,
    dte_target: int = 30,
    dte_close: int = 14,
    verbose: bool = True,
) -> dict:
    """Warm the Theta Data cache for a universe over a date range.

    Returns a summary dict with counts of successful/failed fetches and
    cache statistics.
    """
    if not provider.check_connection():
        raise RuntimeError("Theta Terminal not reachable — cannot backfill")

    provider.reset_stats()
    pipe = DataPipeline()

    summary = {
        "tickers": len(tickers),
        "successful_fetches": 0,
        "failed_fetches": 0,
        "tickers_processed": [],
    }

    offset_close_days = int((dte_target - dte_close) * 7 / 5)  # calendar days approx

    for ticker in tickers:
        if verbose:
            print(f"\n[{ticker}] Backfilling {start} to {end}...")

        # Pre-fetch expirations so find_spread_strikes is fast
        try:
            expirations = provider.list_expirations(ticker)
            if verbose:
                print(f"  Loaded {len(expirations)} expirations")
        except Exception as e:
            if verbose:
                print(f"  Failed to list expirations: {e}")
            summary["failed_fetches"] += 1
            continue

        # Get trading days from yfinance (for the date index)
        try:
            df = pipe.fetch_stock(ticker, start=start, end=end)
        except Exception as e:
            if verbose:
                print(f"  Failed to load stock data: {e}")
            continue

        ticker_ok = 0
        ticker_fail = 0

        # Sample every Nth trading day
        indices = list(range(0, len(df), sample_every_n_days))
        for idx_i, i in enumerate(indices):
            date_ts = df.index[i]
            date_str = pd.Timestamp(date_ts, unit="s").strftime("%Y-%m-%d")

            # Find nearest listed expiry to date + dte_target
            target_dt = datetime.fromisoformat(date_str) + timedelta(days=dte_target)
            best_expiry = None
            best_diff = 999
            for exp in expirations:
                try:
                    exp_dt = datetime.fromisoformat(exp)
                except ValueError:
                    continue
                dte_actual = (exp_dt - datetime.fromisoformat(date_str)).days
                if dte_actual < 1:
                    continue
                diff = abs(dte_actual - dte_target)
                if diff < best_diff:
                    best_diff = diff
                    best_expiry = exp

            if best_expiry is None or best_diff > 14:
                continue

            # Fetch entry chain
            try:
                provider.get_bulk_chain(ticker, best_expiry, date_str, put_call="put")
                ticker_ok += 1
            except PricingError:
                ticker_fail += 1
                continue

            # Fetch exit chain (date + offset_close_days)
            exit_dt = datetime.fromisoformat(date_str) + timedelta(days=offset_close_days)
            if exit_dt.weekday() >= 5:  # weekend
                exit_dt += timedelta(days=7 - exit_dt.weekday())
            exit_date_str = exit_dt.strftime("%Y-%m-%d")
            try:
                provider.get_bulk_chain(ticker, best_expiry, exit_date_str, put_call="put")
                ticker_ok += 1
            except PricingError:
                ticker_fail += 1

            if verbose and (idx_i + 1) % 10 == 0:
                print(f"  Progress: {idx_i + 1}/{len(indices)} sample days "
                      f"({ticker_ok} ok, {ticker_fail} fail)")

        summary["successful_fetches"] += ticker_ok
        summary["failed_fetches"] += ticker_fail
        summary["tickers_processed"].append({
            "ticker": ticker,
            "ok": ticker_ok,
            "fail": ticker_fail,
        })

        if verbose:
            print(f"  [{ticker}] Done: {ticker_ok} successful, {ticker_fail} failed")

    if verbose:
        print()
        provider.print_stats()

    return summary
