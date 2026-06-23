"""Probability engine for historical price movement analysis.

Ported from time_period_gains_calc.ipynb (time_period_loss + Script 2).
"""

from __future__ import annotations

import datetime

import pandas as pd

from tradelab.data import StockDataFetcher


class ProbabilityEngine:
    """Calculates how often a stock historically dropped below a threshold
    within a given number of trading days.

    This is the core edge calculator for options strategies -- it tells you
    the historical probability of a put finishing in the money.
    """

    def __init__(self, df: pd.DataFrame | None = None, close_col: str = "c"):
        """
        Args:
            df: A DataFrame with price data. If None, use fetch() first.
            close_col: Column name for close prices.
        """
        self.df = df
        self.close_col = close_col

    def fetch(
        self,
        ticker: str = "SPY",
        start: str = "15/01/2010",
        end: str = "02/10/2023",
        api_key: str | None = None,
    ) -> "ProbabilityEngine":
        """Fetch data and store internally. Returns self for chaining."""
        fetcher = StockDataFetcher(api_key=api_key)
        self.df = fetcher.fetch(ticker, start, end, columns=[self.close_col])
        return self

    def time_period_loss(
        self,
        offset: int = 10,
        price_adjust: float = 0.95,
    ) -> tuple[int, int]:
        """Count how many times the price was below (price * price_adjust)
        after *offset* trading days.

        Args:
            offset: Number of trading days to look ahead.
            price_adjust: Multiplier on the starting price (e.g. 0.95 = 5% drop).

        Returns:
            (hit_count, total_windows) tuple.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call fetch() or pass df to __init__.")

        closes = self.df[self.close_col].values
        total = len(closes) - offset
        if total <= 0:
            return 0, 0

        hits = 0
        for i in range(total):
            if closes[i] * price_adjust > closes[i + offset]:
                hits += 1

        return hits, total

    def strike_probability_table(
        self,
        offset: int = 260,
        adjust_start: float = 0.80,
        adjust_end: float = 1.10,
        step: float = 0.01,
    ) -> pd.DataFrame:
        """Generate a table showing loss probability at various price adjustments.

        Ported from Script 2 in time_period_gains_calc.ipynb.

        Args:
            offset: Trading days to look ahead.
            adjust_start: Starting price adjustment multiplier.
            adjust_end: Ending price adjustment multiplier.
            step: Increment per row.

        Returns:
            DataFrame with columns: adjust, adj_price, probability, prob_delta.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call fetch() or pass df to __init__.")

        current_price = self.df[self.close_col].iloc[-1]
        rows = []
        last_prob = 0.0
        adjust = adjust_start

        while adjust <= adjust_end + 1e-9:
            hits, total = self.time_period_loss(offset, adjust)
            prob = hits / total if total > 0 else 0.0
            delta = ((prob - last_prob) / prob * 100) if prob > 0 else 0.0

            rows.append(
                {
                    "adjust": round(adjust, 4),
                    "adj_price": round(current_price * adjust, 2),
                    "probability": round(prob * 100, 2),
                    "prob_delta": round(delta, 2),
                }
            )
            last_prob = prob
            adjust += step

        return pd.DataFrame(rows)

    def summary(self, ticker: str, offset: int = 260) -> str:
        """Print a human-readable summary header for the probability table."""
        if self.df is None:
            raise ValueError("No data loaded.")

        current_price = self.df[self.close_col].iloc[-1]
        start_ts = self.df.index[0]
        end_ts = self.df.index[-1]
        start_dt = datetime.datetime.fromtimestamp(start_ts)
        end_dt = datetime.datetime.fromtimestamp(end_ts)
        delta = end_dt - start_dt

        return (
            f"{ticker}: ${current_price:.2f} per share on {end_dt:%Y-%m-%d}\n"
            f"{start_dt:%Y-%m-%d} - {end_dt:%Y-%m-%d}  [{delta.days} days]\n"
            f"Offset: {offset} trading days"
        )
