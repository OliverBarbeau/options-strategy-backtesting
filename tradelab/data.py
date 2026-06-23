"""Data fetchers for stocks and crypto.

.. deprecated::
    ``StockDataFetcher`` and ``CryptoDataFetcher`` use the legacy Finnhub API.
    Use :class:`tradelab.pipeline.DataPipeline` instead, which supports
    yfinance, ccxt, and automatic parquet caching.

    ``load_csv`` has moved to :mod:`tradelab.utils` and is re-exported here
    for backward compatibility.
"""

import datetime
import time
import warnings

import pandas as pd
import requests

from tradelab.config import Config


class StockDataFetcher:
    """Fetches stock candle data from Finnhub."""

    BASE_URL = "https://finnhub.io/api/v1/stock/candle"
    DATE_FMT = "%d/%m/%Y"

    def __init__(self, api_key: str | None = None):
        warnings.warn(
            "StockDataFetcher is deprecated. Use tradelab.pipeline.DataPipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.api_key = api_key or Config.require_finnhub_key()

    def _to_timestamp(self, date_str: str) -> int:
        dt = datetime.datetime.strptime(date_str, self.DATE_FMT)
        return int(time.mktime(dt.timetuple()))

    def fetch(
        self,
        ticker: str = "SPY",
        start: str = "15/01/2001",
        end: str = "31/12/2020",
        resolution: str = "D",
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch candle data and return as a DataFrame indexed by timestamp.

        Args:
            ticker: Stock symbol.
            start: Start date in dd/mm/yyyy format.
            end: End date in dd/mm/yyyy format.
            resolution: Candle resolution (1, 5, 15, 30, 60, D, W, M).
            columns: Which price columns to keep. None keeps all of
                     ['o', 'h', 'l', 'c', 'v']. Pass ['c'] for close-only.
        """
        params = {
            "symbol": ticker,
            "resolution": resolution,
            "from": self._to_timestamp(start),
            "to": self._to_timestamp(end),
            "token": self.api_key,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("s") == "no_data":
            raise ValueError(f"No data returned for {ticker} ({start} - {end})")

        df = pd.DataFrame(data)
        df = df.set_index("t")
        df.index.name = "timestamp"

        if "s" in df.columns:
            df = df.drop(columns=["s"])

        if columns:
            df = df[[c for c in columns if c in df.columns]]

        return df


class CryptoDataFetcher:
    """Fetches crypto candle data from Finnhub."""

    BASE_URL = "https://finnhub.io/api/v1/crypto/candle"
    DATE_FMT = "%d/%m/%Y"

    def __init__(self, api_key: str | None = None):
        warnings.warn(
            "CryptoDataFetcher is deprecated. Use tradelab.pipeline.DataPipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.api_key = api_key or Config.require_finnhub_key()

    def _to_timestamp(self, date_str: str) -> int:
        dt = datetime.datetime.strptime(date_str, self.DATE_FMT)
        return int(time.mktime(dt.timetuple()))

    def fetch(
        self,
        symbol: str = "BINANCE:BTCUSDT",
        start: str = "01/01/2020",
        end: str = "01/01/2021",
        resolution: str = "60",
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch crypto candle data.

        Args:
            symbol: Exchange-prefixed symbol (e.g. BINANCE:BTCUSDT).
            start: Start date dd/mm/yyyy.
            end: End date dd/mm/yyyy.
            resolution: Candle resolution (1, 5, 15, 30, 60, D, W, M).
            columns: Price columns to keep. None keeps all.
        """
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": self._to_timestamp(start),
            "to": self._to_timestamp(end),
            "token": self.api_key,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get("s") == "no_data":
            raise ValueError(f"No data returned for {symbol} ({start} - {end})")

        df = pd.DataFrame(data)
        df = df.set_index("t")
        df.index.name = "timestamp"

        if "s" in df.columns:
            df = df.drop(columns=["s"])

        if columns:
            df = df[[c for c in columns if c in df.columns]]

        return df


# Re-export from canonical location for backward compatibility.
from tradelab.utils import load_csv  # noqa: F401
