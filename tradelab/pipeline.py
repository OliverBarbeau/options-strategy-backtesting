"""Unified data pipeline with multi-source fetching and local caching.

Wraps yfinance, ccxt, and Finnhub behind a single interface.
All fetches go through the local parquet cache automatically.
"""

from __future__ import annotations

import datetime
import time

import numpy as np
import pandas as pd

from tradelab.cache import DataCache


class DataPipeline:
    """Single entry point for all market data.

    Usage::

        pipe = DataPipeline()

        # Stocks -- uses yfinance, cached locally
        aapl = pipe.fetch_stock("AAPL", start="2020-01-01", end="2023-01-01")

        # Multiple tickers at once
        data = pipe.fetch_stocks(["AAPL", "MSFT", "SPY"], start="2020-01-01")

        # Crypto -- uses ccxt (Kraken by default)
        btc = pipe.fetch_crypto("BTC/USDT", start="2021-01-01")

        # Force refresh from API
        aapl = pipe.fetch_stock("AAPL", start="2020-01-01", refresh=True)
    """

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache = DataCache(cache_dir)

    # ------------------------------------------------------------------
    # Stocks / ETFs via yfinance
    # ------------------------------------------------------------------

    def fetch_stock(
        self,
        ticker: str,
        start: str = "2015-01-01",
        end: str | None = None,
        interval: str = "1d",
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch stock OHLCV data via yfinance with local caching.

        Args:
            ticker: Stock symbol (e.g. "AAPL", "SPY").
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD). None = today.
            interval: Candle interval (1d, 1wk, 1mo, 1h, etc.).
            refresh: Force re-download even if cached.

        Returns:
            DataFrame with columns: open, high, low, close, volume.
            Indexed by unix timestamp (int) for compatibility with
            the rest of tradelab.
        """
        resolution = _yf_interval_to_key(interval)

        if not refresh:
            cached = self.cache.get("yfinance", ticker, resolution)
            if cached is not None:
                return _filter_date_range(cached, start, end)

        import yfinance as yf

        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )

        if raw.empty:
            raise ValueError(f"No data returned for {ticker} ({start} - {end})")

        df = _normalize_yf(raw, ticker)
        self.cache.put("yfinance", ticker, resolution, df)
        return _filter_date_range(df, start, end)

    def fetch_stocks(
        self,
        tickers: list[str],
        start: str = "2015-01-01",
        end: str | None = None,
        interval: str = "1d",
        refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Fetch multiple stock tickers. Returns {ticker: DataFrame}."""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = self.fetch_stock(
                    ticker, start, end, interval, refresh
                )
            except (ValueError, Exception) as e:
                print(f"Warning: failed to fetch {ticker}: {e}")
        return results

    # ------------------------------------------------------------------
    # Crypto via ccxt
    # ------------------------------------------------------------------

    def fetch_crypto(
        self,
        symbol: str = "BTC/USDT",
        start: str = "2020-01-01",
        end: str | None = None,
        timeframe: str = "1d",
        exchange: str = "kraken",
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch crypto OHLCV data via ccxt with local caching.

        Args:
            symbol: Trading pair (e.g. "BTC/USDT", "ETH/USDT").
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD). None = now.
            timeframe: Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1w).
            exchange: Exchange name (kraken, coinbasepro, bybit, etc.).
            refresh: Force re-download.

        Returns:
            DataFrame with columns: open, high, low, close, volume.
            Indexed by unix timestamp (int).
        """
        cache_key = f"{exchange}_{symbol.replace('/', '-')}"
        resolution = timeframe

        if not refresh:
            cached = self.cache.get("ccxt", cache_key, resolution)
            if cached is not None:
                return _filter_date_range(cached, start, end)

        import ccxt

        exchange_class = getattr(ccxt, exchange, None)
        if exchange_class is None:
            raise ValueError(f"Unknown exchange: {exchange}")

        ex = exchange_class({"enableRateLimit": True})

        since_ms = int(
            datetime.datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000
        )
        end_ms = (
            int(datetime.datetime.strptime(end, "%Y-%m-%d").timestamp() * 1000)
            if end
            else int(time.time() * 1000)
        )

        all_candles = []
        cursor = since_ms
        while cursor < end_ms:
            candles = ex.fetch_ohlcv(
                symbol, timeframe, since=cursor, limit=500
            )
            if not candles:
                break
            all_candles.extend(candles)
            # Move cursor past the last candle
            cursor = candles[-1][0] + 1
            if len(candles) < 500:
                break

        if not all_candles:
            raise ValueError(
                f"No data returned for {symbol} on {exchange} ({start} - {end})"
            )

        df = pd.DataFrame(
            all_candles, columns=["timestamp_ms", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = (df["timestamp_ms"] // 1000).astype(int)
        df = df.set_index("timestamp")
        df = df.drop(columns=["timestamp_ms"])
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        self.cache.put("ccxt", cache_key, resolution, df)
        return _filter_date_range(df, start, end)

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    @staticmethod
    def add_indicators(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
        """Add common technical indicators to a DataFrame in-place.

        Adds: sma_20, sma_50, sma_200, ema_12, ema_26, macd, hist_vol_30.
        """
        c = df[close_col]
        df["sma_20"] = c.rolling(20).mean()
        df["sma_50"] = c.rolling(50).mean()
        df["sma_200"] = c.rolling(200).mean()
        df["ema_12"] = c.ewm(span=12, adjust=False).mean()
        df["ema_26"] = c.ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

        # 30-day historical volatility (annualized)
        log_returns = np.log(c / c.shift(1))
        df["hist_vol_30"] = log_returns.rolling(30).std() * np.sqrt(252)

        return df


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _normalize_yf(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance output to standard tradelab format.

    yfinance returns MultiIndex columns when downloading single tickers
    as of v1.0+. This flattens to: open, high, low, close, volume
    with a unix timestamp integer index.
    """
    df = raw.copy()

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        # Drop the ticker level -- columns are like ('Close', 'AAPL')
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [col.lower() for col in df.columns]

    # Standardize column names
    rename = {"adj close": "close"}
    df = df.rename(columns=rename)

    # Keep only OHLCV
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]

    # Convert DatetimeIndex to unix timestamp int
    # datetime64 resolution varies (ns, us, ms, s) -- normalize via .timestamp()
    df.index = pd.Index([int(ts.timestamp()) for ts in df.index], name="timestamp")

    return df


def _yf_interval_to_key(interval: str) -> str:
    """Map yfinance interval strings to cache key suffixes."""
    return interval.replace(" ", "")


def _filter_date_range(
    df: pd.DataFrame, start: str, end: str | None
) -> pd.DataFrame:
    """Filter a timestamp-indexed DataFrame to a date range."""
    import calendar

    dt = datetime.datetime.strptime(start, "%Y-%m-%d")
    start_ts = int(calendar.timegm(dt.timetuple()))
    if end:
        dt_end = datetime.datetime.strptime(end, "%Y-%m-%d")
        end_ts = int(calendar.timegm(dt_end.timetuple()))
        return df[(df.index >= start_ts) & (df.index <= end_ts)]
    return df[df.index >= start_ts]
