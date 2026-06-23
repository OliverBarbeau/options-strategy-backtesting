"""Theta Data provider: real historical options data via REST API v3.

Architecture:
  1. Theta Terminal (Java) runs locally on port 25503 (v3 default)
  2. We make HTTP requests to 127.0.0.1:25503/v3/* via httpx
  3. Responses are cached as parquet files for offline use

Setup:
  1. Download ThetaTerminalv3.jar from thetadata.net
  2. Install Java 21+ (Eclipse Temurin recommended)
  3. Create creds.txt with your email and password next to the jar
  4. Launch: java -jar ThetaTerminalv3.jar
  5. Verify terminal: curl http://127.0.0.1:25503/v3/option/list/symbols

Subscription tiers (determines endpoint access):
  - Free: ~1 year EOD, 20 req/min
  - Value ($40/mo): 4 years EOD, 1 thread
  - Standard ($80/mo): 8 years EOD + tick NBBO, 2 threads, Greeks
  - Pro ($160/mo): 12 years, all endpoints, 4 threads

v3 API characteristics:
  - Base URL: http://127.0.0.1:25503/v3
  - Structure: /v3/{asset}/{action}/{subject}
    e.g., /v3/option/history/eod, /v3/option/list/strikes
  - Parameter names: symbol (not root), expiration (not exp), strike in dollars
  - right values: "call", "put", "both" (lowercase in requests, upper in responses)
  - Dates: YYYYMMDD or YYYY-MM-DD, responses always use YYYY-MM-DD
  - Default format: CSV with header row
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from tradelab.cache import DataCache
from tradelab.pricing.base import (
    PricingProvider,
    OptionQuote,
    SpreadQuote,
    PricingError,
)

logger = logging.getLogger(__name__)


@dataclass
class ThetaConfig:
    """Theta Terminal connection config.

    Ports:
      25503 - Terminal v3 default (current)
      25510 - Terminal v2 legacy port
    """
    host: str = "127.0.0.1"
    port: int = 25503  # Terminal v3 default
    timeout: float = 30.0
    rate_limit_per_min: int = 120  # Standard tier; raise if you have Pro


class ThetaDataProvider(PricingProvider):
    """Historical option data from Theta Data REST API v3.

    Args:
        config: Connection config (host/port/timeout).
        cache_dir: Where to store parquet cache.
        risk_free_rate: Used only for fallback B-S calculations.
    """

    name = "thetadata"

    def __init__(
        self,
        config: ThetaConfig | None = None,
        cache_dir: str = "data/theta_cache",
        risk_free_rate: float = 0.05,
        verbose: bool = False,
    ):
        self.config = config or ThetaConfig()
        self.cache = DataCache(cache_dir)
        self.r = risk_free_rate
        self.verbose = verbose
        self._last_request_time = 0.0
        self._client = None

        # Observability: track cache and API behavior
        self.stats = {
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "chain_fetches": 0,
            "strike_lookups": 0,
            "errors": 0,
        }

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}/v3"

    def _get_client(self):
        """Lazy-load httpx client."""
        if self._client is None:
            try:
                import httpx
            except ImportError:
                raise PricingError(
                    "httpx is required for ThetaDataProvider. "
                    "Install with: pip install httpx"
                )
            self._client = httpx.Client(timeout=self.config.timeout)
        return self._client

    def _rate_limit(self):
        """Honor the rate limit."""
        import time
        min_interval = 60.0 / self.config.rate_limit_per_min
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def check_connection(self) -> bool:
        """Test if the Theta Terminal is reachable."""
        try:
            client = self._get_client()
            r = client.get(f"{self.base_url}/option/list/symbols", params={"format": "csv"})
            return r.status_code == 200
        except Exception as e:
            logger.warning(f"Theta Terminal not reachable: {e}")
            return False

    def _format_date(self, date_str: str) -> str:
        """Convert YYYY-MM-DD to YYYYMMDD (Theta v3 accepts both, we use compact)."""
        return date_str.replace("-", "")

    def _request_csv(self, endpoint: str, params: dict) -> pd.DataFrame:
        """Make a rate-limited request and parse CSV response into a DataFrame."""
        self._rate_limit()
        self.stats["api_calls"] += 1
        client = self._get_client()
        url = f"{self.base_url}{endpoint}"
        params = {**params, "format": "csv"}

        if self.verbose:
            logger.info(f"[Theta API] GET {endpoint} {params}")

        try:
            r = client.get(url, params=params)
        except Exception as e:
            self.stats["errors"] += 1
            raise PricingError(
                f"Theta Terminal not reachable at {self.base_url}. "
                f"Is the Java Terminal running? Error: {e}"
            )

        if r.status_code != 200:
            self.stats["errors"] += 1
            raise PricingError(
                f"Theta API error {r.status_code} on {url}: {r.text[:300]}"
            )

        text = r.text.strip()
        if not text or text.startswith("<html"):
            self.stats["errors"] += 1
            raise PricingError(f"Invalid response from {url}: {text[:200]}")

        try:
            return pd.read_csv(io.StringIO(text))
        except Exception as e:
            self.stats["errors"] += 1
            raise PricingError(f"Failed to parse CSV from {url}: {e}\nBody: {text[:300]}")

    def print_stats(self):
        """Print cache and API usage statistics."""
        s = self.stats
        total_reads = s["cache_hits"] + s["cache_misses"]
        hit_rate = (s["cache_hits"] / total_reads * 100) if total_reads > 0 else 0
        print(f"[Theta stats]")
        print(f"  API calls:     {s['api_calls']}")
        print(f"  Cache hits:    {s['cache_hits']}")
        print(f"  Cache misses:  {s['cache_misses']}")
        print(f"  Hit rate:      {hit_rate:.1f}%")
        print(f"  Chain fetches: {s['chain_fetches']}")
        print(f"  Strike lookups: {s['strike_lookups']}")
        print(f"  Errors:        {s['errors']}")

    def reset_stats(self):
        """Reset statistics counters."""
        for k in self.stats:
            self.stats[k] = 0

    # ------------------------------------------------------------------
    # List endpoints
    # ------------------------------------------------------------------

    def list_expirations(self, ticker: str) -> list[str]:
        """Get all option expirations for a ticker. Returns list of YYYY-MM-DD strings."""
        df = self._request_csv(
            "/option/list/expirations",
            {"symbol": ticker},
        )
        if df.empty or "expiration" not in df.columns:
            return []
        return df["expiration"].astype(str).tolist()

    def list_strikes(self, ticker: str, expiry: str) -> list[float]:
        """Get all strikes for a ticker + expiration."""
        df = self._request_csv(
            "/option/list/strikes",
            {"symbol": ticker, "expiration": self._format_date(expiry)},
        )
        if df.empty or "strike" not in df.columns:
            return []
        return sorted(df["strike"].astype(float).tolist())

    def get_stock_eod(self, ticker: str, date: str) -> float | None:
        """Get the raw (unadjusted) EOD close price for a ticker on a date.

        This is critical when working with historical options because option
        strikes are at-traded levels (pre-split). Using yfinance's adjusted
        prices will mismatch with Theta's option strikes after any split.

        Returns None if no data (e.g., weekend, holiday).
        """
        cache_key = f"stock_{ticker}"
        cached = self.cache.get("thetadata_stock_eod", cache_key, self._format_date(date))
        if cached is not None and not cached.empty:
            self.stats["cache_hits"] += 1
            return float(cached.iloc[0]["close"])

        self.stats["cache_misses"] += 1
        try:
            df = self._request_csv(
                "/stock/history/eod",
                {
                    "symbol": ticker,
                    "start_date": self._format_date(date),
                    "end_date": self._format_date(date),
                },
            )
        except PricingError:
            return None

        if df.empty or "close" not in df.columns:
            return None

        self.cache.put("thetadata_stock_eod", cache_key, self._format_date(date), df)
        return float(df.iloc[0]["close"])

    def get_stock_history(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Get raw (unadjusted) EOD stock history for a date range.

        Returns a DataFrame indexed by date (YYYY-MM-DD) with columns:
        open, high, low, close, volume.

        Theta caps stock history requests at 365 days. For longer windows,
        we chunk the request and stitch the results. The combined result
        is cached as a single parquet per (ticker, start, end) tuple.
        """
        from datetime import datetime as _dt, timedelta as _td

        cache_key = f"history_{ticker}_{self._format_date(start_date)}_{self._format_date(end_date)}"
        cached = self.cache.get("thetadata_stock_history", cache_key, "range")
        if cached is not None and not cached.empty:
            self.stats["cache_hits"] += 1
            return cached

        # Chunk into <=350-day windows to stay under Theta's 365-day limit
        start_dt = _dt.fromisoformat(start_date)
        end_dt = _dt.fromisoformat(end_date)
        max_chunk_days = 350

        chunks = []
        chunk_start = start_dt
        while chunk_start <= end_dt:
            chunk_end = min(chunk_start + _td(days=max_chunk_days), end_dt)
            self.stats["cache_misses"] += 1
            df_chunk = self._request_csv(
                "/stock/history/eod",
                {
                    "symbol": ticker,
                    "start_date": self._format_date(chunk_start.strftime("%Y-%m-%d")),
                    "end_date": self._format_date(chunk_end.strftime("%Y-%m-%d")),
                },
            )
            if not df_chunk.empty:
                chunks.append(df_chunk)
            chunk_start = chunk_end + _td(days=1)

        if not chunks:
            raise PricingError(
                f"No stock history for {ticker} from {start_date} to {end_date}"
            )

        df = pd.concat(chunks, ignore_index=True)
        if "close" not in df.columns:
            raise PricingError(
                f"Malformed stock history for {ticker}: no close column"
            )

        # Extract date from the last_trade timestamp, dedupe, sort
        if "last_trade" in df.columns:
            df["date"] = df["last_trade"].astype(str).str[:10]
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df = df.drop_duplicates(subset="date").set_index("date").sort_index()

        self.cache.put("thetadata_stock_history", cache_key, "range", df)
        return df

    # ------------------------------------------------------------------
    # Historical EOD
    # ------------------------------------------------------------------

    def get_bulk_chain(
        self,
        ticker: str,
        expiry: str,
        date: str,
        put_call: str = "put",
    ) -> pd.DataFrame:
        """Get EOD data for all strikes on an expiration.

        This is the CANONICAL cache path. All strike-specific lookups go
        through here and filter the result. One API call per (ticker, expiry,
        date, put_call) tuple — cached forever as parquet.
        """
        self.stats["chain_fetches"] += 1
        cache_key = f"chain_{ticker}_{self._format_date(expiry)}_{put_call}"
        cached = self.cache.get("thetadata_bulk_v3", cache_key, self._format_date(date))
        if cached is not None and not cached.empty:
            self.stats["cache_hits"] += 1
            return cached

        self.stats["cache_misses"] += 1

        params = {
            "symbol": ticker,
            "expiration": self._format_date(expiry),
            "start_date": self._format_date(date),
            "end_date": self._format_date(date),
            "right": put_call.lower(),
        }

        df = self._request_csv("/option/history/eod", params)

        if df.empty:
            raise PricingError(
                f"No chain data for {ticker} exp {expiry} on {date}"
            )

        self.cache.put("thetadata_bulk_v3", cache_key, self._format_date(date), df)
        return df

    def get_eod(
        self,
        ticker: str,
        expiry: str,
        date: str,
        strike: float | None = None,
        put_call: str = "both",
    ) -> pd.DataFrame:
        """Get EOD option data for a specific strike (or all if strike=None).

        Filters from the cached bulk chain when possible. Only hits the API
        for single-strike queries if the bulk chain for that expiry+date
        isn't cached.
        """
        self.stats["strike_lookups"] += 1

        if strike is None:
            # Caller wants all strikes -- dispatch to bulk chain
            if put_call == "both":
                # Need both sides -- fetch puts and calls separately
                puts = self.get_bulk_chain(ticker, expiry, date, put_call="put")
                calls = self.get_bulk_chain(ticker, expiry, date, put_call="call")
                return pd.concat([puts, calls], ignore_index=True)
            return self.get_bulk_chain(ticker, expiry, date, put_call=put_call)

        # Single-strike query: try to filter from cached bulk chain first
        pc = put_call.lower()
        if pc == "both":
            # Can't do single strike with both sides in one filter — fall to direct
            pc = "put"  # assume put (our primary use case)

        try:
            chain = self.get_bulk_chain(ticker, expiry, date, put_call=pc)
            match = chain[abs(chain["strike"] - strike) < 0.01]
            if not match.empty:
                return match
        except PricingError:
            pass

        # Bulk chain didn't have the strike or failed — fall back to direct query
        # (This should be rare; most strikes are in the bulk chain)
        params = {
            "symbol": ticker,
            "expiration": self._format_date(expiry),
            "start_date": self._format_date(date),
            "end_date": self._format_date(date),
            "right": pc,
            "strike": f"{strike:.3f}",
        }
        df = self._request_csv("/option/history/eod", params)
        if df.empty:
            raise PricingError(
                f"No EOD data for {ticker} exp {expiry} strike {strike} on {date}"
            )
        return df

    # ------------------------------------------------------------------
    # Strike/expiry discovery (override base class default)
    # ------------------------------------------------------------------

    def find_spread_strikes(
        self,
        ticker: str,
        date: str,
        buffer: float = 0.10,
        spread_pct: float = 0.02,
        dte_target: int = 30,
        dte_tolerance: int = 7,
        underlying_price: float | None = None,
    ):
        """Find a valid put credit spread from real listed expiries and strikes.

        1. Get underlying price (from Theta stock endpoint -- raw/unadjusted,
           which matches the as-traded option strikes)
        2. List expirations for the ticker
        3. Find the expiration nearest to target DTE
        4. Pull bulk chain for that expiration on the target date
        5. Pick strikes that actually have EOD data with non-zero bid
        6. Return a SpreadQuote with real market prices

        Note: If underlying_price is provided, it MUST be raw (unadjusted)
        to match Theta's option strikes. If not provided, we fetch from
        Theta's stock endpoint directly, which is always raw.
        """
        from datetime import datetime as _dt

        if underlying_price is None:
            underlying_price = self.get_stock_eod(ticker, date)

        # Get listed expirations (cached)
        if not hasattr(self, "_exp_cache"):
            self._exp_cache = {}
        if ticker not in self._exp_cache:
            try:
                self._exp_cache[ticker] = self.list_expirations(ticker)
            except Exception:
                self._exp_cache[ticker] = []
        expirations = self._exp_cache[ticker]
        if not expirations:
            return None

        # Find nearest listed expiration to date + dte_target
        target_date = _dt.fromisoformat(date)
        best_expiry = None
        best_diff = 999
        for exp in expirations:
            try:
                exp_dt = _dt.fromisoformat(exp)
            except ValueError:
                continue
            dte_actual = (exp_dt - target_date).days
            if dte_actual < 1:
                continue
            diff = abs(dte_actual - dte_target)
            if diff < best_diff:
                best_diff = diff
                best_expiry = exp

        if best_expiry is None or best_diff > dte_tolerance * 2:
            return None

        # Pull bulk chain for this expiration on the target date
        try:
            chain = self.get_bulk_chain(ticker, best_expiry, date, put_call="put")
        except PricingError:
            return None

        if chain is None or chain.empty or "strike" not in chain.columns:
            return None

        # Keep strikes with real NBBO data
        if "bid" in chain.columns:
            valid = chain[chain["bid"] > 0]
        else:
            valid = chain
        if valid.empty:
            return None

        available_strikes = sorted(valid["strike"].astype(float).unique())

        # If we still don't have an underlying price, estimate from the chain.
        # The highest-volume strike is typically near ATM. If no volume data,
        # use the median of all listed strikes as a rough proxy.
        if underlying_price is None:
            if "volume" in valid.columns:
                vol_col = valid["volume"].fillna(0)
                if vol_col.max() > 0:
                    atm_row = valid.loc[vol_col.idxmax()]
                    underlying_price = float(atm_row["strike"])
                else:
                    underlying_price = float(valid["strike"].median())
            else:
                underlying_price = float(valid["strike"].median())

        # Target strikes
        target_short = underlying_price * (1 - buffer)
        target_long = target_short - underlying_price * spread_pct

        # Short: nearest listed strike <= target (conservative, keeps buffer >= target)
        valid_shorts = [s for s in available_strikes if s <= target_short + 0.01]
        if not valid_shorts:
            return None
        short_strike = max(valid_shorts)

        # Long: nearest listed strike < short, closest to target
        valid_longs = [s for s in available_strikes if s < short_strike - 0.01]
        if not valid_longs:
            return None
        long_strike = min(valid_longs, key=lambda s: abs(s - target_long))

        # Now price the spread using the bulk chain we already fetched (no extra calls)
        short_row = valid[abs(valid["strike"] - short_strike) < 0.01].iloc[0]
        long_row = valid[abs(valid["strike"] - long_strike) < 0.01].iloc[0]

        short_bid = float(short_row.get("bid", 0))
        short_ask = float(short_row.get("ask", 0))
        short_mid = (short_bid + short_ask) / 2 if (short_bid > 0 and short_ask > 0) else float(short_row.get("close", 0))

        long_bid = float(long_row.get("bid", 0))
        long_ask = float(long_row.get("ask", 0))
        long_mid = (long_bid + long_ask) / 2 if (long_bid > 0 and long_ask > 0) else float(long_row.get("close", 0))

        net_credit = (short_bid - long_ask) * 100
        net_credit_mid = (short_mid - long_mid) * 100
        spread_width = (short_strike - long_strike) * 100
        max_loss = spread_width - net_credit
        credit_potential = net_credit / max_loss if max_loss > 0 else 0.0

        dte = (_dt.fromisoformat(best_expiry) - target_date).days

        short_q = OptionQuote(
            ticker=ticker, strike=short_strike, expiry=best_expiry, put_call="P",
            bid=short_bid, ask=short_ask, mid=short_mid,
            underlying_price=underlying_price, quote_date=date, source=self.name,
        )
        long_q = OptionQuote(
            ticker=ticker, strike=long_strike, expiry=best_expiry, put_call="P",
            bid=long_bid, ask=long_ask, mid=long_mid,
            underlying_price=underlying_price, quote_date=date, source=self.name,
        )

        return SpreadQuote(
            ticker=ticker,
            short_strike=short_strike,
            long_strike=long_strike,
            expiry=best_expiry,
            quote_date=date,
            net_credit=net_credit,
            net_credit_mid=net_credit_mid,
            max_loss=max_loss,
            spread_width=spread_width,
            credit_potential=credit_potential,
            underlying_price=underlying_price,
            dte=dte,
            short_quote=short_q,
            long_quote=long_q,
            source=self.name,
        )

    # ------------------------------------------------------------------
    # PricingProvider interface
    # ------------------------------------------------------------------

    def get_option_quote(
        self,
        ticker: str,
        strike: float,
        expiry: str,
        put_call: str,
        date: str,
        underlying_price: float | None = None,
    ) -> OptionQuote:
        pc = "put" if put_call.upper().startswith("P") else "call"
        df = self.get_eod(ticker, expiry, date, strike=strike, put_call=pc)

        if df.empty:
            raise PricingError(f"No data for {ticker} {put_call} {strike} exp {expiry}")

        row = df.iloc[0]
        bid = float(row.get("bid", 0)) if pd.notna(row.get("bid")) else 0.0
        ask = float(row.get("ask", 0)) if pd.notna(row.get("ask")) else 0.0
        close = float(row.get("close", 0)) if pd.notna(row.get("close")) else 0.0
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else close

        volume = None
        if "volume" in df.columns and pd.notna(row.get("volume")):
            volume = int(row["volume"])

        # Auto-fetch raw underlying price if not provided (cached after first call)
        if underlying_price is None:
            underlying_price = self.get_stock_eod(ticker, date) or 0.0

        return OptionQuote(
            ticker=ticker,
            strike=float(row.get("strike", strike)),
            expiry=str(row.get("expiration", expiry)),
            put_call=put_call.upper()[0],
            bid=bid,
            ask=ask,
            mid=mid,
            volume=volume,
            underlying_price=underlying_price,
            quote_date=date,
            source=self.name,
        )

    def get_spread_quote(
        self,
        ticker: str,
        short_strike: float,
        long_strike: float,
        expiry: str,
        date: str,
        underlying_price: float | None = None,
        put_call: str = "P",
    ) -> SpreadQuote:
        short_q = self.get_option_quote(
            ticker, short_strike, expiry, put_call, date, underlying_price
        )
        long_q = self.get_option_quote(
            ticker, long_strike, expiry, put_call, date, underlying_price
        )

        net_credit = (short_q.bid - long_q.ask) * 100
        net_credit_mid = (short_q.mid - long_q.mid) * 100
        spread_width = (short_strike - long_strike) * 100
        max_loss = spread_width - net_credit
        credit_potential = net_credit / max_loss if max_loss > 0 else 0.0

        dt_start = datetime.fromisoformat(date)
        dt_end = datetime.fromisoformat(expiry)
        dte = max(1, (dt_end - dt_start).days)

        return SpreadQuote(
            ticker=ticker,
            short_strike=short_strike,
            long_strike=long_strike,
            expiry=expiry,
            quote_date=date,
            net_credit=net_credit,
            net_credit_mid=net_credit_mid,
            max_loss=max_loss,
            spread_width=spread_width,
            credit_potential=credit_potential,
            underlying_price=underlying_price or 0,
            dte=dte,
            short_quote=short_q,
            long_quote=long_q,
            source=self.name,
        )

    def supports_greeks(self) -> bool:
        # EOD endpoint doesn't include Greeks; use greeks/eod endpoint (Standard+)
        return False

    def supports_historical(self) -> bool:
        return True

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
