"""Black-Scholes pricing provider.

Wraps the existing analytical pricing in tradelab.options behind the
PricingProvider interface. Fast, free, and approximate -- the baseline.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from tradelab.options import (
    bs_put_price,
    bs_call_price,
    bs_greeks,
    historical_volatility,
)
from tradelab.pipeline import DataPipeline
from tradelab.pricing.base import (
    PricingProvider,
    OptionQuote,
    SpreadQuote,
    PricingError,
)
from tradelab.pricing.strikes import snap_put_credit_spread


class BlackScholesProvider(PricingProvider):
    """Provider that computes option prices from underlying OHLCV using B-S.

    Uses 30-day historical volatility as an IV proxy. Known blind spots:
    IV skew, bid-ask spread, term structure, earnings IV crush.

    Args:
        pipe: DataPipeline instance (reuses the parquet cache).
        risk_free_rate: Annualized risk-free rate (default 0.05).
        vol_window: Window for historical volatility (default 30).
        slippage_pct: Assumed bid-ask spread as fraction of mid (default 0.02).
    """

    name = "blackscholes"

    def __init__(
        self,
        pipe: DataPipeline | None = None,
        risk_free_rate: float = 0.05,
        vol_window: int = 30,
        slippage_pct: float = 0.02,
    ):
        self.pipe = pipe or DataPipeline()
        self.r = risk_free_rate
        self.vol_window = vol_window
        self.slippage_pct = slippage_pct
        self._vol_cache: dict[str, pd.Series] = {}

    def _get_vol(self, ticker: str, date: str) -> float:
        """Get historical volatility for a ticker at a date."""
        if ticker not in self._vol_cache:
            df = self.pipe.fetch_stock(ticker, start="2015-01-01")
            self._vol_cache[ticker] = historical_volatility(
                df["close"], window=self.vol_window
            )

        vol_series = self._vol_cache[ticker]
        date_ts = int(pd.Timestamp(date).timestamp())
        idx = np.searchsorted(vol_series.index.values, date_ts, side="right") - 1
        if idx < 0:
            raise PricingError(f"No volatility data for {ticker} before {date}")
        idx = min(idx, len(vol_series) - 1)
        v = vol_series.iloc[idx]

        if np.isnan(v) or v <= 0:
            raise PricingError(f"No volatility data for {ticker} on {date}")
        return float(v)

    def _get_underlying_price(self, ticker: str, date: str) -> float:
        """Get the underlying price on a specific date."""
        df = self.pipe.fetch_stock(ticker, start="2015-01-01")
        date_ts = int(pd.Timestamp(date).timestamp())
        idx = np.searchsorted(df.index.values, date_ts, side="right") - 1
        if idx < 0:
            raise PricingError(f"No price data for {ticker} before {date}")
        return float(df["close"].iloc[idx])

    def _dte(self, from_date: str, expiry: str) -> int:
        """Calendar days from from_date to expiry."""
        start = datetime.fromisoformat(from_date)
        end = datetime.fromisoformat(expiry)
        return max(1, (end - start).days)

    def get_option_quote(
        self,
        ticker: str,
        strike: float,
        expiry: str,
        put_call: str,
        date: str,
        underlying_price: float | None = None,
    ) -> OptionQuote:
        if underlying_price is None:
            underlying_price = self._get_underlying_price(ticker, date)

        vol = self._get_vol(ticker, date)
        dte = self._dte(date, expiry)
        T = dte / 365

        if put_call.upper().startswith("P"):
            mid = float(bs_put_price(underlying_price, strike, T, self.r, vol))
            greeks = bs_greeks(underlying_price, strike, T, self.r, vol, "put")
        else:
            mid = float(bs_call_price(underlying_price, strike, T, self.r, vol))
            greeks = bs_greeks(underlying_price, strike, T, self.r, vol, "call")

        # Synthesize bid/ask from mid and assumed slippage
        half_spread = mid * self.slippage_pct / 2
        bid = max(0, mid - half_spread)
        ask = mid + half_spread

        return OptionQuote(
            ticker=ticker,
            strike=strike,
            expiry=expiry,
            put_call=put_call.upper()[0],
            bid=bid,
            ask=ask,
            mid=mid,
            delta=greeks.get("delta"),
            gamma=greeks.get("gamma"),
            theta=greeks.get("theta"),
            vega=greeks.get("vega"),
            implied_vol=vol,  # proxy: historical vol
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

        # Net credit per share, then * 100 for contract
        # Conservative (fills at worst): short_bid - long_ask
        net_credit_per_share = short_q.bid - long_q.ask
        net_credit_mid_per_share = short_q.mid - long_q.mid

        net_credit = net_credit_per_share * 100
        net_credit_mid = net_credit_mid_per_share * 100

        spread_width = (short_strike - long_strike) * 100
        max_loss = spread_width - net_credit
        credit_potential = net_credit / max_loss if max_loss > 0 else 0.0

        dte = self._dte(date, expiry)

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
            underlying_price=short_q.underlying_price,
            dte=dte,
            short_quote=short_q,
            long_quote=long_q,
            source=self.name,
        )

    def find_spread_strikes(
        self,
        ticker: str,
        date: str,
        buffer: float = 0.10,
        spread_pct: float = 0.02,
        dte_target: int = 30,
        dte_tolerance: int = 5,
        underlying_price: float | None = None,
    ):
        """Find put credit spread strikes snapped to realistic chain increments.

        Unlike the base class default (which uses fractional strikes), this
        snaps to real increments (e.g., $1, $2.50, $5) based on the ticker
        and price. The short strike is snapped DOWN to ensure the actual
        buffer is at least the target.
        """
        if underlying_price is None:
            underlying_price = self._get_underlying_price(ticker, date)

        short_strike, long_strike = snap_put_credit_spread(
            ticker=ticker,
            underlying_price=underlying_price,
            target_buffer=buffer,
            target_spread_pct=spread_pct,
        )

        if long_strike <= 0:
            return None

        expiry = self._compute_expiry(date, dte_target)

        return self.get_spread_quote(
            ticker=ticker,
            short_strike=short_strike,
            long_strike=long_strike,
            expiry=expiry,
            date=date,
            underlying_price=underlying_price,
        )

    def supports_greeks(self) -> bool:
        return True  # Computed from B-S formula

    def supports_historical(self) -> bool:
        return True
