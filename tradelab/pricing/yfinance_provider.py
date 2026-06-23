"""YFinance pricing provider: free real option chain data (current only).

Uses yfinance.Ticker.option_chain() to fetch live bid/ask/IV/volume/OI from
the current market. Limitations:

- **Current chains only** -- no historical data (yfinance doesn't archive options)
- Quotes are 15-20 min delayed on yfinance's public endpoint
- Reliability varies (same as yfinance itself -- unofficial Yahoo scraping)
- No Greeks beyond IV (no delta, gamma, theta, vega in the response)

Use cases:
1. **B-S model calibration**: compare our theoretical prices to real market
   prices for a snapshot check
2. **Development without Theta Data**: build and test the provider pipeline
   without paying $40-80/mo
3. **Current state verification**: before placing a trade, verify the B-S
   estimated credit is in the right ballpark

For real historical backtesting, use ThetaDataProvider (paid) or accumulate
yfinance snapshots over time.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from tradelab.pricing.base import (
    PricingProvider,
    OptionQuote,
    SpreadQuote,
    PricingError,
)

logger = logging.getLogger(__name__)


class YFinanceProvider(PricingProvider):
    """Current option chain data from yfinance.

    Note: date parameter is ignored for historical requests -- yfinance only
    returns current market data. The `date` arg is kept for interface
    compatibility but must match "today" (approximately).
    """

    name = "yfinance"

    def __init__(self):
        self._chain_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def _load_chain(self, ticker: str, expiry: str, put_call: str = "P") -> pd.DataFrame:
        """Load and cache a ticker's option chain for a specific expiry."""
        key = (ticker, expiry)
        if key in self._chain_cache:
            df = self._chain_cache[key]
        else:
            try:
                import yfinance as yf
                t = yf.Ticker(ticker)
                available = t.options
                if expiry not in available:
                    # Find nearest expiration
                    target = pd.Timestamp(expiry)
                    diffs = [(abs(pd.Timestamp(e) - target).days, e) for e in available]
                    if not diffs:
                        raise PricingError(f"No option expirations for {ticker}")
                    diffs.sort()
                    actual_expiry = diffs[0][1]
                    logger.info(f"Expiry {expiry} not listed; using nearest {actual_expiry}")
                else:
                    actual_expiry = expiry

                chain = t.option_chain(actual_expiry)
                # Combine puts and calls with a right column
                puts = chain.puts.copy()
                puts["right"] = "P"
                calls = chain.calls.copy()
                calls["right"] = "C"
                df = pd.concat([puts, calls], ignore_index=True)
                df["expiry"] = actual_expiry
                self._chain_cache[key] = df
            except PricingError:
                raise
            except Exception as e:
                raise PricingError(f"Failed to fetch {ticker} chain for {expiry}: {e}")

        if put_call:
            return df[df["right"] == put_call.upper()[0]]
        return df

    def _get_underlying_price(self, ticker: str) -> float:
        """Get current underlying price from yfinance."""
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.history(period="1d")
            if info.empty:
                raise PricingError(f"No price data for {ticker}")
            return float(info["Close"].iloc[-1])
        except Exception as e:
            raise PricingError(f"Failed to get price for {ticker}: {e}")

    def get_option_quote(
        self,
        ticker: str,
        strike: float,
        expiry: str,
        put_call: str,
        date: str,
        underlying_price: float | None = None,
    ) -> OptionQuote:
        chain = self._load_chain(ticker, expiry, put_call)

        # Find the exact strike (yfinance returns all strikes)
        match = chain[abs(chain["strike"] - strike) < 0.01]
        if match.empty:
            # Try nearest strike
            chain["strike_diff"] = abs(chain["strike"] - strike)
            match = chain.nsmallest(1, "strike_diff")
            if match.empty:
                raise PricingError(
                    f"No strike near {strike} for {ticker} {expiry} {put_call}"
                )

        row = match.iloc[0]
        bid = float(row.get("bid", 0))
        ask = float(row.get("ask", 0))
        last = float(row.get("lastPrice", 0))
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else last

        if underlying_price is None:
            underlying_price = self._get_underlying_price(ticker)

        return OptionQuote(
            ticker=ticker,
            strike=float(row["strike"]),
            expiry=str(row.get("expiry", expiry)),
            put_call=put_call.upper()[0],
            bid=bid,
            ask=ask,
            mid=mid,
            implied_vol=float(row.get("impliedVolatility", 0)) or None,
            volume=int(row.get("volume", 0)) if pd.notna(row.get("volume")) else None,
            open_interest=int(row.get("openInterest", 0)) if pd.notna(row.get("openInterest")) else None,
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
        if underlying_price is None:
            underlying_price = self._get_underlying_price(ticker)

        short_q = self.get_option_quote(
            ticker, short_strike, expiry, put_call, date, underlying_price
        )
        long_q = self.get_option_quote(
            ticker, long_strike, expiry, put_call, date, underlying_price
        )

        # Use the actual strikes returned (may have been snapped to nearest listed)
        actual_short = short_q.strike
        actual_long = long_q.strike

        net_credit = (short_q.bid - long_q.ask) * 100
        net_credit_mid = (short_q.mid - long_q.mid) * 100
        spread_width = (actual_short - actual_long) * 100
        max_loss = spread_width - net_credit
        credit_potential = net_credit / max_loss if max_loss > 0 else 0.0

        dt_start = datetime.fromisoformat(date)
        dt_end = datetime.fromisoformat(short_q.expiry)
        dte = max(1, (dt_end - dt_start).days)

        return SpreadQuote(
            ticker=ticker,
            short_strike=actual_short,
            long_strike=actual_long,
            expiry=short_q.expiry,
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

    def supports_historical(self) -> bool:
        return False  # Current chains only

    def supports_greeks(self) -> bool:
        return False  # Only IV, not full Greeks
