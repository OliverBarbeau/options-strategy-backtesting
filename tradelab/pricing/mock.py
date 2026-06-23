"""Mock pricing provider for deterministic testing.

Returns fixed, predictable prices based on strike and moneyness. Used in
unit tests to avoid network dependencies and to test calibration logic
with known expected values.
"""

from __future__ import annotations

from datetime import datetime

from tradelab.pricing.base import (
    PricingProvider,
    OptionQuote,
    SpreadQuote,
)


class MockProvider(PricingProvider):
    """Deterministic mock provider.

    Pricing model (intentionally simple):
      - Put mid = max(0.01, (strike - S) * 0.5 + T * 0.3 * strike * 0.02)
        i.e., proportional to intrinsic + time value estimate
      - Bid = mid * 0.98, ask = mid * 1.02

    Args:
        default_underlying: Underlying price to use when not provided.
        credit_multiplier: Scale the synthesized credit (for testing strategies).
    """

    name = "mock"

    def __init__(
        self,
        default_underlying: float = 100.0,
        credit_multiplier: float = 1.0,
    ):
        self.default_underlying = default_underlying
        self.credit_multiplier = credit_multiplier

    def _dte_years(self, from_date: str, expiry: str) -> float:
        start = datetime.fromisoformat(from_date)
        end = datetime.fromisoformat(expiry)
        return max(1, (end - start).days) / 365

    def _put_mid(self, S: float, K: float, T: float) -> float:
        """Simple intrinsic + time value estimate."""
        intrinsic = max(0.0, K - S)
        time_value = T * S * 0.02 * max(0.1, 1 - abs(S - K) / S)
        return max(0.01, (intrinsic + time_value) * self.credit_multiplier)

    def _call_mid(self, S: float, K: float, T: float) -> float:
        intrinsic = max(0.0, S - K)
        time_value = T * S * 0.02 * max(0.1, 1 - abs(S - K) / S)
        return max(0.01, (intrinsic + time_value) * self.credit_multiplier)

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
            underlying_price = self.default_underlying

        T = self._dte_years(date, expiry)

        if put_call.upper().startswith("P"):
            mid = self._put_mid(underlying_price, strike, T)
        else:
            mid = self._call_mid(underlying_price, strike, T)

        return OptionQuote(
            ticker=ticker,
            strike=strike,
            expiry=expiry,
            put_call=put_call.upper()[0],
            bid=mid * 0.98,
            ask=mid * 1.02,
            mid=mid,
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

        start = datetime.fromisoformat(date)
        end = datetime.fromisoformat(expiry)
        dte = max(1, (end - start).days)

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
