"""Test that Simulator dispatches to pricing_provider when set."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tradelab.pricing.base import SpreadQuote, OptionQuote
from tradelab.simulator import Simulator


class TestProviderDispatch:
    """Verify _price_spread_close uses provider when available."""

    def _make_mock_quote(self, credit: float = 50.0) -> SpreadQuote:
        oq = OptionQuote(
            ticker="TEST", strike=95.0, expiry="2024-02-01",
            put_call="P", quote_date="2024-01-15", bid=1.0, ask=1.2, mid=1.1,
            underlying_price=100.0, source="mock",
        )
        return SpreadQuote(
            ticker="TEST", short_strike=95.0, long_strike=90.0,
            expiry="2024-02-01", quote_date="2024-01-15",
            net_credit=credit, net_credit_mid=credit,
            max_loss=500 - credit, spread_width=500.0,
            credit_potential=credit / (500 - credit),
            underlying_price=100.0, dte=17,
            short_quote=oq, long_quote=oq, source="mock",
        )

    def test_price_spread_close_uses_provider(self, tmp_path):
        """When pricing_provider is set, _price_spread_close should call it."""
        from tradelab.account import SimulatedAccount

        acct = SimulatedAccount.load_or_create(
            str(tmp_path / "test.json"), starting_capital=10000.0, name="test"
        )
        provider = MagicMock()
        provider.get_spread_quote.return_value = self._make_mock_quote(75.0)

        sim = Simulator(acct, pricing_provider=provider)

        result = sim._price_spread_close(
            ticker="TEST", short_strike=95.0, long_strike=90.0,
            price=100.0, vol=0.25, dte_remaining=14,
            date_str="2024-01-15", expiry="2024-02-01", contracts=2,
        )

        provider.get_spread_quote.assert_called_once()
        # Should return net_credit_mid * contracts = 75.0 * 2
        assert result == pytest.approx(150.0)

    def test_price_spread_close_falls_back_to_bs(self, tmp_path):
        """When provider raises, should fall back to B-S."""
        from tradelab.account import SimulatedAccount

        acct = SimulatedAccount.load_or_create(
            str(tmp_path / "test.json"), starting_capital=10000.0, name="test"
        )
        provider = MagicMock()
        provider.get_spread_quote.side_effect = Exception("API down")

        sim = Simulator(acct, pricing_provider=provider)

        result = sim._price_spread_close(
            ticker="TEST", short_strike=95.0, long_strike=90.0,
            price=100.0, vol=0.25, dte_remaining=14,
            date_str="2024-01-15", expiry="2024-02-01", contracts=1,
        )

        # Should still return a value (from B-S fallback)
        assert result is not None
        assert result > 0

    def test_price_spread_close_bs_only(self, tmp_path):
        """When no provider, should use B-S directly."""
        from tradelab.account import SimulatedAccount

        acct = SimulatedAccount.load_or_create(
            str(tmp_path / "test.json"), starting_capital=10000.0, name="test"
        )
        sim = Simulator(acct, pricing_provider=None)

        result = sim._price_spread_close(
            ticker="TEST", short_strike=95.0, long_strike=90.0,
            price=100.0, vol=0.25, dte_remaining=14,
            date_str="2024-01-15", expiry="2024-02-01", contracts=1,
        )

        assert result is not None
        assert result > 0
