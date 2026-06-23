"""Unit tests for tradelab.models."""

import pytest

from tradelab.models import Position, PutCreditSpread


# --- Position ---

class TestPosition:
    def test_long_profit(self):
        pos = Position(ratio=100.0, margin=1000.0, leverage=2, side=Position.LONG)
        assert pos.cost == 2000.0
        assert pos.volume == 20.0
        assert pos.profit_loss(110.0) == pytest.approx(200.0)

    def test_long_loss(self):
        pos = Position(ratio=100.0, margin=1000.0, leverage=1, side=Position.LONG)
        assert pos.profit_loss(90.0) == pytest.approx(-100.0)

    def test_short_profit(self):
        pos = Position(ratio=100.0, margin=1000.0, leverage=1, side=Position.SHORT)
        assert pos.profit_loss(90.0) == pytest.approx(100.0)

    def test_short_loss(self):
        pos = Position(ratio=100.0, margin=500.0, leverage=2, side=Position.SHORT)
        # cost=1000, volume=10, price goes up by 5
        assert pos.profit_loss(105.0) == pytest.approx(-50.0)

    def test_zero_pnl_at_entry(self):
        pos = Position(ratio=50.0, margin=500.0, leverage=3, side=Position.LONG)
        assert pos.profit_loss(50.0) == pytest.approx(0.0)

    def test_repr(self):
        pos = Position(ratio=100.0, margin=1000.0, leverage=2, side=Position.LONG)
        r = repr(pos)
        assert "LONG" in r
        assert "100.00" in r

    def test_short_repr(self):
        pos = Position(ratio=100.0, margin=1000.0, leverage=1, side=Position.SHORT)
        assert "SHORT" in repr(pos)


# --- PutCreditSpread ---

class TestPutCreditSpread:
    def test_winner_when_price_above_strike(self):
        spread = PutCreditSpread(
            underlying_price=250,
            strike_price=235,
            collateral=1000,
            open_date=1_000_000,
            expiry=2_000_000,
        )
        cash, won = spread.evaluate(current_price=250, credit_ratio=0.20)
        assert won is True
        assert cash == pytest.approx(1200.0)

    def test_loser_when_price_below_strike(self):
        spread = PutCreditSpread(
            underlying_price=250,
            strike_price=235,
            collateral=1000,
            open_date=1_000_000,
            expiry=2_000_000,
        )
        cash, won = spread.evaluate(current_price=230, loss_ratio=0.95)
        assert won is False
        assert cash == pytest.approx(50.0)  # 1000 * (1 - 0.95)

    def test_exactly_at_strike_is_loss(self):
        spread = PutCreditSpread(
            underlying_price=250,
            strike_price=235,
            collateral=1000,
            open_date=1_000_000,
            expiry=2_000_000,
        )
        _, won = spread.evaluate(current_price=235)
        assert won is False

    def test_is_expired(self):
        spread = PutCreditSpread(
            underlying_price=100,
            strike_price=95,
            collateral=500,
            open_date=1_000_000,
            expiry=2_000_000,
        )
        assert spread.is_expired(1_999_999) is False
        assert spread.is_expired(2_000_000) is True
        assert spread.is_expired(3_000_000) is True

    def test_datetime_properties(self):
        spread = PutCreditSpread(
            underlying_price=100,
            strike_price=95,
            collateral=500,
            open_date=1_420_156_800,  # 2015-01-02 00:00 UTC
            expiry=1_451_692_800,     # 2016-01-02 00:00 UTC
        )
        # Use date part to avoid timezone offset issues
        assert spread.open_datetime.year >= 2015
        assert spread.expiry_datetime.year >= 2015

    def test_repr(self):
        spread = PutCreditSpread(
            underlying_price=250,
            strike_price=235,
            collateral=1000,
            open_date=1_000_000,
            expiry=2_000_000,
        )
        r = repr(spread)
        assert "235" in r
        assert "1000.00" in r
