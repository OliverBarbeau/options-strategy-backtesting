"""Unit tests for tradelab.utils."""

import pytest

from tradelab.utils import (
    reverse_interest,
    compound_interest,
    days_to_trading_days,
    calc_fee_rate,
    get_leverage,
)


# --- reverse_interest ---

class TestReverseInterest:
    def test_doubling_in_10_years(self):
        rate = reverse_interest(10, 1, 2)
        assert 0.071 < rate < 0.073  # ~7.177%

    def test_no_growth(self):
        assert reverse_interest(5, 100, 100) == 0.0

    def test_loss(self):
        rate = reverse_interest(1, 100, 50)
        assert rate == pytest.approx(-0.5)

    def test_zero_years_returns_zero(self):
        assert reverse_interest(0, 100, 200) == 0.0

    def test_zero_principal_returns_zero(self):
        assert reverse_interest(5, 0, 200) == 0.0


# --- compound_interest ---

class TestCompoundInterest:
    def test_single_period(self):
        assert compound_interest(1000, 0.10, 1) == pytest.approx(1100.0)

    def test_multiple_periods(self):
        result = compound_interest(1000, 0.10, 3)
        assert result == pytest.approx(1000 * 1.1**3)

    def test_zero_periods(self):
        assert compound_interest(500, 0.10, 0) == pytest.approx(500.0)

    def test_negative_rate(self):
        result = compound_interest(1000, -0.05, 2)
        assert result == pytest.approx(1000 * 0.95**2)


# --- days_to_trading_days ---

class TestDaysToTradingDays:
    def test_one_year(self):
        assert days_to_trading_days(365) == 252

    def test_one_week(self):
        result = days_to_trading_days(7)
        assert result == 4  # floor(7 * 252/365) = floor(4.83) = 4

    def test_zero(self):
        assert days_to_trading_days(0) == 0


# --- calc_fee_rate ---

class TestCalcFeeRate:
    def test_base_rate_at_zero_volume(self):
        assert calc_fee_rate(0) == 0.0026

    def test_first_tier(self):
        assert calc_fee_rate(50_001) == pytest.approx(0.0024)

    def test_max_tier(self):
        rate = calc_fee_rate(10_000_001)
        assert rate == pytest.approx(0.001, abs=1e-9)

    def test_never_goes_negative(self):
        assert calc_fee_rate(999_999_999_999) >= 0.0

    def test_custom_base_rate(self):
        assert calc_fee_rate(0, base_rate=0.005) == 0.005


# --- get_leverage ---

class TestGetLeverage:
    def test_minimum_leverage(self):
        # Nearly equal predictions -> base leverage
        assert get_leverage([0.50, 0.50], 0.50) == 2

    def test_high_confidence_prediction(self):
        lev = get_leverage([0.10, 0.90], 0.50)
        assert lev >= 3

    def test_respects_max(self):
        lev = get_leverage([0.0, 1.0], 0.50, max_leverage=5)
        assert lev <= 5

    def test_custom_base(self):
        lev = get_leverage([0.50, 0.50], 0.50, base_leverage=3)
        assert lev >= 3
