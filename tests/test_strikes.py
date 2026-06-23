"""Tests for strike snapping utilities."""

import pytest

from tradelab.pricing.strikes import (
    strike_increment,
    snap_to_increment,
    snap_put_credit_spread,
    effective_buffer,
    DENSE_STRIKE_TICKERS,
)


class TestStrikeIncrement:
    def test_dense_ticker_always_one_dollar(self):
        assert strike_increment("SPY", 500) == 1.0
        assert strike_increment("NVDA", 177) == 1.0
        assert strike_increment("META", 574) == 1.0
        assert strike_increment("AAPL", 250) == 1.0

    def test_low_price_half_dollar(self):
        assert strike_increment("XYZ", 15) == 0.50

    def test_mid_price_one_dollar(self):
        assert strike_increment("XYZ", 100) == 1.0
        assert strike_increment("XYZ", 150) == 1.0

    def test_high_price_two_fifty(self):
        assert strike_increment("XYZ", 300) == 2.50

    def test_very_high_price_five_dollar(self):
        assert strike_increment("XYZ", 800) == 5.0


class TestSnapToIncrement:
    def test_snap_nearest_one_dollar(self):
        assert snap_to_increment(159.651, 1.0) == 160.0
        assert snap_to_increment(159.4, 1.0) == 159.0
        assert snap_to_increment(159.5, 1.0) == 160.0  # round half up

    def test_snap_down(self):
        assert snap_to_increment(159.9, 1.0, mode="down") == 159.0
        assert snap_to_increment(159.1, 1.0, mode="down") == 159.0

    def test_snap_up(self):
        assert snap_to_increment(159.1, 1.0, mode="up") == 160.0
        assert snap_to_increment(159.9, 1.0, mode="up") == 160.0

    def test_snap_half_dollar(self):
        assert snap_to_increment(15.3, 0.5) == 15.5
        assert snap_to_increment(15.2, 0.5) == 15.0

    def test_snap_two_fifty(self):
        assert snap_to_increment(283.1, 2.5) == 282.5
        assert snap_to_increment(284.0, 2.5) == 285.0


class TestSnapPutCreditSpread:
    """The main entry point for realistic strike selection."""

    def test_nvda_at_177(self):
        """NVDA at $177, 10% buffer = target 159.30, 2% width."""
        sk, lk = snap_put_credit_spread("NVDA", 177.39, 0.10, target_spread_pct=0.02)
        # $1 increments, snap DOWN from 159.651 to 159
        assert sk == 159.0
        # spread width rounded: 177.39 * 0.02 = 3.55, rounds to 4 at $1 inc
        # long strike = 159 - 4 = 155
        assert lk == 155.0

    def test_meta_at_574(self):
        """META is in DENSE_STRIKE_TICKERS, so $1 increments."""
        sk, lk = snap_put_credit_spread("META", 574.46, 0.10, target_spread_pct=0.02)
        # target short = 517.014, snap down = 517
        assert sk == 517.0
        # spread = 574.46 * 0.02 = 11.49, rounds to 11
        assert lk == 506.0

    def test_spy_dense(self):
        """SPY always has $1 increments."""
        sk, lk = snap_put_credit_spread("SPY", 655.83, 0.10, target_spread_pct=0.02)
        # target short = 590.247, snap down = 590
        assert sk == 590.0
        # spread = 655.83 * 0.02 = 13.12, rounds to 13
        assert lk == 577.0

    def test_low_priced_half_dollar(self):
        """Stocks under $25 use $0.50 increments."""
        sk, lk = snap_put_credit_spread("LOWPRICE", 20.0, 0.10, target_spread_pct=0.02)
        # target short = 18.0, already on 0.50 increment
        assert sk == 18.0
        # spread = 0.40, rounds to minimum 0.50
        assert lk == 17.5

    def test_buffer_always_conservative(self):
        """Snapping DOWN means actual buffer >= target buffer."""
        for ticker, price, buf in [
            ("NVDA", 177.39, 0.10),
            ("META", 574.46, 0.10),
            ("AAPL", 255.92, 0.08),
            ("SPY", 655.83, 0.05),
        ]:
            sk, lk = snap_put_credit_spread(ticker, price, buf)
            actual_buffer = effective_buffer(price, sk)
            assert actual_buffer >= buf, (
                f"{ticker}: snapped buffer {actual_buffer:.3%} should be >= target {buf:.3%}"
            )

    def test_non_dense_high_price(self):
        """Non-dense ticker at high price uses $2.50 increments."""
        sk, lk = snap_put_credit_spread("RANDOM", 350, 0.10, target_spread_pct=0.02)
        # Target short = 315, already on 2.5 increment (315 / 2.5 = 126)
        assert sk == 315.0
        # spread = 7, rounds to nearest 2.5 = 7.5
        assert lk == 307.5

    def test_spread_width_minimum(self):
        """Spread width is at least one increment."""
        sk, lk = snap_put_credit_spread("NVDA", 50, 0.10, target_spread_pct=0.001)
        # Tiny target spread, should floor at 1 increment = $1
        assert sk - lk >= 1.0


class TestEffectiveBuffer:
    def test_basic(self):
        assert effective_buffer(100, 90) == pytest.approx(0.10)

    def test_exact_ten_percent(self):
        assert effective_buffer(200, 180) == pytest.approx(0.10)

    def test_zero_price(self):
        assert effective_buffer(0, 90) == 0.0


class TestSimulatorIntegration:
    """Verify the Simulator picks realistic strikes, not fractional."""

    def test_simulator_uses_snapped_strikes(self):
        """Directly check the utility that simulator calls."""
        # This is what the simulator does internally
        sk, lk = snap_put_credit_spread("NVDA", 177.39, 0.10, target_spread_pct=0.02)

        # Strikes should be whole dollars for NVDA
        assert sk == int(sk), f"Short strike {sk} should be a whole dollar"
        assert lk == int(lk), f"Long strike {lk} should be a whole dollar"
        # And within a reasonable range
        assert 155 <= sk <= 162
        assert 150 <= lk <= 160
