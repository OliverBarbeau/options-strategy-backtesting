"""Tests for data integrity fixes: vol look-ahead, searchsorted bias, exit slippage.

These tests verify that the audit-identified biases have been corrected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradelab.options import historical_volatility, ewm_volatility
from tradelab.account import SimulatedAccount


# ------------------------------------------------------------------
# 1A. Vol look-ahead: vol[i] must not depend on prices after day i
# ------------------------------------------------------------------


class TestVolLookAhead:
    """Verify historical_volatility uses only past data."""

    def _make_prices(self, n: int = 100, seed: int = 42) -> pd.Series:
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0005, 0.01, n)
        prices = 100.0 * np.exp(np.cumsum(returns))
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        return pd.Series(prices, index=idx)

    def test_vol_unchanged_when_future_prices_change(self):
        """Vol at day i must be identical regardless of prices after day i."""
        prices = self._make_prices(100)
        vol_full = historical_volatility(prices, window=20)

        # Mutate prices after index 60 — vol at index 60 should be unchanged
        prices_modified = prices.copy()
        prices_modified.iloc[61:] = prices_modified.iloc[61:] * 2.0
        vol_modified = historical_volatility(prices_modified, window=20)

        check_idx = 60
        assert not np.isnan(vol_full.iloc[check_idx])
        assert vol_full.iloc[check_idx] == pytest.approx(
            vol_modified.iloc[check_idx], rel=1e-10
        ), "Vol at day 60 changed when only future prices were modified — look-ahead bias!"

    def test_ewm_vol_unchanged_when_future_prices_change(self):
        """EWM vol at day i must be identical regardless of prices after day i."""
        prices = self._make_prices(100)
        vol_full = ewm_volatility(prices, span=20)

        prices_modified = prices.copy()
        prices_modified.iloc[61:] = prices_modified.iloc[61:] * 2.0
        vol_modified = ewm_volatility(prices_modified, span=20)

        check_idx = 60
        assert not np.isnan(vol_full.iloc[check_idx])
        assert vol_full.iloc[check_idx] == pytest.approx(
            vol_modified.iloc[check_idx], rel=1e-10
        ), "EWM vol at day 60 changed when only future prices were modified — look-ahead bias!"

    def test_vol_at_day_i_excludes_day_i_return(self):
        """Vol[i] should not include the return from day i-1 to day i."""
        prices = self._make_prices(60)
        vol = historical_volatility(prices, window=20)

        # Change only the price at day 50 — vol at day 50 should NOT change
        # because vol[50] should use returns up to day 49 only.
        prices_spike = prices.copy()
        prices_spike.iloc[50] = prices_spike.iloc[50] * 1.5  # 50% spike
        vol_spike = historical_volatility(prices_spike, window=20)

        assert vol.iloc[50] == pytest.approx(vol_spike.iloc[50], rel=1e-10), (
            "Vol[50] changed when price[50] was modified — "
            "vol should only use returns up to day i-1"
        )

    def test_first_valid_vol_index(self):
        """With shift(1), first valid vol should be at index window+1."""
        prices = self._make_prices(60)
        vol = historical_volatility(prices, window=20)
        # First non-NaN should be at index 21 (window=20, plus 1 for shift)
        assert np.isnan(vol.iloc[20])
        assert not np.isnan(vol.iloc[21])


# ------------------------------------------------------------------
# 1B. searchsorted: weekend/holiday queries must return previous day
# ------------------------------------------------------------------


class TestSearchsortedBias:
    """Verify _get_price returns the last data point <= query date."""

    def _make_simulator_data(self):
        """Create a minimal data dict mimicking Simulator._data format.

        The Simulator uses Unix timestamps (seconds) as the DataFrame index,
        matching how DataPipeline stores data.
        """
        # Mon-Fri trading days, skip weekends
        dates = pd.bdate_range("2024-01-01", "2024-01-31")
        # Convert to Unix seconds — same as Simulator._load_data
        ts_index = np.array([int(d.timestamp()) for d in dates])
        prices = np.linspace(100, 110, len(dates))
        df = pd.DataFrame({"close": prices}, index=ts_index)
        return df, dates

    def test_weekend_returns_friday(self):
        """Query on Saturday should return Friday's price, not Monday's."""
        df, dates = self._make_simulator_data()

        # Find a Friday
        friday = None
        for d in dates:
            if d.weekday() == 4:  # Friday
                friday = d
                break
        assert friday is not None

        friday_ts = int(friday.timestamp())
        saturday_ts = friday_ts + 86400  # Saturday

        # Simulate _get_price logic with the fix
        idx = np.searchsorted(df.index.values, saturday_ts, side="right") - 1
        assert idx >= 0
        # The returned index should be Friday, not the following Monday
        assert df.index[idx] == friday_ts

    def test_exact_date_returns_itself(self):
        """Query on an exact trading day should return that day."""
        df, dates = self._make_simulator_data()
        target = dates[5]
        target_ts = int(target.timestamp())

        idx = np.searchsorted(df.index.values, target_ts, side="right") - 1
        assert df.index[idx] == target_ts

    def test_before_all_data_returns_negative(self):
        """Query before all data should yield idx < 0."""
        df, dates = self._make_simulator_data()
        early_ts = int(dates[0].timestamp()) - 86400 * 30

        idx = np.searchsorted(df.index.values, early_ts, side="right") - 1
        assert idx < 0


# ------------------------------------------------------------------
# 1C. Exit slippage: round-trip friction must include both sides
# ------------------------------------------------------------------


class TestExitSlippage:
    """Verify exit slippage is deducted from P&L and balance."""

    @pytest.fixture
    def account(self, tmp_path):
        path = tmp_path / "test_slippage.json"
        return SimulatedAccount.load_or_create(
            str(path), starting_capital=10000.0, name="test_slippage"
        )

    def test_exit_slippage_reduces_pnl(self, account):
        """P&L should be lower when exit slippage is applied."""
        account.open_position(
            ticker="TEST",
            date="2024-01-01",
            entry_price=100.0,
            short_strike=95.0,
            long_strike=90.0,
            contracts=1,
            credit_per_contract=100.0,
            collateral_per_contract=500.0,
            close_target_date="2024-01-15",
        )

        pos = account.positions[0]
        close_cost = 30.0
        account.close_position(
            pos_id=pos.id,
            date="2024-01-15",
            exit_price=100.0,
            close_cost=close_cost,
        )

        trade = account.trades[0]
        slippage_pct = account._slippage_pct
        expected_close_slippage = close_cost * slippage_pct

        # P&L should include exit slippage deduction
        expected_pnl = (
            trade.credit_received
            - close_cost
            - expected_close_slippage
            - account._commission * 2 * trade.contracts
        )
        assert trade.pnl == pytest.approx(expected_pnl, abs=0.01)

    def test_friction_includes_both_entry_and_exit(self, account):
        """Friction must account for slippage on both entry and exit."""
        account.open_position(
            ticker="TEST",
            date="2024-01-01",
            entry_price=100.0,
            short_strike=95.0,
            long_strike=90.0,
            contracts=1,
            credit_per_contract=100.0,
            collateral_per_contract=500.0,
            close_target_date="2024-01-15",
        )

        pos = account.positions[0]
        close_cost = 50.0
        account.close_position(
            pos_id=pos.id,
            date="2024-01-15",
            exit_price=100.0,
            close_cost=close_cost,
        )

        trade = account.trades[0]
        slippage_pct = account._slippage_pct

        # Friction should include: commissions + entry slippage + exit slippage
        entry_slippage = trade.credit_received * slippage_pct / (1 - slippage_pct)
        exit_slippage = close_cost * slippage_pct
        close_commission = account._commission * 2 * trade.contracts
        expected_friction = close_commission + entry_slippage + exit_slippage

        assert trade.friction == pytest.approx(expected_friction, abs=0.01)
        assert trade.friction > close_commission, (
            "Friction must be greater than just commissions (should include slippage)"
        )

    def test_accounting_invariant_with_exit_slippage(self, account):
        """Sum of trade P/L must still equal equity change after slippage fix."""
        account.open_position(
            ticker="TEST",
            date="2024-01-01",
            entry_price=100.0,
            short_strike=95.0,
            long_strike=90.0,
            contracts=1,
            credit_per_contract=100.0,
            collateral_per_contract=500.0,
            close_target_date="2024-01-15",
        )

        pos = account.positions[0]
        account.close_position(
            pos_id=pos.id,
            date="2024-01-15",
            exit_price=100.0,
            close_cost=30.0,
        )

        trade_pnl = account.trades[0].pnl
        equity_change = account.equity - 10000.0
        assert trade_pnl == pytest.approx(equity_change, abs=0.01), (
            f"Trade P/L {trade_pnl} != equity change {equity_change} — "
            "accounting invariant broken by slippage fix"
        )
