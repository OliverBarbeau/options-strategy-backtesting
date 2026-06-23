"""Tests for tradelab.options (Black-Scholes pricing)."""

import numpy as np
import pandas as pd
import pytest

from tradelab.options import (
    bs_put_price,
    bs_call_price,
    bs_greeks,
    historical_volatility,
    ewm_volatility,
    put_credit_spread_price,
    price_spread_series,
)


# ------------------------------------------------------------------
# Black-Scholes pricing
# ------------------------------------------------------------------


class TestBSPricing:
    """Validate against known textbook values."""

    def test_atm_put_is_positive(self):
        price = bs_put_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert price > 0

    def test_atm_call_is_positive(self):
        price = bs_call_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert price > 0

    def test_put_call_parity(self):
        """C - P = S - K*e^(-rT)"""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
        C = bs_call_price(S, K, T, r, sigma)
        P = bs_put_price(S, K, T, r, sigma)
        parity_rhs = S - K * np.exp(-r * T)
        assert C - P == pytest.approx(parity_rhs, abs=1e-10)

    def test_deep_otm_put_near_zero(self):
        # Strike way below price
        price = bs_put_price(S=100, K=50, T=0.1, r=0.05, sigma=0.20)
        assert price < 0.01

    def test_deep_itm_put_near_intrinsic(self):
        # Strike way above price
        price = bs_put_price(S=100, K=150, T=0.1, r=0.05, sigma=0.20)
        intrinsic = 150 * np.exp(-0.05 * 0.1) - 100
        assert price == pytest.approx(intrinsic, abs=1.0)

    def test_longer_expiry_higher_premium(self):
        p_short = bs_put_price(S=100, K=95, T=0.1, r=0.05, sigma=0.20)
        p_long = bs_put_price(S=100, K=95, T=1.0, r=0.05, sigma=0.20)
        assert p_long > p_short

    def test_higher_vol_higher_premium(self):
        p_low = bs_put_price(S=100, K=95, T=0.5, r=0.05, sigma=0.10)
        p_high = bs_put_price(S=100, K=95, T=0.5, r=0.05, sigma=0.40)
        assert p_high > p_low

    def test_vectorized(self):
        S = np.array([100, 200, 300])
        K = np.array([95, 190, 285])
        prices = bs_put_price(S, K, T=0.5, r=0.05, sigma=0.20)
        assert len(prices) == 3
        assert all(p > 0 for p in prices)

    def test_known_value(self):
        """S=100, K=100, T=1, r=5%, vol=20% -> put ~5.57"""
        price = bs_put_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert price == pytest.approx(5.57, abs=0.1)


# ------------------------------------------------------------------
# Greeks
# ------------------------------------------------------------------


class TestGreeks:
    def test_put_delta_negative(self):
        g = bs_greeks(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="put")
        assert g["delta"] < 0

    def test_call_delta_positive(self):
        g = bs_greeks(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
        assert g["delta"] > 0

    def test_gamma_positive(self):
        g = bs_greeks(S=100, K=100, T=0.5, r=0.05, sigma=0.20)
        assert g["gamma"] > 0

    def test_theta_negative_for_put(self):
        g = bs_greeks(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="put")
        # Theta is typically negative (time decay), though deep ITM puts
        # near expiry can have positive theta due to interest. For ATM, negative.
        assert g["theta"] < 0

    def test_vega_positive(self):
        g = bs_greeks(S=100, K=100, T=0.5, r=0.05, sigma=0.20)
        assert g["vega"] > 0

    def test_otm_put_delta_small(self):
        g = bs_greeks(S=100, K=80, T=0.1, r=0.05, sigma=0.20, option_type="put")
        assert abs(g["delta"]) < 0.1

    def test_greeks_returns_all_keys(self):
        g = bs_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert set(g.keys()) == {"delta", "gamma", "theta", "vega", "rho"}


# ------------------------------------------------------------------
# Volatility estimation
# ------------------------------------------------------------------


class TestVolatility:
    def test_historical_vol_returns_series(self):
        np.random.seed(42)
        prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0, 0.01, 100)))
        vol = historical_volatility(prices, window=20)
        assert isinstance(vol, pd.Series)
        assert len(vol) == 100
        # First 20 values should be NaN (not enough window)
        assert vol.iloc[:20].isna().all()
        # Remaining should be positive
        assert (vol.dropna() > 0).all()

    def test_vol_magnitude_reasonable(self):
        """For ~1% daily returns, annualized vol should be ~16%."""
        np.random.seed(42)
        prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0, 0.01, 252)))
        vol = historical_volatility(prices, window=30)
        median_vol = vol.dropna().median()
        assert 0.05 < median_vol < 0.40

    def test_ewm_vol(self):
        np.random.seed(42)
        prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0, 0.01, 100)))
        vol = ewm_volatility(prices, span=20)
        assert isinstance(vol, pd.Series)
        assert (vol.dropna() > 0).all()


# ------------------------------------------------------------------
# Spread pricing
# ------------------------------------------------------------------


class TestSpreadPricing:
    def test_net_credit_positive(self):
        """Selling higher strike and buying lower -> net credit."""
        result = put_credit_spread_price(
            S=150, K_short=145, K_long=140, T=30 / 365, r=0.05, sigma=0.20
        )
        assert result["net_credit"] > 0
        assert result["short_premium"] > result["long_premium"]

    def test_max_loss_bounded(self):
        result = put_credit_spread_price(
            S=150, K_short=145, K_long=140, T=30 / 365, r=0.05, sigma=0.20
        )
        assert result["max_loss"] < result["spread_width"]
        assert result["max_loss"] > 0

    def test_credit_potential_reasonable(self):
        result = put_credit_spread_price(
            S=150, K_short=145, K_long=140, T=30 / 365, r=0.05, sigma=0.20
        )
        # Credit potential (return on risk) should be between 0% and 100%
        assert 0 < result["credit_potential"] < 1.0

    def test_wider_spread_more_credit(self):
        narrow = put_credit_spread_price(
            S=150, K_short=145, K_long=143, T=30 / 365, r=0.05, sigma=0.20
        )
        wide = put_credit_spread_price(
            S=150, K_short=145, K_long=135, T=30 / 365, r=0.05, sigma=0.20
        )
        assert wide["net_credit"] > narrow["net_credit"]

    def test_higher_vol_more_credit(self):
        low_vol = put_credit_spread_price(
            S=150, K_short=145, K_long=140, T=30 / 365, r=0.05, sigma=0.15
        )
        high_vol = put_credit_spread_price(
            S=150, K_short=145, K_long=140, T=30 / 365, r=0.05, sigma=0.35
        )
        assert high_vol["net_credit"] > low_vol["net_credit"]

    def test_matches_real_trade_ballpark(self):
        """Compare with a historical trade:
        AAPL at ~$135, put credit spread 132/130, ~11 DTE,
        observed credit ~$0.13/share ($13.10 per contract).
        """
        result = put_credit_spread_price(
            S=135, K_short=132, K_long=130,
            T=11 / 365, r=0.05, sigma=0.25,
        )
        # B-S won't match exactly (IV != HV, skew, etc.)
        # but should be in the right order of magnitude
        assert 0.01 < result["net_credit"] < 1.50


class TestPriceSpreadSeries:
    def test_output_shape(self, stock_df):
        df = stock_df.copy()
        df = df.rename(columns={"c": "close"})
        result = price_spread_series(
            df, strike_buffer=0.05, spread_width=5.0,
            days_to_expiry=30, vol_window=30,
        )
        assert "net_credit" in result.columns
        assert "credit_potential" in result.columns
        assert "sigma" in result.columns
        # Should have fewer rows than input (vol window drops first N)
        assert len(result) < len(df)
        assert len(result) > 0

    def test_credit_potential_positive(self, stock_df):
        df = stock_df.copy()
        df = df.rename(columns={"c": "close"})
        result = price_spread_series(
            df, strike_buffer=0.05, spread_width=5.0, days_to_expiry=30,
        )
        assert (result["credit_potential"] > 0).all()
