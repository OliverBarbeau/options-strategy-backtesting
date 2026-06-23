"""Tests for pricing provider abstraction."""

import pytest

from tradelab.pricing import (
    MockProvider,
    BlackScholesProvider,
    OptionQuote,
    SpreadQuote,
    PricingError,
)
from tradelab.pricing.base import PricingProvider
from tradelab.pricing.calibration import Calibrator, CalibrationReport


class TestMockProvider:
    """Mock provider gives deterministic results without network."""

    def test_single_option_quote(self):
        provider = MockProvider(default_underlying=100.0)
        q = provider.get_option_quote(
            ticker="TEST",
            strike=95.0,
            expiry="2026-05-01",
            put_call="P",
            date="2026-04-01",
        )
        assert q.ticker == "TEST"
        assert q.strike == 95.0
        assert q.put_call == "P"
        assert q.source == "mock"
        assert q.mid > 0
        assert q.bid < q.mid < q.ask
        assert q.underlying_price == 100.0

    def test_spread_quote(self):
        provider = MockProvider(default_underlying=100.0)
        spread = provider.get_spread_quote(
            ticker="TEST",
            short_strike=95.0,
            long_strike=92.0,
            expiry="2026-05-01",
            date="2026-04-01",
        )
        assert spread.ticker == "TEST"
        assert spread.short_strike == 95.0
        assert spread.long_strike == 92.0
        assert spread.spread_width == 300.0  # (95-92) * 100
        # Net credit at mid should be positive (short mid > long mid because short strike closer to ATM)
        assert spread.net_credit_mid > 0
        assert spread.max_loss > 0
        assert spread.short_quote is not None
        assert spread.long_quote is not None
        # Short put should have higher mid than long put (closer to ATM)
        assert spread.short_quote.mid > spread.long_quote.mid

    def test_dte_calculation(self):
        provider = MockProvider()
        spread = provider.get_spread_quote(
            ticker="TEST",
            short_strike=95.0,
            long_strike=92.0,
            expiry="2026-05-01",
            date="2026-04-01",
        )
        assert spread.dte == 30  # April 1 -> May 1

    def test_deterministic(self):
        """Same inputs should give same outputs."""
        p1 = MockProvider()
        p2 = MockProvider()
        q1 = p1.get_option_quote("TEST", 95, "2026-05-01", "P", "2026-04-01", 100)
        q2 = p2.get_option_quote("TEST", 95, "2026-05-01", "P", "2026-04-01", 100)
        assert q1.mid == q2.mid
        assert q1.bid == q2.bid

    def test_credit_multiplier(self):
        """Credit multiplier scales the pricing."""
        p1 = MockProvider(credit_multiplier=1.0)
        p2 = MockProvider(credit_multiplier=1.5)
        q1 = p1.get_option_quote("TEST", 95, "2026-05-01", "P", "2026-04-01", 100)
        q2 = p2.get_option_quote("TEST", 95, "2026-05-01", "P", "2026-04-01", 100)
        assert q2.mid > q1.mid


class TestBlackScholesProvider:
    """BlackScholesProvider uses our existing options.py functions."""

    def test_name(self):
        assert BlackScholesProvider().name == "blackscholes"

    def test_supports_greeks(self):
        assert BlackScholesProvider().supports_greeks() is True

    def test_supports_historical(self):
        assert BlackScholesProvider().supports_historical() is True


class TestProviderInterface:
    """All providers implement the same interface."""

    def test_all_providers_are_subclasses(self):
        assert issubclass(MockProvider, PricingProvider)
        assert issubclass(BlackScholesProvider, PricingProvider)

    def test_required_methods(self):
        mock = MockProvider()
        assert hasattr(mock, "get_option_quote")
        assert hasattr(mock, "get_spread_quote")
        assert hasattr(mock, "find_spread_strikes")

    def test_find_spread_strikes_default_impl(self):
        provider = MockProvider(default_underlying=150.0)
        spread = provider.find_spread_strikes(
            ticker="TEST",
            date="2026-04-01",
            buffer=0.10,
            spread_pct=0.02,
            dte_target=30,
            underlying_price=150.0,
        )
        assert spread is not None
        # Short strike should be ~10% below, long strike ~12% below
        assert spread.short_strike == pytest.approx(135.0, abs=1)
        assert spread.long_strike == pytest.approx(132.0, abs=1)


class TestCalibrator:
    """Calibration compares two providers."""

    def test_compare_identical_providers(self):
        """Two mock providers with same config should have near-zero bias."""
        p1 = MockProvider(default_underlying=100.0)
        p2 = MockProvider(default_underlying=100.0)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            log_path = f.name

        cal = Calibrator(baseline=p1, reference=p2, log_path=log_path)
        entry = cal.compare_spread(
            ticker="TEST",
            short_strike=95,
            long_strike=92,
            expiry="2026-05-01",
            date="2026-04-01",
            underlying_price=100.0,
        )
        assert entry is not None
        assert abs(entry.credit_bias_pct) < 0.001

    def test_compare_biased_providers(self):
        """Provider with credit_multiplier=1.5 should show +50% bias."""
        baseline = MockProvider(default_underlying=100.0, credit_multiplier=1.0)
        reference = MockProvider(default_underlying=100.0, credit_multiplier=1.5)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            log_path = f.name

        cal = Calibrator(baseline=baseline, reference=reference, log_path=log_path)
        entry = cal.compare_spread(
            ticker="TEST",
            short_strike=95,
            long_strike=92,
            expiry="2026-05-01",
            date="2026-04-01",
            underlying_price=100.0,
        )
        assert entry is not None
        # Reference credit should be ~50% higher than baseline
        assert entry.credit_bias_pct > 0.3

    def test_calibration_report_empty(self):
        report = CalibrationReport([])
        assert len(report) == 0
        assert report.mean_credit_bias() == 0
        assert report.correction_factor() == 1.0

    def test_correction_factor_inverse_of_bias(self):
        """If bias is +20%, correction factor should be 1/1.2 ~= 0.833."""
        from tradelab.pricing.calibration import CalibrationEntry
        entries = []
        for i in range(5):
            entries.append(CalibrationEntry(
                timestamp="2026-04-01T00:00:00",
                ticker="TEST",
                date="2026-04-01",
                expiry="2026-05-01",
                dte=30,
                short_strike=95,
                long_strike=92,
                underlying_price=100,
                baseline_source="mock",
                baseline_net_credit=100,
                baseline_short_mid=2.0,
                baseline_long_mid=1.0,
                reference_source="mock",
                reference_net_credit=120,  # 20% higher
                reference_short_bid=2.3,
                reference_short_ask=2.4,
                reference_long_bid=1.1,
                reference_long_ask=1.2,
                credit_bias_pct=0.20,
                short_bias_pct=0.175,
                long_bias_pct=0.15,
                reference_short_spread_pct=0.04,
                reference_long_spread_pct=0.09,
            ))
        report = CalibrationReport(entries)
        assert report.median_credit_bias() == pytest.approx(0.20)
        assert report.correction_factor() == pytest.approx(1 / 1.20, abs=0.001)

    def test_report_summary_output(self):
        from tradelab.pricing.calibration import CalibrationEntry
        entry = CalibrationEntry(
            timestamp="2026-04-01T00:00:00",
            ticker="AAPL", date="2026-04-01", expiry="2026-05-01", dte=30,
            short_strike=150, long_strike=145, underlying_price=170,
            baseline_source="blackscholes", baseline_net_credit=50,
            baseline_short_mid=1.0, baseline_long_mid=0.5,
            reference_source="thetadata", reference_net_credit=55,
            reference_short_bid=1.05, reference_short_ask=1.15,
            reference_long_bid=0.50, reference_long_ask=0.55,
            credit_bias_pct=0.10, short_bias_pct=0.10, long_bias_pct=0.05,
            reference_short_spread_pct=0.09, reference_long_spread_pct=0.10,
        )
        report = CalibrationReport([entry])
        text = report.summary()
        assert "blackscholes" in text
        assert "thetadata" in text
        assert "Credit bias" in text
        assert "Correction factor" in text
