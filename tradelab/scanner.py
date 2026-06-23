"""Market scanner: find tickers that fit the put credit spread strategy.

Screens the broad market for stocks meeting our research-backed criteria:
- HV30 between 25-50% (sweet spot for premium vs breach rate)
- Price $50-$500 (feasible collateral for small accounts)
- Average daily volume > 1M (liquid options)
- Not in financials/energy (tail risk too severe)
- No upcoming earnings within the hold period
- Historically rare 10%+ drops in 14 days
- Currently in a pullback (optional, for immediate entry)

Usage::

    scanner = StrategyScanner()
    candidates = scanner.scan()
    for c in candidates:
        print(c)

    # Backtest all candidates
    results = scanner.backtest_candidates(candidates, start="2020-01-01")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from tradelab.pipeline import DataPipeline
from tradelab.options import (
    historical_volatility,
    put_credit_spread_price,
)

logger = logging.getLogger(__name__)

# Broad universe to scan from -- large/mid cap US equities with active options
SCAN_UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOG", "META", "AMZN", "NVDA", "AMD", "AVGO",
    "CRM", "ADBE", "ORCL", "INTC", "QCOM", "MU", "ANET", "PANW",
    "NOW", "SNOW", "SHOP", "SQ", "UBER", "ABNB", "DDOG", "NET",
    "PLTR", "MRVL", "LRCX", "KLAC", "SNPS", "CDNS",
    # Industrials
    "CAT", "DE", "HON", "UNP", "GE", "RTX", "LMT", "BA",
    "MMM", "EMR", "ITW", "FDX", "UPS",
    # Consumer
    "WMT", "COST", "HD", "LOW", "TGT", "NKE", "SBUX", "MCD",
    "PG", "KO", "PEP",
    # Healthcare
    "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT",
    "ISRG", "DXCM", "MRNA",
    # Financials (included for scanning but flagged)
    "JPM", "GS", "BAC", "V", "MA", "AXP",
    # Energy (included for scanning but flagged)
    "XOM", "CVX", "COP", "SLB",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE",
]

# Sectors to flag as higher risk
HIGH_RISK_SECTORS = {"Financial", "Energy"}

# Rough sector mapping
SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOG": "Tech", "META": "Tech",
    "AMZN": "Tech", "NVDA": "Semi", "AMD": "Semi", "AVGO": "Semi",
    "CRM": "Tech", "ADBE": "Tech", "ORCL": "Tech", "INTC": "Semi",
    "QCOM": "Semi", "MU": "Semi", "ANET": "Tech", "PANW": "Tech",
    "NOW": "Tech", "SNOW": "Tech", "SHOP": "Tech", "SQ": "Tech",
    "UBER": "Tech", "ABNB": "Tech", "DDOG": "Tech", "NET": "Tech",
    "PLTR": "Tech", "MRVL": "Semi", "LRCX": "Semi", "KLAC": "Semi",
    "SNPS": "Tech", "CDNS": "Tech",
    "CAT": "Industrial", "DE": "Industrial", "HON": "Industrial",
    "UNP": "Industrial", "GE": "Industrial", "RTX": "Industrial",
    "LMT": "Industrial", "BA": "Industrial", "MMM": "Industrial",
    "EMR": "Industrial", "ITW": "Industrial", "FDX": "Industrial",
    "UPS": "Industrial",
    "WMT": "Consumer", "COST": "Consumer", "HD": "Consumer",
    "LOW": "Consumer", "TGT": "Consumer", "NKE": "Consumer",
    "SBUX": "Consumer", "MCD": "Consumer", "PG": "Consumer",
    "KO": "Consumer", "PEP": "Consumer",
    "JNJ": "Healthcare", "UNH": "Healthcare", "LLY": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "PFE": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "ISRG": "Healthcare",
    "DXCM": "Healthcare", "MRNA": "Healthcare",
    "JPM": "Financial", "GS": "Financial", "BAC": "Financial",
    "V": "Financial", "MA": "Financial", "AXP": "Financial",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "XLK": "ETF", "XLF": "ETF", "XLE": "ETF",
}


def validate_universe(
    tickers: list[str],
    start_date: str,
    pipe: DataPipeline | None = None,
    min_history_days: int = 60,
) -> tuple[list[str], list[str]]:
    """Filter tickers to those with data available before start_date.

    Checks that each ticker has price history beginning at least
    ``min_history_days`` before ``start_date``. This mitigates survivorship
    bias by excluding tickers that IPO'd after the backtest start.

    Args:
        tickers: List of ticker symbols to validate.
        start_date: Backtest start date (YYYY-MM-DD).
        pipe: DataPipeline instance (created if None).
        min_history_days: Required days of data before start_date.

    Returns:
        (valid, excluded): Lists of valid and excluded tickers.
    """
    if pipe is None:
        pipe = DataPipeline()

    start_dt = datetime.fromisoformat(start_date)
    required_start = (start_dt - timedelta(days=min_history_days)).strftime("%Y-%m-%d")

    valid = []
    excluded = []

    for ticker in tickers:
        try:
            df = pipe.fetch_stock(ticker, start="2010-01-01", end=start_date)
            if df is None or len(df) == 0:
                excluded.append(ticker)
                continue
            # Check first available date is early enough
            first_ts = df.index[0]
            first_date = pd.Timestamp(first_ts, unit="s").strftime("%Y-%m-%d")
            if first_date > required_start:
                logger.info(
                    f"{ticker}: first data {first_date} is after required "
                    f"{required_start} — excluding (likely IPO'd too recently)"
                )
                excluded.append(ticker)
            else:
                valid.append(ticker)
        except Exception as e:
            logger.warning(f"{ticker}: failed to fetch data — {e}")
            excluded.append(ticker)

    if excluded:
        logger.info(
            f"Universe validation: {len(valid)} valid, {len(excluded)} excluded "
            f"({', '.join(excluded)})"
        )

    return valid, excluded


@dataclass
class ScanResult:
    """Result of scanning a single ticker."""
    ticker: str
    sector: str
    price: float
    hv30: float
    hv30_percentile: float  # vs own history
    adv_20: float  # 20-day average daily volume
    pullback_pct: float  # drawdown from 20-day high
    credit_potential: float  # estimated credit/risk ratio
    breach_rate: float  # historical % of 10%+ drops in 14 days
    has_earnings_soon: bool
    score: float  # composite viability score 0-100
    flags: list[str] = field(default_factory=list)

    @property
    def qualifies(self) -> bool:
        return self.score >= 60 and not self.has_earnings_soon

    def __repr__(self):
        flag_str = f" [{', '.join(self.flags)}]" if self.flags else ""
        return (
            f"{self.ticker:<6} score={self.score:.0f}  "
            f"${self.price:.0f}  HV={self.hv30:.0%}  "
            f"pullback={self.pullback_pct:.1%}  "
            f"credit_pot={self.credit_potential:.1%}  "
            f"breach={self.breach_rate:.1%}{flag_str}"
        )


class StrategyScanner:
    """Scan the market for tickers viable for put credit spreads.

    Args:
        universe: List of tickers to scan. Defaults to SCAN_UNIVERSE.
        min_hv: Minimum HV30 (default 0.20).
        max_hv: Maximum HV30 (default 0.55).
        min_price: Minimum stock price (default 30).
        max_price: Maximum stock price (default 600).
        min_adv: Minimum 20-day average daily volume (default 500_000).
        buffer: Buffer for credit potential calculation (default 0.10).
        spread_pct: Spread width for credit potential (default 0.02).
        lookback_years: Years of history for breach rate calc (default 5).
        exclude_sectors: Sectors to exclude (default: Financial, Energy).
        pipe: DataPipeline instance.
    """

    def __init__(
        self,
        universe: list[str] | None = None,
        min_hv: float = 0.20,
        max_hv: float = 0.55,
        min_price: float = 30,
        max_price: float = 600,
        min_adv: float = 500_000,
        buffer: float = 0.10,
        spread_pct: float = 0.02,
        lookback_years: int = 5,
        exclude_sectors: set[str] | None = None,
        pipe: DataPipeline | None = None,
    ):
        self.universe = universe or SCAN_UNIVERSE
        self.min_hv = min_hv
        self.max_hv = max_hv
        self.min_price = min_price
        self.max_price = max_price
        self.min_adv = min_adv
        self.buffer = buffer
        self.spread_pct = spread_pct
        self.lookback_years = lookback_years
        self.exclude_sectors = exclude_sectors if exclude_sectors is not None else HIGH_RISK_SECTORS
        self.pipe = pipe or DataPipeline()

    def scan(self, pullback_only: bool = False) -> list[ScanResult]:
        """Scan all tickers and return scored results.

        Args:
            pullback_only: If True, only return tickers currently in a 3%+ pullback.

        Returns:
            List of ScanResult sorted by score (best first).
        """
        results = []
        for ticker in self.universe:
            try:
                result = self._evaluate_ticker(ticker)
                if result is not None:
                    if pullback_only and result.pullback_pct > -0.03:
                        continue
                    results.append(result)
            except Exception as e:
                logger.debug(f"Skipping {ticker}: {e}")

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _evaluate_ticker(self, ticker: str) -> ScanResult | None:
        """Evaluate a single ticker against all criteria."""
        end = datetime.now().strftime("%Y-%m-%d")
        start_recent = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        start_history = (datetime.now() - timedelta(days=365 * self.lookback_years)).strftime("%Y-%m-%d")

        try:
            df = self.pipe.fetch_stock(ticker, start=start_history, end=end)
        except Exception:
            return None

        if len(df) < 252:  # need at least 1 year
            return None

        close = df["close"]
        price = float(close.iloc[-1])
        sector = SECTOR_MAP.get(ticker, "Unknown")

        # --- Price filter ---
        if price < self.min_price or price > self.max_price:
            return None

        # --- Volume filter ---
        if "volume" in df.columns:
            adv_20 = float(df["volume"].iloc[-20:].mean())
        else:
            adv_20 = 0

        # --- Volatility ---
        vol = historical_volatility(close, window=30)
        hv30 = float(vol.iloc[-1]) if not np.isnan(vol.iloc[-1]) else 0
        if hv30 <= 0:
            return None

        vol_clean = vol.dropna()
        hv30_percentile = float((vol_clean < hv30).mean()) if len(vol_clean) > 30 else 0.5

        # --- Pullback ---
        recent_high = float(close.iloc[-20:].max())
        pullback_pct = (price - recent_high) / recent_high

        # --- Credit potential (using realistic chain strikes) ---
        from tradelab.pricing.strikes import snap_put_credit_spread
        sk, lk = snap_put_credit_spread(
            ticker=ticker,
            underlying_price=price,
            target_buffer=self.buffer,
            target_spread_pct=self.spread_pct,
        )
        credit_potential = 0.0
        if lk > 0 and hv30 > 0:
            try:
                sp = put_credit_spread_price(price, sk, lk, 30 / 365, 0.05, hv30)
                credit_potential = sp["credit_potential"]
            except Exception:
                pass

        # --- Breach rate (how often does price drop 10%+ in 14 trading days?) ---
        closes = close.values
        offset = 10  # ~14 calendar days
        total_windows = max(1, len(closes) - offset)
        breaches = 0
        for i in range(total_windows):
            if closes[i + offset] < closes[i] * 0.90:
                breaches += 1
        breach_rate = breaches / total_windows

        # --- Earnings check ---
        has_earnings = self._check_earnings(ticker)

        # --- Flags ---
        flags = []
        if sector in self.exclude_sectors:
            flags.append(f"HIGH_RISK_SECTOR:{sector}")
        if breach_rate > 0.05:
            flags.append("HIGH_BREACH_RATE")
        if hv30 < self.min_hv:
            flags.append("LOW_VOL")
        if hv30 > self.max_hv:
            flags.append("HIGH_VOL")
        if adv_20 < self.min_adv:
            flags.append("LOW_VOLUME")
        if has_earnings:
            flags.append("EARNINGS_SOON")
        if hv30_percentile > 0.80:
            flags.append("VOL_ELEVATED")
        if pullback_pct < -0.05:
            flags.append("DEEP_PULLBACK")

        # --- Composite score (0-100) ---
        score = self._compute_score(
            hv30, hv30_percentile, price, adv_20, pullback_pct,
            credit_potential, breach_rate, sector, has_earnings,
        )

        return ScanResult(
            ticker=ticker,
            sector=sector,
            price=price,
            hv30=hv30,
            hv30_percentile=hv30_percentile,
            adv_20=adv_20,
            pullback_pct=pullback_pct,
            credit_potential=credit_potential,
            breach_rate=breach_rate,
            has_earnings_soon=has_earnings,
            score=score,
            flags=flags,
        )

    def _compute_score(
        self, hv30, hv30_pct, price, adv, pullback, credit_pot, breach_rate,
        sector, has_earnings,
    ) -> float:
        """Compute a 0-100 viability score."""
        score = 50.0  # baseline

        # Vol in sweet spot (25-45%): +20
        if 0.25 <= hv30 <= 0.45:
            score += 20
        elif 0.20 <= hv30 <= 0.50:
            score += 10
        elif hv30 < 0.15 or hv30 > 0.60:
            score -= 20

        # Credit potential: +15 for > 10%
        if credit_pot > 0.15:
            score += 15
        elif credit_pot > 0.08:
            score += 10
        elif credit_pot > 0.03:
            score += 5
        else:
            score -= 10

        # Low breach rate: +15 for < 2%
        if breach_rate < 0.02:
            score += 15
        elif breach_rate < 0.04:
            score += 8
        elif breach_rate > 0.06:
            score -= 15

        # Currently in pullback: +10
        if pullback < -0.03:
            score += 10
        if pullback < -0.05:
            score += 5  # additional for deep pullback

        # Vol elevated vs history (richer premiums): +5
        if hv30_pct > 0.65:
            score += 5

        # Good price range for small accounts: +5
        if 50 <= price <= 300:
            score += 5

        # Adequate volume: +5
        if adv > 2_000_000:
            score += 5

        # Penalties
        if sector in self.exclude_sectors:
            score -= 15
        if has_earnings:
            score -= 25
        if breach_rate > 0.08:
            score -= 10

        return max(0, min(100, score))

    def _check_earnings(self, ticker: str, hold_days: int = 20) -> bool:
        """Check if earnings are within the hold period. Uses yfinance."""
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            dates = t.earnings_dates
            if dates is None or dates.empty:
                return False
            now = pd.Timestamp.now(tz="America/New_York")
            end = now + pd.Timedelta(days=hold_days)
            upcoming = dates.index[(dates.index >= now) & (dates.index <= end)]
            return len(upcoming) > 0
        except Exception:
            return False  # if we can't check, allow the trade

    def top_candidates(self, n: int = 10, pullback_only: bool = False) -> list[ScanResult]:
        """Return the top N candidates by score."""
        all_results = self.scan(pullback_only=pullback_only)
        qualified = [r for r in all_results if r.qualifies]
        return qualified[:n]

    def backtest_candidate(
        self,
        ticker: str,
        start: str = "2020-01-01",
        end: str | None = None,
    ) -> dict | None:
        """Quick backtest a candidate using the pullback strategy."""
        from tradelab.strategies.pullback_entry import PullbackEntryStrategy

        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")

        try:
            df = self.pipe.fetch_stock(ticker, start=start, end=end)
            if len(df) < 100:
                return None
        except Exception:
            return None

        strat = PullbackEntryStrategy(
            buffer=self.buffer,
            spread_pct=self.spread_pct,
        )
        result = strat.run(df, max_contracts=10)

        return {
            "ticker": ticker,
            "trades": result.total_trades,
            "win_rate": result.win_rate,
            "total_pnl": result.total_pnl,
            "avg_pnl": result.total_pnl / result.total_trades if result.total_trades > 0 else 0,
            "max_dd": result.max_drawdown_pct,
        }

    def scan_and_backtest(
        self,
        n: int = 10,
        start: str = "2020-01-01",
        pullback_only: bool = False,
    ) -> list[dict]:
        """Scan, then backtest top candidates. Returns combined results."""
        candidates = self.top_candidates(n=n * 2, pullback_only=pullback_only)
        results = []
        for c in candidates[:n]:
            bt = self.backtest_candidate(c.ticker, start=start)
            if bt and bt["trades"] > 0:
                results.append({
                    **bt,
                    "score": c.score,
                    "sector": c.sector,
                    "hv30": c.hv30,
                    "pullback_pct": c.pullback_pct,
                    "credit_potential": c.credit_potential,
                    "breach_rate": c.breach_rate,
                    "has_earnings_soon": c.has_earnings_soon,
                })
        results.sort(key=lambda r: r["total_pnl"], reverse=True)
        return results
