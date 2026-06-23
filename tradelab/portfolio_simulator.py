"""Portfolio-level backtesting simulator using real historical options data.

Runs a strategy across multiple tickers simultaneously with SHARED capital.
Uses ThetaDataProvider for real option pricing (no B-S approximations).

Design decisions:
- Standalone class (not an extension of Simulator, which is coupled to inline B-S)
- Uses SimulatedAccount as-is for state management
- Pullback depth determines selection priority when multiple tickers qualify
- Risk-capped position sizing: 15% of equity per position, min 1 contract
- Fixed trading-day offset for close (no calendar-day ambiguity)
- Raw/unadjusted stock prices from Theta (matches as-traded option strikes)
- No earnings filter (yfinance is unreliable for historical dates)

Usage::

    from tradelab.account import SimulatedAccount
    from tradelab.pricing.thetadata import ThetaDataProvider
    from tradelab.portfolio_simulator import PortfolioSimulator, PortfolioConfig

    account = SimulatedAccount.load_or_create("accounts/pf_2024.json", 25000)
    provider = ThetaDataProvider()
    config = PortfolioConfig(
        tickers=["NVDA", "AVGO", "MSFT", "GOOG", "CAT", "AAPL"],
        start_date="2024-01-02",
        end_date="2024-12-31",
        starting_capital=25000,
    )
    sim = PortfolioSimulator(account, provider, config)
    sim.run()
    print(sim.report())
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from tradelab.account import SimulatedAccount
from tradelab.pricing.base import PricingError
from tradelab.pricing.thetadata import ThetaDataProvider

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Configuration for a portfolio-level backtest."""
    tickers: list[str]
    start_date: str  # YYYY-MM-DD
    end_date: str
    starting_capital: float = 25000.0

    # Position sizing and capacity
    max_positions: int = 6
    max_pct_per_position: float = 0.15
    max_contracts: int = 10

    # Strategy parameters
    buffer: float = 0.10
    spread_pct: float = 0.02
    pullback_threshold: float = 0.03
    pullback_lookback: int = 20
    dte_open: int = 30
    dte_close: int = 14

    # Friction (applied to P/L in reports; account already applies via its own params)
    commission_per_contract: float = 0.65
    slippage_pct: float = 0.02

    # --- Safety features (all opt-in, off by default for baseline) ---

    # Max portfolio loss: pause all trading when cumulative loss exceeds
    # this fraction of starting capital. None = disabled.
    max_portfolio_loss_pct: float | None = None

    # Portfolio heat limit: max total collateral-at-risk as fraction of equity.
    # When heat >= this, no new positions. None = disabled.
    max_heat: float | None = None

    # Drawdown position scaling: reduce max_positions when equity drops.
    # If enabled, max_positions is reduced to:
    #   normal when DD < 5%, half when DD 5-15%, 1 when DD > 15%
    drawdown_scaling: bool = False

    # Market regime filter: require a reference ticker (SPY) to be above
    # its N-day SMA to enter trades. 0 = disabled.
    trend_sma_days: int = 0

    # Idle capital yield: annual rate earned on un-deployed cash (e.g., T-bills).
    # Applied daily to (balance - locked) on each snapshot day.
    # 0.0 = disabled (cash earns nothing). Typical: 0.04-0.05 for T-bills.
    idle_yield_annual: float = 0.0

    # Risk-free rate for Sharpe/Sortino calculations and T-bill benchmark.
    risk_free_rate: float = 0.045


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Returns (lower, upper) bounds. No scipy dependency.
    """
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class PortfolioReport:
    """Aggregated results of a portfolio simulation."""
    total_return_pct: float
    total_pnl: float
    starting_capital: float
    ending_equity: float
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    max_drawdown_pct: float
    max_drawdown_dollars: float
    per_ticker_pnl: dict[str, float] = field(default_factory=dict)
    per_ticker_trades: dict[str, int] = field(default_factory=dict)
    per_ticker_win_rate: dict[str, float] = field(default_factory=dict)
    trading_days: int = 0
    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    sharpe_ci_lower: float = 0.0
    sharpe_ci_upper: float = 0.0
    risk_free_rate: float = 0.045
    # Win rate confidence interval (Wilson score, 95%)
    win_rate_ci_lower: float = 0.0
    win_rate_ci_upper: float = 0.0
    # Benchmark comparison
    spy_return_pct: float = 0.0
    spy_sharpe: float = 0.0
    spy_max_drawdown_pct: float = 0.0
    tbill_return_pct: float = 0.0
    alpha_vs_spy: float = 0.0

    def summary(self) -> str:
        wr_lo, wr_hi = self.win_rate_ci_lower, self.win_rate_ci_upper
        lines = [
            "=" * 70,
            "PORTFOLIO SIMULATION REPORT",
            "=" * 70,
            f"  Starting capital:  ${self.starting_capital:>12,.2f}",
            f"  Ending equity:     ${self.ending_equity:>12,.2f}",
            f"  Total P/L:         ${self.total_pnl:>+12,.2f}",
            f"  Total return:      {self.total_return_pct:>+12.1f}%",
            f"  Trading days:      {self.trading_days:>12}",
            "",
            f"  Trades:            {self.total_trades:>12}  ({self.winners}W / {self.losers}L)",
            f"  Win rate:          {self.win_rate:>11.1%}  [{wr_lo:.1%}, {wr_hi:.1%}] 95% CI",
            f"  Max drawdown:      {self.max_drawdown_pct:>11.1%}  (${self.max_drawdown_dollars:,.0f})",
            "",
            "  Risk metrics:",
            f"    Sharpe ratio:    {self.sharpe_ratio:>12.2f}  [{self.sharpe_ci_lower:.2f}, {self.sharpe_ci_upper:.2f}] 95% CI",
            f"    Sortino ratio:   {self.sortino_ratio:>12.2f}",
            f"    Risk-free rate:  {self.risk_free_rate:>11.1%}",
            "",
            "  Benchmarks:",
            f"    SPY buy & hold:  {self.spy_return_pct:>+11.1f}%  (Sharpe {self.spy_sharpe:.2f}, DD {self.spy_max_drawdown_pct:.1%})",
            f"    T-bill baseline: {self.tbill_return_pct:>+11.1f}%",
            f"    Alpha vs SPY:    {self.alpha_vs_spy:>+11.1f}%",
            "",
            "  Per-ticker contribution:",
        ]
        for ticker in sorted(self.per_ticker_pnl.keys(), key=lambda t: self.per_ticker_pnl[t], reverse=True):
            pnl = self.per_ticker_pnl[ticker]
            trades = self.per_ticker_trades.get(ticker, 0)
            wr = self.per_ticker_win_rate.get(ticker, 0)
            lines.append(
                f"    {ticker:<6} ${pnl:>+10,.2f}  ({trades} trades, {wr:.0%} WR)"
            )
        lines.append("=" * 70)
        return "\n".join(lines)


class PortfolioSimulator:
    """Portfolio-level backtester using real historical options data.

    Runs the pullback entry strategy across multiple tickers with shared
    capital, using ThetaDataProvider for all option pricing.
    """

    def __init__(
        self,
        account: SimulatedAccount,
        provider: ThetaDataProvider,
        config: PortfolioConfig,
        verbose: bool = False,
    ):
        self.account = account
        self.provider = provider
        self.config = config
        self.verbose = verbose

        # Per-ticker raw price history (from Theta, not yfinance)
        self._prices: dict[str, pd.DataFrame] = {}
        # Master trading-day calendar (sorted unique dates across all tickers)
        self._calendar: list[str] = []

        # Track retry attempts per stuck position (position_id -> attempts)
        self._retry_counts: dict[str, int] = {}
        # After this many failed retries, force-close at intrinsic value
        self._max_retries = 5

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_price_data(self):
        """Fetch stock price history for all tickers.

        Uses yfinance (via DataPipeline) for price history — this is
        split-adjusted, which is fine for pullback signals since those are
        relative measurements. Theta Data option pricing uses its own raw
        underlying price via get_stock_eod() independently.

        Why not Theta for stock prices? The Standard Options plan includes
        FREE stock access which only covers ~1 year of trailing data. For
        multi-year backtests (2018-2025), yfinance is the reliable source.
        """
        from tradelab.pipeline import DataPipeline

        start_dt = datetime.fromisoformat(self.config.start_date)
        lookback_start = (start_dt - timedelta(days=self.config.pullback_lookback + 10)).strftime("%Y-%m-%d")

        pipe = DataPipeline()

        if self.verbose:
            print(f"Loading price history for {len(self.config.tickers)} tickers (yfinance)...")

        # Always include SPY for benchmark comparison
        tickers_to_load = list(self.config.tickers)
        if "SPY" not in tickers_to_load:
            tickers_to_load.append("SPY")

        for ticker in tickers_to_load:
            try:
                df = pipe.fetch_stock(ticker, start=lookback_start, end=self.config.end_date)
                # Convert from unix-timestamp index to YYYY-MM-DD string index
                df.index = pd.Index(
                    [pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d") for ts in df.index],
                    name="date",
                )
                self._prices[ticker] = df
                if self.verbose:
                    print(f"  {ticker}: {len(df)} days")
            except Exception as e:
                logger.warning(f"Failed to load {ticker}: {e}")

    def _build_calendar(self):
        """Build the master trading-day calendar from loaded price data.

        Returns the intersection of all ticker histories, sorted, within
        the configured date range.
        """
        if not self._prices:
            return

        # Use the first ticker's calendar as the reference
        # (All major US tickers have the same trading days except for halts)
        reference = next(iter(self._prices.values()))
        all_dates = reference.index.tolist()

        # Filter to configured range
        self._calendar = sorted([
            d for d in all_dates
            if self.config.start_date <= d <= self.config.end_date
        ])

    # ------------------------------------------------------------------
    # Signal and selection
    # ------------------------------------------------------------------

    def _pullback_depth(self, ticker: str, date: str) -> float | None:
        """Compute drawdown from recent high. Negative = in pullback.

        Returns None if insufficient history or no price on that date.
        """
        df = self._prices.get(ticker)
        if df is None or date not in df.index:
            return None

        idx = df.index.get_loc(date)
        if idx < self.config.pullback_lookback:
            return None

        window = df["close"].iloc[idx - self.config.pullback_lookback : idx + 1]
        recent_high = window.max()
        current = window.iloc[-1]
        return (current - recent_high) / recent_high

    def _should_trade(self, date: str) -> bool:
        """Portfolio-level gate: should we enter any new trades today?"""
        c = self.config

        # Max portfolio loss: stop trading if cumulative loss exceeds threshold
        if c.max_portfolio_loss_pct is not None:
            loss_pct = (self.account.equity - c.starting_capital) / c.starting_capital
            if loss_pct <= -c.max_portfolio_loss_pct:
                return False

        # Portfolio heat: total locked collateral vs equity
        if c.max_heat is not None and self.account.equity > 0:
            heat = self.account.locked / self.account.equity
            if heat >= c.max_heat:
                return False

        # Market trend filter: require SPY above its N-day SMA
        if c.trend_sma_days > 0:
            spy_df = self._prices.get("SPY")
            if spy_df is None:
                # Load SPY if not in our ticker list
                try:
                    from tradelab.pipeline import DataPipeline
                    pipe = DataPipeline()
                    start_dt = datetime.fromisoformat(c.start_date)
                    lookback = (start_dt - timedelta(days=c.trend_sma_days + 30)).strftime("%Y-%m-%d")
                    df = pipe.fetch_stock("SPY", start=lookback, end=c.end_date)
                    df.index = pd.Index(
                        [pd.Timestamp(ts, unit="s").strftime("%Y-%m-%d") for ts in df.index],
                        name="date",
                    )
                    self._prices["SPY"] = df
                    spy_df = df
                except Exception:
                    pass

            if spy_df is not None and date in spy_df.index:
                idx = spy_df.index.get_loc(date)
                if idx >= c.trend_sma_days:
                    sma = spy_df["close"].iloc[idx - c.trend_sma_days : idx].mean()
                    current = spy_df["close"].iloc[idx]
                    if current < sma:
                        return False

        return True

    def _effective_max_positions(self) -> int:
        """Drawdown-adjusted max positions."""
        c = self.config
        if not c.drawdown_scaling:
            return c.max_positions

        if self.account.equity_curve:
            peak = max(s.equity for s in self.account.equity_curve)
            dd = (self.account.equity - peak) / peak if peak > 0 else 0
        else:
            dd = 0

        if dd < -0.15:
            return 1
        elif dd < -0.05:
            return max(1, c.max_positions // 2)
        return c.max_positions

    def _qualifies(self, ticker: str, date: str) -> tuple[bool, float]:
        """Check if a ticker qualifies for entry today. Returns (ok, depth)."""
        # Skip if we already have a position on this ticker
        if ticker in self.account.open_tickers:
            return False, 0.0

        depth = self._pullback_depth(ticker, date)
        if depth is None:
            return False, 0.0

        return depth <= -self.config.pullback_threshold, depth

    def _select_entries(self, date: str) -> list[tuple[str, float]]:
        """Rank qualifying tickers by pullback depth (deepest first)."""
        qualified = []
        for ticker in self.config.tickers:
            ok, depth = self._qualifies(ticker, date)
            if ok:
                qualified.append((ticker, depth))

        qualified.sort(key=lambda x: x[1])  # most negative first
        return qualified

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def _trading_day_offset(self, date: str, offset: int) -> str | None:
        """Find the trading day N days forward from `date` in the calendar."""
        try:
            idx = self._calendar.index(date)
        except ValueError:
            return None
        target_idx = idx + offset
        if target_idx >= len(self._calendar):
            return None
        return self._calendar[target_idx]

    def _intrinsic_value_close(self, pos, date: str) -> float:
        """Compute intrinsic value of a put credit spread from current stock price.

        When real option data is unavailable (stale strikes after split, etc.),
        we estimate the spread's value from the underlying price.

        Uses Theta's raw stock EOD (unadjusted) so the price matches the
        as-traded option strikes. Falls back to max-loss if no price available.
        """
        # Try Theta raw price first (matches option strike levels)
        stock_px = self.provider.get_stock_eod(pos.ticker, date)

        if stock_px is None:
            # Theta stock EOD may not be available (FREE stock subscription).
            # Fall back to max loss to be safe.
            return (pos.short_strike - pos.long_strike) * 100 * pos.contracts

        spread_width = pos.short_strike - pos.long_strike

        if stock_px >= pos.short_strike:
            return 0.0  # both puts OTM, worth nothing
        elif stock_px <= pos.long_strike:
            return spread_width * 100 * pos.contracts  # max loss
        else:
            # Partial: short put ITM, long put OTM
            return (pos.short_strike - stock_px) * 100 * pos.contracts

    def _close_due_positions(self, date: str):
        """Close any positions whose close_target_date has arrived or passed.

        If the real option chain lookup fails (strike delisted, split effects,
        etc.), retry next day. After self._max_retries failed attempts,
        force-close at intrinsic value computed from the underlying stock price.
        """
        for pos in list(self.account.positions):
            if pos.close_target_date > date:
                continue

            close_cost = None
            exit_reason = "checkpoint"

            try:
                quote = self.provider.get_spread_quote(
                    ticker=pos.ticker,
                    short_strike=pos.short_strike,
                    long_strike=pos.long_strike,
                    expiry=pos.notes if pos.notes else "",
                    date=date,
                    underlying_price=None,
                )
                close_cost = quote.net_credit_mid * pos.contracts
                self._retry_counts.pop(pos.id, None)
            except (PricingError, Exception) as e:
                retries = self._retry_counts.get(pos.id, 0) + 1
                self._retry_counts[pos.id] = retries

                if retries < self._max_retries:
                    if self.verbose and retries == 1:
                        print(f"  [{date}] Retry close {pos.ticker} {pos.short_strike}/{pos.long_strike}: {e}")
                    continue

                # Exhausted retries -- force close at intrinsic value
                close_cost = self._intrinsic_value_close(pos, date)
                exit_reason = "forced_intrinsic"
                if self.verbose:
                    print(f"  [{date}] FORCE CLOSE {pos.ticker} {pos.short_strike}/{pos.long_strike} "
                          f"at intrinsic ${close_cost:.2f} after {retries} retries")
                self._retry_counts.pop(pos.id, None)

            exit_price = self._prices.get(pos.ticker)
            exit_px = 0.0
            if exit_price is not None and date in exit_price.index:
                exit_px = float(exit_price.loc[date, "close"])

            self.account.close_position(
                pos_id=pos.id,
                date=date,
                exit_price=exit_px,
                close_cost=close_cost,
                exit_reason=exit_reason,
            )
            if self.verbose and exit_reason == "checkpoint":
                print(f"  [{date}] Close {pos.ticker} {pos.short_strike}/{pos.long_strike}  cost=${close_cost:.2f}")

    def _try_open(self, ticker: str, date: str) -> bool:
        """Attempt to open a position on ticker at date. Returns True on success."""
        # Trading-day offset close target (~11 trading days = 16 calendar days)
        offset_close = max(1, int((self.config.dte_open - self.config.dte_close) * 21 / 30))
        close_target_date = self._trading_day_offset(date, offset_close)
        if close_target_date is None:
            return False  # not enough calendar left

        # Ask provider to find a valid spread
        try:
            quote = self.provider.find_spread_strikes(
                ticker=ticker,
                date=date,
                buffer=self.config.buffer,
                spread_pct=self.config.spread_pct,
                dte_target=self.config.dte_open,
                underlying_price=None,  # provider fetches raw price
            )
        except PricingError:
            return False

        if quote is None or quote.net_credit_mid <= 0 or quote.max_loss <= 0:
            return False

        collateral_per_contract = quote.max_loss + quote.net_credit_mid  # = spread_width
        credit_per_contract = quote.net_credit_mid

        # Risk-capped sizing: cap at max_pct_per_position of equity
        equity = self.account.equity
        max_allocation = equity * self.config.max_pct_per_position
        max_by_cap = max(1, int(max_allocation / collateral_per_contract))
        contracts = min(self.config.max_contracts, max_by_cap)

        # Ensure we can afford at least 1 contract
        if collateral_per_contract > self.account.balance:
            return False

        # Store expiry in notes field so we can close the right contract later
        pos = self.account.open_position(
            ticker=ticker,
            date=date,
            entry_price=quote.underlying_price,
            short_strike=quote.short_strike,
            long_strike=quote.long_strike,
            contracts=contracts,
            credit_per_contract=credit_per_contract,
            collateral_per_contract=collateral_per_contract,
            close_target_date=close_target_date,
            buffer=self.config.buffer,
            entry_vol=0.0,
            entry_regime="pullback_theta",
            notes=quote.expiry,  # expiry stored here for close lookup
        )

        if pos is None:
            return False

        if self.verbose:
            print(
                f"  [{date}] Open {ticker} {quote.short_strike}/{quote.long_strike} "
                f"exp {quote.expiry} x{contracts}  credit=${credit_per_contract:.2f}  "
                f"col=${collateral_per_contract:.2f}"
            )
        return True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Run the portfolio simulation day by day."""
        self._load_price_data()
        self._build_calendar()

        if not self._calendar:
            print("No trading days in configured range.")
            return

        if self.verbose:
            print(f"\nRunning simulation: {len(self._calendar)} trading days")
            print(f"  {self._calendar[0]} to {self._calendar[-1]}")
            print()

        for day_idx, date in enumerate(self._calendar):
            # 1. Close due positions (checkpoint reached)
            self._close_due_positions(date)

            # 2. Check portfolio-level safety features before opening
            effective_max = self._effective_max_positions()
            should_trade = self._should_trade(date)

            # 3. Select and open new positions (if allowed)
            if should_trade and len(self.account.positions) < effective_max:
                candidates = self._select_entries(date)
                for ticker, depth in candidates:
                    if len(self.account.positions) >= effective_max:
                        break
                    self._try_open(ticker, date)

            # 4. Credit idle capital yield (T-bills on un-deployed cash)
            if self.config.idle_yield_annual > 0:
                idle_cash = max(0, self.account.balance)
                daily_rate = self.config.idle_yield_annual / 252
                interest = idle_cash * daily_rate
                self.account.balance += interest

            # 5. Snapshot equity curve
            self.account.snapshot(date)

            # Progress indicator
            if self.verbose and (day_idx + 1) % 20 == 0:
                print(
                    f"  [{date}] {day_idx + 1}/{len(self._calendar)}  "
                    f"equity=${self.account.equity:,.0f}  "
                    f"open={len(self.account.positions)}  "
                    f"trades={self.account.total_trades_count}"
                )

        # Force-close any remaining positions at the final date
        final_date = self._calendar[-1]
        for pos in list(self.account.positions):
            close_cost = None
            try:
                quote = self.provider.get_spread_quote(
                    ticker=pos.ticker,
                    short_strike=pos.short_strike,
                    long_strike=pos.long_strike,
                    expiry=pos.notes,
                    date=final_date,
                    underlying_price=None,
                )
                close_cost = quote.net_credit_mid * pos.contracts
            except Exception:
                close_cost = self._intrinsic_value_close(pos, final_date)

            self.account.close_position(
                pos_id=pos.id,
                date=final_date,
                exit_price=0.0,
                close_cost=close_cost,
                exit_reason="forced_final",
            )
            if self.verbose:
                print(f"  [{final_date}] FORCE CLOSE (end) {pos.ticker} {pos.short_strike}/{pos.long_strike}  cost=${close_cost:.2f}")

        if self.verbose:
            print(f"\nSimulation complete. Final equity: ${self.account.equity:,.2f}")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> PortfolioReport:
        """Compute aggregate metrics and return a PortfolioReport."""
        starting = self.config.starting_capital
        ending = self.account.equity
        pnl = ending - starting
        total_return = pnl / starting * 100 if starting > 0 else 0

        # Per-ticker contribution
        per_ticker_pnl: dict[str, float] = {}
        per_ticker_trades: dict[str, int] = {}
        per_ticker_winners: dict[str, int] = {}
        for trade in self.account.trades:
            per_ticker_pnl[trade.ticker] = per_ticker_pnl.get(trade.ticker, 0) + trade.pnl
            per_ticker_trades[trade.ticker] = per_ticker_trades.get(trade.ticker, 0) + 1
            if trade.winner:
                per_ticker_winners[trade.ticker] = per_ticker_winners.get(trade.ticker, 0) + 1

        per_ticker_wr = {
            t: per_ticker_winners.get(t, 0) / n
            for t, n in per_ticker_trades.items()
            if n > 0
        }

        # Max drawdown from equity curve
        equity_values = [s.equity for s in self.account.equity_curve]
        max_dd_dollars = 0.0
        max_dd_pct = 0.0
        if equity_values:
            peak = equity_values[0]
            for val in equity_values:
                peak = max(peak, val)
                dd = val - peak
                if dd < max_dd_dollars:
                    max_dd_dollars = dd
                    max_dd_pct = dd / peak if peak > 0 else 0

        # --- Risk metrics ---
        rf = self.config.risk_free_rate if hasattr(self.config, "risk_free_rate") else 0.045
        rf_daily = rf / 252

        sharpe = 0.0
        sortino = 0.0
        sharpe_ci_lo = 0.0
        sharpe_ci_hi = 0.0
        daily_returns = np.array([])

        if len(equity_values) > 2:
            daily_returns = np.diff(equity_values) / np.array(equity_values[:-1])
            excess = daily_returns - rf_daily
            if daily_returns.std() > 0:
                sharpe = (excess.mean() / daily_returns.std()) * np.sqrt(252)
            # Sortino: downside deviation only
            downside = np.minimum(excess, 0)
            downside_std = np.sqrt(np.mean(downside ** 2))
            if downside_std > 0:
                sortino = (excess.mean() / downside_std) * np.sqrt(252)
            # Sharpe confidence interval: SE = sqrt((1 + S^2/2) / (n-1))
            n_obs = len(daily_returns)
            if n_obs > 1:
                se = math.sqrt((1 + sharpe ** 2 / 2) / (n_obs - 1))
                sharpe_ci_lo = sharpe - 1.96 * se
                sharpe_ci_hi = sharpe + 1.96 * se

        total_trades = len(self.account.trades)
        winners = sum(1 for t in self.account.trades if t.winner)
        losers = total_trades - winners
        win_rate = winners / total_trades if total_trades else 0
        wr_ci_lo, wr_ci_hi = _wilson_ci(winners, total_trades)

        # --- Benchmark: SPY buy-and-hold ---
        spy_return = 0.0
        spy_sharpe = 0.0
        spy_max_dd = 0.0
        spy_df = self._prices.get("SPY")
        if spy_df is not None and len(self._calendar) >= 2:
            cal_dates = [d for d in self._calendar if d in spy_df.index]
            if len(cal_dates) >= 2:
                spy_start = float(spy_df.loc[cal_dates[0], "close"])
                spy_end = float(spy_df.loc[cal_dates[-1], "close"])
                spy_return = (spy_end / spy_start - 1) * 100
                # SPY Sharpe
                spy_closes = [float(spy_df.loc[d, "close"]) for d in cal_dates]
                spy_daily = np.diff(spy_closes) / np.array(spy_closes[:-1])
                spy_excess = spy_daily - rf_daily
                if spy_daily.std() > 0:
                    spy_sharpe = (spy_excess.mean() / spy_daily.std()) * np.sqrt(252)
                # SPY max drawdown
                spy_peak = spy_closes[0]
                for v in spy_closes:
                    spy_peak = max(spy_peak, v)
                    dd = (v - spy_peak) / spy_peak if spy_peak > 0 else 0
                    if dd < spy_max_dd:
                        spy_max_dd = dd

        # T-bill baseline
        years = len(self._calendar) / 252 if self._calendar else 0
        tbill_return = ((1 + rf) ** years - 1) * 100 if years > 0 else 0

        # Alpha = strategy CAGR - SPY CAGR
        strat_cagr = ((ending / starting) ** (1 / years) - 1) * 100 if years > 0 and starting > 0 else 0
        spy_cagr = ((1 + spy_return / 100) ** (1 / years) - 1) * 100 if years > 0 and spy_return != 0 else 0
        alpha = strat_cagr - spy_cagr

        return PortfolioReport(
            total_return_pct=total_return,
            total_pnl=pnl,
            starting_capital=starting,
            ending_equity=ending,
            total_trades=total_trades,
            winners=winners,
            losers=losers,
            win_rate=win_rate,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_dollars=max_dd_dollars,
            per_ticker_pnl=per_ticker_pnl,
            per_ticker_trades=per_ticker_trades,
            per_ticker_win_rate=per_ticker_wr,
            trading_days=len(self._calendar),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            sharpe_ci_lower=sharpe_ci_lo,
            sharpe_ci_upper=sharpe_ci_hi,
            risk_free_rate=rf,
            win_rate_ci_lower=wr_ci_lo,
            win_rate_ci_upper=wr_ci_hi,
            spy_return_pct=spy_return,
            spy_sharpe=spy_sharpe,
            spy_max_drawdown_pct=spy_max_dd,
            tbill_return_pct=tbill_return,
            alpha_vs_spy=alpha,
        )
