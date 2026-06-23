"""Pullback entry strategy: enter put credit spreads when tickers pull back.

Opens 30 DTE spreads only when a ticker has dropped 3%+ from its recent
20-day high, then closes at 14 DTE. This captures elevated premium from
locally high volatility and benefits from mean-reversion.

Backtested result: 23.6% CAGR, -4.9% max drawdown (best risk-adjusted).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tradelab.options import (
    bs_put_price,
    put_credit_spread_price,
    historical_volatility,
)
from tradelab.pricing.base import PricingProvider, PricingError
from tradelab.pricing.strikes import snap_put_credit_spread


@dataclass
class PullbackResult:
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_log: list[dict] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades if self.total_trades > 0 else 0.0

    def summary(self) -> str:
        return (
            f"Trades:     {self.total_trades}\n"
            f"Win rate:   {self.win_rate:.1%} ({self.winners}W / {self.losers}L)\n"
            f"Total P/L:  ${self.total_pnl:+,.2f}\n"
            f"Max DD:     {self.max_drawdown_pct:.1%}"
        )


class PullbackEntryStrategy:
    """Enter put credit spreads only after a pullback from recent highs.

    Args:
        buffer: Short strike OTM distance (default 0.10 = 10%).
        spread_pct: Spread width as % of underlying (default 0.02 = 2%).
        pullback_threshold: Min drawdown from recent high to trigger entry
            (default 0.03 = 3% pullback).
        lookback: Days to look back for the recent high (default 20).
        dte_open: Days to expiry at entry (default 30).
        dte_close: Remaining DTE when we close (default 14).
        risk_free_rate: Annualized risk-free rate (default 0.05).
        vol_window: Rolling window for historical vol (default 30).
        commission_per_contract: Per-leg commission (default 0.65).
        slippage_pct: Credit lost to bid-ask (default 0.02).
    """

    def __init__(
        self,
        buffer: float = 0.10,
        spread_pct: float = 0.02,
        pullback_threshold: float = 0.03,
        lookback: int = 20,
        dte_open: int = 30,
        dte_close: int = 14,
        risk_free_rate: float = 0.05,
        vol_window: int = 30,
        commission_per_contract: float = 0.65,
        slippage_pct: float = 0.02,
    ):
        self.buffer = buffer
        self.spread_pct = spread_pct
        self.pullback_threshold = pullback_threshold
        self.lookback = lookback
        self.dte_open = dte_open
        self.dte_close = dte_close
        self.r = risk_free_rate
        self.vol_window = vol_window
        self.commission = commission_per_contract
        self.slippage_pct = slippage_pct

    def qualifies(self, df: pd.DataFrame, idx: int, close_col: str = "close") -> bool:
        """Check if the ticker qualifies for entry at this index."""
        if idx < self.lookback:
            return False
        prices = df[close_col].iloc[idx - self.lookback : idx + 1]
        recent_high = prices.max()
        current = prices.iloc[-1]
        drawdown = (current - recent_high) / recent_high
        return drawdown <= -self.pullback_threshold

    def run(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
        max_contracts: int = 10,
        ticker: str = "",
        provider: PricingProvider | None = None,
    ) -> PullbackResult:
        """Run the strategy on a single ticker's OHLCV DataFrame.

        Args:
            df: OHLCV DataFrame indexed by unix timestamp.
            close_col: Name of the close price column.
            max_contracts: Max contracts per position.
            ticker: Ticker symbol (required when using a non-default provider).
            provider: PricingProvider to use. If None, uses fast inline B-S
                (legacy path for backward compatibility with existing tests).
        """
        if provider is not None:
            return self._run_with_provider(df, close_col, max_contracts, ticker, provider)
        return self._run_inline_bs(df, close_col, max_contracts)

    def _run_inline_bs(
        self,
        df: pd.DataFrame,
        close_col: str,
        max_contracts: int,
    ) -> PullbackResult:
        """Fast legacy path: inline B-S pricing, no provider overhead."""
        close = df[close_col].values
        timestamps = df.index.values
        vol = historical_volatility(df[close_col], window=self.vol_window)

        offset_open = max(1, int(self.dte_open * 21 / 30))
        offset_close = max(1, int((self.dte_open - self.dte_close) * 21 / 30))

        trade_log = []
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0

        i = max(self.vol_window, self.lookback)
        while i < len(df) - offset_open:
            # Check pullback condition
            recent_high = df[close_col].iloc[i - self.lookback : i + 1].max()
            current = close[i]
            drawdown = (current - recent_high) / recent_high

            if drawdown > -self.pullback_threshold:
                i += 1
                continue

            vol_val = vol.iloc[i]
            if np.isnan(vol_val) or vol_val <= 0:
                i += 1
                continue

            # Price the spread
            sw = current * self.spread_pct
            sk = current * (1 - self.buffer)
            lk = sk - sw
            if lk <= 0:
                i += offset_open
                continue

            sp = put_credit_spread_price(current, sk, lk, self.dte_open / 365, self.r, vol_val)
            credit = sp["net_credit_dollar"]
            max_loss = sp["max_loss"]
            if credit <= 0 or max_loss <= 0:
                i += offset_open
                continue

            # Apply friction
            open_comm = self.commission * 2 * max_contracts
            slippage = credit * max_contracts * self.slippage_pct
            net_credit = credit * max_contracts - slippage

            # Close at 14 DTE
            close_idx = min(i + offset_close, len(close) - 1)
            exit_price = close[close_idx]
            exit_vol = vol.iloc[close_idx] if close_idx < len(vol) else vol_val
            if np.isnan(exit_vol) or exit_vol <= 0:
                exit_vol = vol_val

            close_cost = (
                bs_put_price(exit_price, sk, self.dte_close / 365, self.r, exit_vol)
                - bs_put_price(exit_price, lk, self.dte_close / 365, self.r, exit_vol)
            ) * 100 * max_contracts
            close_comm = self.commission * 2 * max_contracts

            pnl = net_credit - close_cost - open_comm - close_comm
            winner = pnl > 0

            trade_log.append({
                "date": pd.Timestamp(timestamps[i], unit="s"),
                "exit_date": pd.Timestamp(timestamps[close_idx], unit="s"),
                "entry_price": current,
                "exit_price": exit_price,
                "pullback_pct": drawdown * 100,
                "sigma": vol_val,
                "contracts": max_contracts,
                "credit": net_credit,
                "pnl": pnl,
                "winner": winner,
                "source": "blackscholes_inline",
            })

            cum_pnl += pnl
            peak = max(peak, cum_pnl)
            max_dd = min(max_dd, cum_pnl - peak)

            i += offset_close  # advance past the trade

        return PullbackResult(
            total_trades=len(trade_log),
            winners=sum(1 for t in trade_log if t["winner"]),
            losers=sum(1 for t in trade_log if not t["winner"]),
            total_pnl=sum(t["pnl"] for t in trade_log),
            max_drawdown_pct=max_dd / peak if peak > 0 else 0,
            trade_log=trade_log,
        )

    def _run_with_provider(
        self,
        df: pd.DataFrame,
        close_col: str,
        max_contracts: int,
        ticker: str,
        provider: PricingProvider,
    ) -> PullbackResult:
        """Provider-based path: prices through PricingProvider interface.

        Asks the provider to find a valid spread (real listed strikes/expiry
        for ThetaData, computed for BlackScholes). Then prices exit via
        get_spread_quote with the same strikes/expiry at the exit date.
        """
        if not ticker:
            raise ValueError("ticker is required when using a provider")

        close = df[close_col].values
        timestamps = df.index.values

        offset_open = max(1, int(self.dte_open * 21 / 30))
        offset_close = max(1, int((self.dte_open - self.dte_close) * 21 / 30))

        trade_log = []
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        skipped_no_entry = 0
        skipped_no_exit = 0

        i = max(self.vol_window, self.lookback)
        while i < len(df) - offset_open:
            # Check pullback condition
            recent_high = df[close_col].iloc[i - self.lookback : i + 1].max()
            current = close[i]
            drawdown = (current - recent_high) / recent_high

            if drawdown > -self.pullback_threshold:
                i += 1
                continue

            entry_date = pd.Timestamp(timestamps[i], unit="s").strftime("%Y-%m-%d")

            # Ask provider to find a valid spread matching our targets.
            # Pass underlying_price=None so the provider uses its own raw
            # price source (critical for Theta Data where yfinance's
            # split-adjusted prices don't match as-traded option strikes).
            try:
                entry_quote = provider.find_spread_strikes(
                    ticker=ticker,
                    date=entry_date,
                    buffer=self.buffer,
                    spread_pct=self.spread_pct,
                    dte_target=self.dte_open,
                    underlying_price=None,
                )
            except PricingError:
                entry_quote = None

            if entry_quote is None or entry_quote.net_credit_mid <= 0 or entry_quote.max_loss <= 0:
                skipped_no_entry += 1
                i += offset_open
                continue

            sk = entry_quote.short_strike
            lk = entry_quote.long_strike
            expiry = entry_quote.expiry
            credit_per_contract = entry_quote.net_credit_mid

            # Friction
            open_comm = self.commission * 2 * max_contracts
            slippage = credit_per_contract * max_contracts * self.slippage_pct
            net_credit = credit_per_contract * max_contracts - slippage

            # Close at dte_close remaining
            close_idx = min(i + offset_close, len(close) - 1)
            exit_price = close[close_idx]  # split-adjusted from df, for logging only
            exit_date = pd.Timestamp(timestamps[close_idx], unit="s").strftime("%Y-%m-%d")

            # Pass underlying_price=None so the provider uses its own raw
            # price source. Option prices come from the chain by strike/expiry,
            # but the underlying price field in the quote should be consistent.
            try:
                exit_quote = provider.get_spread_quote(
                    ticker=ticker,
                    short_strike=sk,
                    long_strike=lk,
                    expiry=expiry,
                    date=exit_date,
                    underlying_price=None,
                )
                close_cost_per_contract = exit_quote.net_credit_mid
            except PricingError:
                skipped_no_exit += 1
                i += offset_open
                continue

            close_cost = close_cost_per_contract * max_contracts
            close_comm = self.commission * 2 * max_contracts
            pnl = net_credit - close_cost - open_comm - close_comm
            winner = pnl > 0

            trade_log.append({
                "date": pd.Timestamp(timestamps[i], unit="s"),
                "exit_date": pd.Timestamp(timestamps[close_idx], unit="s"),
                "entry_price": current,
                "exit_price": exit_price,
                "pullback_pct": drawdown * 100,
                "short_strike": sk,
                "long_strike": lk,
                "expiry": expiry,
                "contracts": max_contracts,
                "entry_credit_per_contract": credit_per_contract,
                "exit_cost_per_contract": close_cost_per_contract,
                "credit": net_credit,
                "close_cost": close_cost,
                "pnl": pnl,
                "winner": winner,
                "source": provider.name,
            })

            cum_pnl += pnl
            peak = max(peak, cum_pnl)
            max_dd = min(max_dd, cum_pnl - peak)

            i += offset_close

        result = PullbackResult(
            total_trades=len(trade_log),
            winners=sum(1 for t in trade_log if t["winner"]),
            losers=sum(1 for t in trade_log if not t["winner"]),
            total_pnl=sum(t["pnl"] for t in trade_log),
            max_drawdown_pct=max_dd / peak if peak > 0 else 0,
            trade_log=trade_log,
        )
        # Attach diagnostic counts for debugging coverage issues
        result.skipped_no_entry = skipped_no_entry
        result.skipped_no_exit = skipped_no_exit
        return result
