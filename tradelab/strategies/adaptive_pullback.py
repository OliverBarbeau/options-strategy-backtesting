"""Adaptive Pullback Strategy: pullback base + layered safety features.

Extends the research-validated pullback strategy with five opt-in behaviors
designed to address specific empirical weaknesses:

1. **Vol regime pause** (vol_pause_threshold): skip new entries when SPY HV30
   exceeds a threshold. Addresses 2022-style bear markets where losses cluster.

2. **Ticker cooldown after loss** (cooldown_days): don't re-enter a ticker for
   N days after a loss. Addresses the empirical clustering of losses on
   trending-down names (e.g., AAPL taking 3 consecutive losses in Oct 2021).

3. **Adaptive pullback threshold** (adaptive_pullback): scale the required
   pullback by current vs median HV. In quiet markets (HV < median), 3% is
   meaningful. In volatile markets (HV > median), demand 5-6%+ to filter noise.

4. **Early stop-loss on breach** (stop_loss_breach): close the position
   immediately if the stock trades below the short strike for 2+ consecutive
   days. Avoids the last-mile gamma acceleration near expiry.

5. **Aggressive profit take** (fast_profit_target): close at 75% of max credit
   if hit within the first 5 trading days. Redeploys capital on fast winners
   rather than waiting for the 14 DTE checkpoint.

Each feature is independently toggleable so we can backtest the contribution
of each one in isolation. Default values are tuned from the research insights
but can be overridden for sweeps.

Baseline (all features disabled) should behave identically to
PullbackEntryStrategy for validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from tradelab.options import (
    bs_put_price,
    put_credit_spread_price,
    historical_volatility,
)


@dataclass
class AdaptiveResult:
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_log: list[dict] = field(default_factory=list)

    # Diagnostic counts
    skipped_vol_pause: int = 0
    skipped_cooldown: int = 0
    skipped_adaptive_threshold: int = 0
    closed_early_stop_loss: int = 0
    closed_fast_profit: int = 0

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades if self.total_trades > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Trades:     {self.total_trades}",
            f"Win rate:   {self.win_rate:.1%} ({self.winners}W / {self.losers}L)",
            f"Total P/L:  ${self.total_pnl:+,.2f}",
            f"Max DD:     {self.max_drawdown_pct:.1%}",
        ]
        # Only show feature stats if any triggered
        if self.skipped_vol_pause:
            lines.append(f"Skipped (vol pause):       {self.skipped_vol_pause}")
        if self.skipped_cooldown:
            lines.append(f"Skipped (cooldown):        {self.skipped_cooldown}")
        if self.skipped_adaptive_threshold:
            lines.append(f"Skipped (adaptive thresh): {self.skipped_adaptive_threshold}")
        if self.closed_early_stop_loss:
            lines.append(f"Closed (stop loss):        {self.closed_early_stop_loss}")
        if self.closed_fast_profit:
            lines.append(f"Closed (fast profit):      {self.closed_fast_profit}")
        return "\n".join(lines)


class AdaptivePullbackStrategy:
    """Pullback strategy with layered opt-in safety features.

    Base params match the validated pullback strategy. Feature params are
    all off by default, so the baseline behaves identically to the original
    PullbackEntryStrategy for regression testing.

    Args:
        buffer: Short strike OTM distance (default 0.10).
        spread_pct: Spread width as % of underlying (default 0.02).
        pullback_threshold: Minimum drawdown from recent high (default 0.03).
        lookback: Days to look back for recent high (default 20).
        dte_open: Days to expiry at entry (default 30).
        dte_close: Remaining DTE at close checkpoint (default 14).
        risk_free_rate: Annualized risk-free rate.
        vol_window: Window for historical volatility (default 30).
        commission_per_contract: Per-leg commission (default 0.65).
        slippage_pct: Credit lost to bid-ask (default 0.02).

        # --- Feature 1: Vol regime pause ---
        vol_pause_threshold: Market HV30 above which we pause new entries.
            None = disabled. Typical values: 0.25-0.30.

        # --- Feature 2: Ticker cooldown ---
        cooldown_days: Trading days to skip a ticker after a loss.
            0 = disabled. Typical values: 5-15.

        # --- Feature 3: Adaptive pullback threshold ---
        adaptive_pullback: If True, scale pullback_threshold by HV ratio.
            When HV > median, requires deeper pullback; when HV < median,
            standard threshold applies.

        # --- Feature 4: Early stop-loss on breach ---
        stop_loss_breach: If True, close the position when the underlying
            closes below the short strike for 2 consecutive days.

        # --- Feature 5: Aggressive profit take ---
        fast_profit_target: Fraction of max credit to take quick profit
            (0.75 = 75%). None or 0 = disabled.
        fast_profit_window: Trading days during which fast profit applies.
            Default 5.
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
        # Features (all opt-in)
        vol_pause_threshold: float | None = None,
        cooldown_days: int = 0,
        adaptive_pullback: bool = False,
        stop_loss_breach: bool = False,
        fast_profit_target: float | None = None,
        fast_profit_window: int = 5,
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

        self.vol_pause_threshold = vol_pause_threshold
        self.cooldown_days = cooldown_days
        self.adaptive_pullback = adaptive_pullback
        self.stop_loss_breach = stop_loss_breach
        self.fast_profit_target = fast_profit_target
        self.fast_profit_window = fast_profit_window

    def _effective_pullback_threshold(self, current_vol: float, median_vol: float) -> float:
        """Adaptive threshold: deeper pullback required in high-vol regimes."""
        if not self.adaptive_pullback:
            return self.pullback_threshold
        if median_vol <= 0:
            return self.pullback_threshold
        vol_ratio = current_vol / median_vol
        # Scale threshold: 1x at median, 1.5x at 2x median vol, capped at 2x threshold
        scale = min(2.0, max(1.0, 1.0 + (vol_ratio - 1.0) * 0.5))
        return self.pullback_threshold * scale

    def run(
        self,
        df: pd.DataFrame,
        market_vol_series: pd.Series | None = None,
        close_col: str = "close",
        max_contracts: int = 10,
    ) -> AdaptiveResult:
        """Run the adaptive strategy.

        Args:
            df: OHLCV DataFrame indexed by unix timestamp.
            market_vol_series: SPY HV30 series for vol regime pause.
                Required if vol_pause_threshold is set.
            close_col: Close price column.
            max_contracts: Max contracts per trade.
        """
        close = df[close_col].values
        timestamps = df.index.values
        vol = historical_volatility(df[close_col], window=self.vol_window)
        median_vol = float(vol.dropna().median()) if not vol.dropna().empty else 0.25

        offset_open = max(1, int(self.dte_open * 21 / 30))
        offset_close = max(1, int((self.dte_open - self.dte_close) * 21 / 30))

        trade_log = []
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0

        # Feature state
        last_loss_idx = -1  # for cooldown
        skipped_vol_pause = 0
        skipped_cooldown = 0
        skipped_adaptive = 0
        closed_stop_loss = 0
        closed_fast_profit = 0

        def market_vol_at(idx: int) -> float | None:
            """Get SPY HV30 at this index, if provided."""
            if market_vol_series is None:
                return None
            try:
                ts = timestamps[idx]
                pos = market_vol_series.index.get_indexer([ts], method="nearest")[0]
                v = market_vol_series.iloc[pos]
                return float(v) if not np.isnan(v) else None
            except Exception:
                return None

        i = max(self.vol_window, self.lookback)
        while i < len(df) - offset_open:
            # ---- Pullback condition ----
            recent_high = df[close_col].iloc[i - self.lookback : i + 1].max()
            current = close[i]
            drawdown = (current - recent_high) / recent_high

            vol_val = vol.iloc[i]
            if np.isnan(vol_val) or vol_val <= 0:
                i += 1
                continue

            # Feature 3: Adaptive threshold based on vol regime
            effective_threshold = self._effective_pullback_threshold(vol_val, median_vol)
            if drawdown > -effective_threshold:
                if drawdown <= -self.pullback_threshold and self.adaptive_pullback:
                    # Would have entered with base threshold, but adaptive requires deeper
                    skipped_adaptive += 1
                i += 1
                continue

            # Feature 1: Vol regime pause
            if self.vol_pause_threshold is not None:
                mv = market_vol_at(i)
                if mv is not None and mv >= self.vol_pause_threshold:
                    skipped_vol_pause += 1
                    i += 1
                    continue

            # Feature 2: Ticker cooldown after loss
            if self.cooldown_days > 0 and last_loss_idx >= 0:
                days_since_loss = i - last_loss_idx
                if days_since_loss < self.cooldown_days:
                    skipped_cooldown += 1
                    i += 1
                    continue

            # ---- Open position ----
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

            open_comm = self.commission * 2 * max_contracts
            slippage = credit * max_contracts * self.slippage_pct
            net_credit = credit * max_contracts - slippage

            # ---- Determine close: check each day for early exit ----
            close_idx = min(i + offset_close, len(close) - 1)
            exit_reason = "checkpoint"
            days_below_strike = 0

            # Iterate through holding period checking for early exits
            for d in range(1, offset_close + 1):
                check_idx = i + d
                if check_idx >= len(close):
                    close_idx = len(close) - 1
                    break

                check_price = close[check_idx]
                check_vol_val = vol.iloc[check_idx] if check_idx < len(vol) else vol_val
                if np.isnan(check_vol_val) or check_vol_val <= 0:
                    check_vol_val = vol_val

                # Feature 5: Fast profit take
                if self.fast_profit_target and d <= self.fast_profit_window:
                    days_remaining = self.dte_open - int(d * 30 / 21)
                    days_remaining = max(1, days_remaining)
                    current_spread_cost = (
                        bs_put_price(check_price, sk, days_remaining / 365, self.r, check_vol_val)
                        - bs_put_price(check_price, lk, days_remaining / 365, self.r, check_vol_val)
                    ) * 100 * max_contracts

                    unrealized = net_credit - current_spread_cost
                    max_possible = net_credit
                    if max_possible > 0 and unrealized / max_possible >= self.fast_profit_target:
                        close_idx = check_idx
                        exit_reason = "fast_profit"
                        closed_fast_profit += 1
                        break

                # Feature 4: Early stop-loss on strike breach
                if self.stop_loss_breach:
                    if check_price < sk:
                        days_below_strike += 1
                        if days_below_strike >= 2:
                            close_idx = check_idx
                            exit_reason = "stop_loss_breach"
                            closed_stop_loss += 1
                            break
                    else:
                        days_below_strike = 0

            # ---- Compute P/L at close ----
            exit_price = close[close_idx]
            exit_vol = vol.iloc[close_idx] if close_idx < len(vol) else vol_val
            if np.isnan(exit_vol) or exit_vol <= 0:
                exit_vol = vol_val

            days_held = close_idx - i
            days_remaining = max(1, self.dte_open - int(days_held * 30 / 21))

            close_cost = (
                bs_put_price(exit_price, sk, days_remaining / 365, self.r, exit_vol)
                - bs_put_price(exit_price, lk, days_remaining / 365, self.r, exit_vol)
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
                "effective_threshold": effective_threshold,
                "sigma": vol_val,
                "contracts": max_contracts,
                "credit": net_credit,
                "pnl": pnl,
                "winner": winner,
                "exit_reason": exit_reason,
                "days_held": days_held,
            })

            cum_pnl += pnl
            peak = max(peak, cum_pnl)
            max_dd = min(max_dd, cum_pnl - peak)

            if not winner:
                last_loss_idx = close_idx

            # Advance past the trade (to close_idx, not offset_close, since we may have exited early)
            i = max(i + offset_close, close_idx)

        return AdaptiveResult(
            total_trades=len(trade_log),
            winners=sum(1 for t in trade_log if t["winner"]),
            losers=sum(1 for t in trade_log if not t["winner"]),
            total_pnl=sum(t["pnl"] for t in trade_log),
            max_drawdown_pct=max_dd / peak if peak > 0 else 0,
            trade_log=trade_log,
            skipped_vol_pause=skipped_vol_pause,
            skipped_cooldown=skipped_cooldown,
            skipped_adaptive_threshold=skipped_adaptive,
            closed_early_stop_loss=closed_stop_loss,
            closed_fast_profit=closed_fast_profit,
        )
