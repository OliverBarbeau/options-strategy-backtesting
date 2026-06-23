"""Per-ticker hybrid exit strategy.

Empirical finding from Experiment 6 (10-year hybrid sweep): no single exit
policy dominates across tickers. The winning exit rule depends on the
ticker's volatility regime and premium richness. For example:

- High-vol names (NVDA, QQQ, GOOG) benefit from always closing at 14 DTE,
  since holding to expiry exposes you to full tail risk.
- AVGO is the rare case where holding to expiry wins outright — its
  premiums are rich enough that reserving the theta decay is worth it.
- Mid-vol names (MSFT, META, AAPL) benefit from a hybrid exit: a tight
  3-5% pullback checkpoint at 14 DTE, with or without an additional
  emergency stop-loss at 10%.

This strategy takes the per-ticker findings literally: you pass in a
ticker symbol, and it dispatches to the empirically-best exit policy
for that ticker. If the ticker isn't in the lookup table, it falls back
to the project default (always close at 14 DTE).

The entry filter is unchanged from the baseline PullbackEntryStrategy:
3%+ pullback from the 20-day high. Only the exit logic varies by ticker.

Exit modes
----------
- always_close: Open 30 DTE, close unconditionally at `dte_close` (14)
- hold: Open 30 DTE, hold all the way to expiration
- hybrid: Open 30 DTE; at the 14 DTE checkpoint, if the underlying has
  dropped more than `checkpoint_drop_pct` from entry, close early;
  otherwise hold to expiry. Optionally, an `emergency_stop_pct` triggers
  an immediate close on any day the underlying drops more than that
  amount from entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tradelab.options import (
    bs_put_price,
    put_credit_spread_price,
    historical_volatility,
)


# Empirically-best exit config per ticker (from Experiment 6 hybrid sweep).
# Modes:
#   {"mode": "always_close", "dte_close": 14}
#   {"mode": "hold"}
#   {"mode": "hybrid", "checkpoint_drop_pct": X, "emergency_stop_pct": Y|None}
_TICKER_CONFIGS: dict[str, dict[str, Any]] = {
    "QQQ":  {"mode": "always_close", "dte_close": 14},
    "AVGO": {"mode": "hold"},
    "CAT":  {"mode": "hybrid", "checkpoint_drop_pct": 0.05, "emergency_stop_pct": 0.10},
    "MSFT": {"mode": "hybrid", "checkpoint_drop_pct": 0.03, "emergency_stop_pct": 0.10},
    "NVDA": {"mode": "always_close", "dte_close": 14},
    "AAPL": {"mode": "hybrid", "checkpoint_drop_pct": 0.03, "emergency_stop_pct": None},
    "SPY":  {"mode": "hybrid", "checkpoint_drop_pct": 0.05, "emergency_stop_pct": None},
    "META": {"mode": "hybrid", "checkpoint_drop_pct": 0.03, "emergency_stop_pct": None},
    "AMD":  {"mode": "hybrid", "checkpoint_drop_pct": 0.05, "emergency_stop_pct": None},
    "GOOG": {"mode": "always_close", "dte_close": 14},
}

_DEFAULT_CONFIG: dict[str, Any] = {"mode": "always_close", "dte_close": 14}


@dataclass
class PerTickerHybridResult:
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_log: list[dict] = field(default_factory=list)

    # Diagnostics
    exit_mode: str = ""
    exit_reason_counts: dict[str, int] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades if self.total_trades > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Exit mode:  {self.exit_mode}",
            f"Trades:     {self.total_trades}",
            f"Win rate:   {self.win_rate:.1%} ({self.winners}W / {self.losers}L)",
            f"Total P/L:  ${self.total_pnl:+,.2f}",
            f"Max DD:     {self.max_drawdown_pct:.1%}",
        ]
        if self.exit_reason_counts:
            reasons = ", ".join(
                f"{k}={v}" for k, v in sorted(self.exit_reason_counts.items())
            )
            lines.append(f"Exit reasons: {reasons}")
        return "\n".join(lines)


class PerTickerHybridStrategy:
    """Put credit spread with per-ticker exit logic baked in.

    Entry: identical to PullbackEntryStrategy (3%+ pullback from 20-day high).
    Exit: dispatches based on ticker lookup table.

    Args:
        ticker: Required. Determines which exit policy to use.
        buffer: Short strike OTM distance (default 0.10).
        spread_pct: Spread width as % of underlying (default 0.02).
        pullback_threshold: Min drawdown from recent high (default 0.03).
        lookback: Days to look back for recent high (default 20).
        dte_open: DTE at entry (default 30).
        risk_free_rate: Annualized risk-free rate.
        vol_window: HV rolling window.
        commission_per_contract: Per-leg commission.
        slippage_pct: Bid-ask slippage.
    """

    def __init__(
        self,
        ticker: str,
        buffer: float = 0.10,
        spread_pct: float = 0.02,
        pullback_threshold: float = 0.03,
        lookback: int = 20,
        dte_open: int = 30,
        risk_free_rate: float = 0.05,
        vol_window: int = 30,
        commission_per_contract: float = 0.65,
        slippage_pct: float = 0.02,
    ):
        if not ticker:
            raise ValueError("ticker is required for PerTickerHybridStrategy")
        self.ticker = ticker.upper()
        self.buffer = buffer
        self.spread_pct = spread_pct
        self.pullback_threshold = pullback_threshold
        self.lookback = lookback
        self.dte_open = dte_open
        self.r = risk_free_rate
        self.vol_window = vol_window
        self.commission = commission_per_contract
        self.slippage_pct = slippage_pct
        self.config = _TICKER_CONFIGS.get(self.ticker, _DEFAULT_CONFIG)

    # ------------------------------------------------------------------
    # Exit pricing helpers
    # ------------------------------------------------------------------
    def _spread_value(
        self,
        spot: float,
        sk: float,
        lk: float,
        dte_remaining_days: int,
        sigma: float,
    ) -> float:
        """Per-contract dollar cost to close the spread (buy back short,
        sell long). Returns the net debit * 100."""
        if dte_remaining_days <= 0:
            # At expiry: intrinsic only
            short_intrinsic = max(sk - spot, 0.0)
            long_intrinsic = max(lk - spot, 0.0)
            return (short_intrinsic - long_intrinsic) * 100
        t = dte_remaining_days / 365
        short_val = bs_put_price(spot, sk, t, self.r, sigma)
        long_val = bs_put_price(spot, lk, t, self.r, sigma)
        return (short_val - long_val) * 100

    def _simulate_exit(
        self,
        close: np.ndarray,
        vol_series: pd.Series,
        entry_idx: int,
        offset_open: int,
        entry_price: float,
        sk: float,
        lk: float,
        entry_vol: float,
    ) -> tuple[int, float, str]:
        """Walk forward from entry, apply the ticker's exit policy.

        Returns (exit_idx, exit_cost_per_contract_dollars, reason).
        """
        mode = self.config["mode"]
        n = len(close)
        expiry_idx = min(entry_idx + offset_open, n - 1)
        dte_at_entry = self.dte_open

        # Helper: day -> remaining DTE (approx linear mapping matching
        # the offset_open conversion used elsewhere in the codebase).
        def dte_remaining(day_idx: int) -> int:
            days_elapsed_trading = day_idx - entry_idx
            # offset_open = 21 trading days -> 30 calendar days
            calendar_elapsed = int(round(days_elapsed_trading * (dte_at_entry / offset_open)))
            return max(0, dte_at_entry - calendar_elapsed)

        def get_vol(day_idx: int) -> float:
            if day_idx < len(vol_series):
                v = vol_series.iloc[day_idx]
                if not np.isnan(v) and v > 0:
                    return float(v)
            return entry_vol

        # ----- Mode: always_close -----
        if mode == "always_close":
            dte_close = int(self.config.get("dte_close", 14))
            # Use same offset formula as other strategies
            offset_close = max(1, int((dte_at_entry - dte_close) * 21 / 30))
            exit_idx = min(entry_idx + offset_close, n - 1)
            spot = float(close[exit_idx])
            sigma = get_vol(exit_idx)
            cost = self._spread_value(spot, sk, lk, dte_close, sigma)
            return exit_idx, cost, "scheduled_close"

        # ----- Mode: hold to expiry -----
        if mode == "hold":
            exit_idx = expiry_idx
            spot = float(close[exit_idx])
            # At the exact expiry offset, remaining DTE ~ 0
            remaining = dte_remaining(exit_idx)
            sigma = get_vol(exit_idx)
            cost = self._spread_value(spot, sk, lk, remaining, sigma)
            return exit_idx, cost, "expiry"

        # ----- Mode: hybrid -----
        if mode == "hybrid":
            checkpoint_drop = float(self.config["checkpoint_drop_pct"])
            emergency_stop = self.config.get("emergency_stop_pct")
            # Checkpoint at 14 DTE remaining: matches standard dte_open=30 -> dte_close=14 offset
            offset_checkpoint = max(1, int((dte_at_entry - 14) * 21 / 30))
            checkpoint_idx = min(entry_idx + offset_checkpoint, expiry_idx)

            # Scan day-by-day for emergency stop first
            if emergency_stop is not None:
                emergency_threshold = entry_price * (1 - float(emergency_stop))
                for j in range(entry_idx + 1, expiry_idx + 1):
                    if float(close[j]) <= emergency_threshold:
                        spot = float(close[j])
                        remaining = dte_remaining(j)
                        sigma = get_vol(j)
                        cost = self._spread_value(spot, sk, lk, remaining, sigma)
                        return j, cost, "emergency_stop"

            # Checkpoint test
            checkpoint_spot = float(close[checkpoint_idx])
            drawdown_from_entry = (checkpoint_spot - entry_price) / entry_price
            if drawdown_from_entry <= -checkpoint_drop:
                remaining = dte_remaining(checkpoint_idx)
                sigma = get_vol(checkpoint_idx)
                cost = self._spread_value(
                    checkpoint_spot, sk, lk, remaining, sigma
                )
                return checkpoint_idx, cost, "checkpoint_close"

            # Neither triggered: hold to expiry
            exit_idx = expiry_idx
            spot = float(close[exit_idx])
            remaining = dte_remaining(exit_idx)
            sigma = get_vol(exit_idx)
            cost = self._spread_value(spot, sk, lk, remaining, sigma)
            return exit_idx, cost, "expiry"

        raise ValueError(f"Unknown exit mode: {mode}")

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------
    def run(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
        max_contracts: int = 10,
    ) -> PerTickerHybridResult:
        close = df[close_col].values
        timestamps = df.index.values
        vol = historical_volatility(df[close_col], window=self.vol_window)

        offset_open = max(1, int(self.dte_open * 21 / 30))

        trade_log: list[dict] = []
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        exit_reason_counts: dict[str, int] = {}

        i = max(self.vol_window, self.lookback)
        while i < len(df) - offset_open:
            # --- Pullback entry condition (same as baseline) ---
            recent_high = df[close_col].iloc[i - self.lookback : i + 1].max()
            current = float(close[i])
            drawdown = (current - recent_high) / recent_high

            if drawdown > -self.pullback_threshold:
                i += 1
                continue

            vol_val = vol.iloc[i]
            if np.isnan(vol_val) or vol_val <= 0:
                i += 1
                continue
            vol_val = float(vol_val)

            # --- Price the spread ---
            sw = current * self.spread_pct
            sk = current * (1 - self.buffer)
            lk = sk - sw
            if lk <= 0:
                i += offset_open
                continue

            sp = put_credit_spread_price(
                current, sk, lk, self.dte_open / 365, self.r, vol_val
            )
            credit = sp["net_credit_dollar"]
            max_loss = sp["max_loss"]
            if credit <= 0 or max_loss <= 0:
                i += offset_open
                continue

            # Friction (open)
            open_comm = self.commission * 2 * max_contracts
            slippage = credit * max_contracts * self.slippage_pct
            net_credit = credit * max_contracts - slippage

            # --- Dispatch to per-ticker exit logic ---
            exit_idx, close_cost_per_contract, reason = self._simulate_exit(
                close=close,
                vol_series=vol,
                entry_idx=i,
                offset_open=offset_open,
                entry_price=current,
                sk=sk,
                lk=lk,
                entry_vol=vol_val,
            )

            close_cost = close_cost_per_contract * max_contracts
            close_comm = self.commission * 2 * max_contracts
            pnl = net_credit - close_cost - open_comm - close_comm
            winner = pnl > 0

            trade_log.append({
                "date": pd.Timestamp(timestamps[i], unit="s"),
                "exit_date": pd.Timestamp(timestamps[exit_idx], unit="s"),
                "entry_price": current,
                "exit_price": float(close[exit_idx]),
                "pullback_pct": drawdown * 100,
                "sigma": vol_val,
                "contracts": max_contracts,
                "credit": net_credit,
                "close_cost": close_cost,
                "pnl": pnl,
                "winner": winner,
                "exit_reason": reason,
            })

            exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
            cum_pnl += pnl
            peak = max(peak, cum_pnl)
            max_dd = min(max_dd, cum_pnl - peak)

            # Advance past the trade to the exit, at minimum
            i = max(exit_idx, i + 1)

        return PerTickerHybridResult(
            total_trades=len(trade_log),
            winners=sum(1 for t in trade_log if t["winner"]),
            losers=sum(1 for t in trade_log if not t["winner"]),
            total_pnl=sum(t["pnl"] for t in trade_log),
            max_drawdown_pct=max_dd / peak if peak > 0 else 0,
            trade_log=trade_log,
            exit_mode=self.config["mode"],
            exit_reason_counts=exit_reason_counts,
        )
