"""Composite pullback: best-of-breed combination from empirical sweep.

Empirical findings from the strategy shootout (2014-2024, B-S pricing):

1. **RSI < 25 is the strongest single filter** (+$16.75/trade vs baseline
   -$3.66/trade), but generates only ~2 trades/year — too few to be
   practical as a standalone strategy.

2. **RSI + SMA combo is too restrictive** — RSI < 35 stocks are usually
   near/below their 50 SMA, so the dual filter produces near-zero trades.

3. **Gap recovery underperforms** rolling pullback — the 20-day pullback
   signal subsumes most gap-downs and also catches gradual declines.

4. **Vol-scaling hurts** in B-S backtests because lower sizing during high
   vol means less premium capture, and high-vol trades are often the most
   profitable (the "volatility premium" effect).

This composite strategy addresses the trade-off between selectivity and
sample size by using a **tiered entry system**:

- **Tier 1 (best)**: RSI < 25 AND pullback > 3%. Maximum conviction.
  Full position size.
- **Tier 2 (good)**: RSI < 35 AND pullback > 4%. Strong signal, requires
  slightly deeper dip to compensate for weaker RSI confirmation.
  80% position size.
- **Tier 3 (standard)**: Pullback > 5% (no RSI requirement). Deep dips
  without RSI confirmation are still valuable — the depth itself signals
  mean-reversion potential. 60% position size.

This ensures we capture the high-quality RSI signals at full size while
still generating enough trades to be practical.

Additional safety: vol regime pause at HV30 > 40% (empirically, the
2022 bear market had HV30 > 30% for months — we pause even later at 40%
to preserve the high-vol premium capture).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import pandas as pd

from tradelab.options import (
    bs_put_price,
    put_credit_spread_price,
    historical_volatility,
)
from tradelab.strategies.rsi_pullback import compute_rsi


@dataclass
class CompositeResult:
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    trade_log: list[dict] = field(default_factory=list)

    # Per-tier diagnostics
    tier1_trades: int = 0
    tier2_trades: int = 0
    tier3_trades: int = 0
    skipped_vol_pause: int = 0

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades if self.total_trades > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Trades:     {self.total_trades}",
            f"Win rate:   {self.win_rate:.1%} ({self.winners}W / {self.losers}L)",
            f"Total P/L:  ${self.total_pnl:+,.2f}",
            f"Max DD:     {self.max_drawdown_pct:.1%}",
            f"Tier 1 (RSI<25 + 3%pb):  {self.tier1_trades}",
            f"Tier 2 (RSI<35 + 4%pb):  {self.tier2_trades}",
            f"Tier 3 (5%+ pullback):   {self.tier3_trades}",
        ]
        if self.skipped_vol_pause:
            lines.append(f"Skipped (vol pause):     {self.skipped_vol_pause}")
        return "\n".join(lines)


class CompositePullbackStrategy:
    """Tiered entry with RSI-scaled position sizing.

    Args:
        buffer: Short strike OTM distance (default 0.10).
        spread_pct: Spread width as % of underlying (default 0.02).
        dte_open: DTE at entry (default 30).
        dte_close: Remaining DTE at close (default 14).
        max_contracts: Full-size position (Tier 1 gets this, others scale down).
        tier1_rsi: RSI threshold for Tier 1 (default 25).
        tier1_pullback: Pullback threshold for Tier 1 (default 0.03).
        tier2_rsi: RSI threshold for Tier 2 (default 35).
        tier2_pullback: Pullback threshold for Tier 2 (default 0.04).
        tier3_pullback: Pullback threshold for Tier 3, no RSI (default 0.05).
        tier2_scale: Position scale for Tier 2 (default 0.8).
        tier3_scale: Position scale for Tier 3 (default 0.6).
        vol_pause: Pause entries when ticker HV > this (default 0.40).
        lookback: Days to look back for recent high (default 20).
        rsi_period: RSI period (default 14).
        risk_free_rate: Annualized risk-free rate.
        vol_window: HV rolling window.
        commission_per_contract: Per-leg commission.
        slippage_pct: Bid-ask slippage.
    """

    def __init__(
        self,
        buffer: float = 0.10,
        spread_pct: float = 0.02,
        dte_open: int = 30,
        dte_close: int = 14,
        max_contracts: int = 10,
        tier1_rsi: float = 25.0,
        tier1_pullback: float = 0.03,
        tier2_rsi: float = 35.0,
        tier2_pullback: float = 0.04,
        tier3_pullback: float = 0.05,
        tier2_scale: float = 0.80,
        tier3_scale: float = 0.60,
        vol_pause: float = 0.40,
        lookback: int = 20,
        rsi_period: int = 14,
        risk_free_rate: float = 0.05,
        vol_window: int = 30,
        commission_per_contract: float = 0.65,
        slippage_pct: float = 0.02,
    ):
        self.buffer = buffer
        self.spread_pct = spread_pct
        self.dte_open = dte_open
        self.dte_close = dte_close
        self.max_contracts = max_contracts
        self.tier1_rsi = tier1_rsi
        self.tier1_pullback = tier1_pullback
        self.tier2_rsi = tier2_rsi
        self.tier2_pullback = tier2_pullback
        self.tier3_pullback = tier3_pullback
        self.tier2_scale = tier2_scale
        self.tier3_scale = tier3_scale
        self.vol_pause = vol_pause
        self.lookback = lookback
        self.rsi_period = rsi_period
        self.r = risk_free_rate
        self.vol_window = vol_window
        self.commission = commission_per_contract
        self.slippage_pct = slippage_pct

    def _classify_entry(self, pullback_pct: float, rsi_val: float) -> tuple[str, float] | None:
        """Classify entry tier and return (tier_name, scale_factor) or None."""
        pb = abs(pullback_pct)  # pullback as positive number

        # Tier 1: RSI deeply oversold + any qualifying pullback
        if rsi_val <= self.tier1_rsi and pb >= self.tier1_pullback:
            return ("tier1", 1.0)

        # Tier 2: RSI moderately oversold + deeper pullback
        if rsi_val <= self.tier2_rsi and pb >= self.tier2_pullback:
            return ("tier2", self.tier2_scale)

        # Tier 3: Very deep pullback, no RSI required
        if pb >= self.tier3_pullback:
            return ("tier3", self.tier3_scale)

        return None

    def run(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
        max_contracts: int | None = None,
    ) -> CompositeResult:
        if max_contracts is None:
            max_contracts = self.max_contracts

        close = df[close_col].values
        timestamps = df.index.values
        vol = historical_volatility(df[close_col], window=self.vol_window)
        rsi = compute_rsi(df[close_col], period=self.rsi_period)

        offset_open = max(1, int(self.dte_open * 21 / 30))
        offset_close = max(1, int((self.dte_open - self.dte_close) * 21 / 30))

        trade_log = []
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        tier_counts = {"tier1": 0, "tier2": 0, "tier3": 0}
        skipped_vol_pause = 0

        start = max(self.vol_window, self.lookback, self.rsi_period + 1)
        i = start
        while i < len(df) - offset_open:
            current = close[i]

            # --- Vol pause ---
            vol_val = vol.iloc[i]
            if np.isnan(vol_val) or vol_val <= 0:
                i += 1
                continue
            if vol_val > self.vol_pause:
                skipped_vol_pause += 1
                i += 1
                continue

            # --- Pullback magnitude ---
            recent_high = df[close_col].iloc[i - self.lookback : i + 1].max()
            drawdown = (current - recent_high) / recent_high  # negative number

            # --- RSI ---
            rsi_val = rsi.iloc[i]
            if np.isnan(rsi_val):
                i += 1
                continue

            # --- Classify entry tier ---
            classification = self._classify_entry(drawdown, rsi_val)
            if classification is None:
                i += 1
                continue

            tier_name, scale = classification

            # --- Position size ---
            contracts = max(1, math.floor(max_contracts * scale))

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

            # Friction
            open_comm = self.commission * 2 * contracts
            slippage = credit * contracts * self.slippage_pct
            net_credit = credit * contracts - slippage

            # Close at checkpoint
            close_idx = min(i + offset_close, len(close) - 1)
            exit_price = close[close_idx]
            exit_vol = vol.iloc[close_idx] if close_idx < len(vol) else vol_val
            if np.isnan(exit_vol) or exit_vol <= 0:
                exit_vol = vol_val

            close_cost = (
                bs_put_price(exit_price, sk, self.dte_close / 365, self.r, exit_vol)
                - bs_put_price(exit_price, lk, self.dte_close / 365, self.r, exit_vol)
            ) * 100 * contracts
            close_comm = self.commission * 2 * contracts

            pnl = net_credit - close_cost - open_comm - close_comm
            winner = pnl > 0

            trade_log.append({
                "date": pd.Timestamp(timestamps[i], unit="s"),
                "exit_date": pd.Timestamp(timestamps[close_idx], unit="s"),
                "entry_price": current,
                "exit_price": exit_price,
                "pullback_pct": drawdown * 100,
                "rsi": rsi_val,
                "tier": tier_name,
                "contracts": contracts,
                "sigma": vol_val,
                "credit": net_credit,
                "pnl": pnl,
                "winner": winner,
            })

            tier_counts[tier_name] += 1
            cum_pnl += pnl
            peak = max(peak, cum_pnl)
            max_dd = min(max_dd, cum_pnl - peak)

            i += offset_close

        return CompositeResult(
            total_trades=len(trade_log),
            winners=sum(1 for t in trade_log if t["winner"]),
            losers=sum(1 for t in trade_log if not t["winner"]),
            total_pnl=sum(t["pnl"] for t in trade_log),
            max_drawdown_pct=max_dd / peak if peak > 0 else 0,
            trade_log=trade_log,
            tier1_trades=tier_counts["tier1"],
            tier2_trades=tier_counts["tier2"],
            tier3_trades=tier_counts["tier3"],
            skipped_vol_pause=skipped_vol_pause,
        )
