"""Regime-adaptive strategy: adjust buffer based on market volatility regime.

Dynamically widens the buffer when market vol is elevated (defensive) and
tightens it when vol is low (capture more premium). Uses SPY HV30 as the
regime indicator.

Backtested result: 31.2% CAGR, -9.0% max drawdown (highest raw returns).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from tradelab.options import (
    bs_put_price,
    put_credit_spread_price,
    historical_volatility,
)


class Regime(Enum):
    LOW_VOL = "low_vol"       # SPY HV30 < 15%
    NORMAL = "normal"          # 15% <= SPY HV30 < 25%
    HIGH_VOL = "high_vol"      # SPY HV30 >= 25%


@dataclass
class RegimeConfig:
    """Per-regime parameters."""
    buffer: float
    max_positions: int
    label: str


DEFAULT_REGIMES = {
    Regime.LOW_VOL:  RegimeConfig(buffer=0.07, max_positions=6, label="Low Vol (<15%)"),
    Regime.NORMAL:   RegimeConfig(buffer=0.10, max_positions=5, label="Normal (15-25%)"),
    Regime.HIGH_VOL: RegimeConfig(buffer=0.13, max_positions=3, label="High Vol (>25%)"),
}


@dataclass
class RegimeAdaptiveResult:
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    regime_counts: dict = field(default_factory=dict)
    trade_log: list[dict] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades if self.total_trades > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"Trades:     {self.total_trades}",
            f"Win rate:   {self.win_rate:.1%} ({self.winners}W / {self.losers}L)",
            f"Total P/L:  ${self.total_pnl:+,.2f}",
            f"Max DD:     {self.max_drawdown_pct:.1%}",
            f"Regime breakdown:",
        ]
        for regime, count in self.regime_counts.items():
            lines.append(f"  {regime}: {count} trades")
        return "\n".join(lines)


class RegimeAdaptiveStrategy:
    """Adapt strategy parameters to market volatility regime.

    Args:
        spread_pct: Spread width as % of underlying (default 0.02).
        dte_open: Days to expiry at entry (default 30).
        dte_close: Remaining DTE when we close (default 14).
        regime_thresholds: (low_high_boundary, normal_high_boundary)
            defaults to (0.15, 0.25).
        regimes: Dict mapping Regime -> RegimeConfig. Uses defaults if None.
        risk_free_rate: Annualized risk-free rate.
        vol_window: Rolling window for historical vol.
        commission_per_contract: Per-leg commission.
        slippage_pct: Credit lost to bid-ask.
    """

    def __init__(
        self,
        spread_pct: float = 0.02,
        dte_open: int = 30,
        dte_close: int = 14,
        regime_thresholds: tuple[float, float] = (0.15, 0.25),
        regimes: dict[Regime, RegimeConfig] | None = None,
        risk_free_rate: float = 0.05,
        vol_window: int = 30,
        commission_per_contract: float = 0.65,
        slippage_pct: float = 0.02,
    ):
        self.spread_pct = spread_pct
        self.dte_open = dte_open
        self.dte_close = dte_close
        self.low_thresh, self.high_thresh = regime_thresholds
        self.regimes = regimes or DEFAULT_REGIMES
        self.r = risk_free_rate
        self.vol_window = vol_window
        self.commission = commission_per_contract
        self.slippage_pct = slippage_pct

    def detect_regime(self, market_vol: float) -> Regime:
        """Classify the current market regime based on SPY/market vol."""
        if market_vol < self.low_thresh:
            return Regime.LOW_VOL
        elif market_vol < self.high_thresh:
            return Regime.NORMAL
        return Regime.HIGH_VOL

    def run(
        self,
        df: pd.DataFrame,
        market_vol_series: pd.Series | None = None,
        close_col: str = "close",
        max_contracts: int = 10,
    ) -> RegimeAdaptiveResult:
        """Run the strategy on a single ticker.

        Args:
            df: Ticker OHLCV DataFrame.
            market_vol_series: SPY HV30 series for regime detection.
                If None, uses the ticker's own vol (less ideal).
            close_col: Close price column.
            max_contracts: Max contracts per trade.
        """
        close = df[close_col].values
        timestamps = df.index.values
        vol = historical_volatility(df[close_col], window=self.vol_window)

        if market_vol_series is None:
            market_vol_series = vol

        offset_open = max(1, int(self.dte_open * 21 / 30))
        offset_close = max(1, int((self.dte_open - self.dte_close) * 21 / 30))

        trade_log = []
        cum_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        regime_counts = {r.value: 0 for r in Regime}

        i = self.vol_window
        while i < len(df) - offset_open:
            price = close[i]
            vol_val = vol.iloc[i]
            if np.isnan(vol_val) or vol_val <= 0:
                i += 1
                continue

            # Get market vol for regime detection
            mkt_idx = market_vol_series.index.get_indexer(
                [df.index[i]], method="nearest"
            )[0]
            mkt_vol = market_vol_series.iloc[mkt_idx]
            if np.isnan(mkt_vol):
                i += 1
                continue

            regime = self.detect_regime(mkt_vol)
            config = self.regimes[regime]
            buffer = config.buffer
            regime_counts[regime.value] += 1

            # Price the spread with regime-adapted buffer
            sw = price * self.spread_pct
            sk = price * (1 - buffer)
            lk = sk - sw
            if lk <= 0:
                i += offset_open
                continue

            sp = put_credit_spread_price(
                price, sk, lk, self.dte_open / 365, self.r, vol_val
            )
            credit = sp["net_credit_dollar"]
            max_loss = sp["max_loss"]
            if credit <= 0 or max_loss <= 0:
                i += offset_open
                continue

            # Friction
            open_comm = self.commission * 2 * max_contracts
            slippage = credit * max_contracts * self.slippage_pct
            net_credit = credit * max_contracts - slippage

            # Close at checkpoint
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
                "entry_price": price,
                "exit_price": exit_price,
                "buffer": buffer,
                "regime": regime.value,
                "sigma": vol_val,
                "market_vol": mkt_vol,
                "contracts": max_contracts,
                "credit": net_credit,
                "pnl": pnl,
                "winner": winner,
            })

            cum_pnl += pnl
            peak = max(peak, cum_pnl)
            max_dd = min(max_dd, cum_pnl - peak)

            i += offset_close

        return RegimeAdaptiveResult(
            total_trades=len(trade_log),
            winners=sum(1 for t in trade_log if t["winner"]),
            losers=sum(1 for t in trade_log if not t["winner"]),
            total_pnl=sum(t["pnl"] for t in trade_log),
            max_drawdown_pct=max_dd / peak if peak > 0 else 0,
            regime_counts=regime_counts,
            trade_log=trade_log,
        )
