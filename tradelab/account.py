"""Simulated brokerage account with JSON persistence.

Tracks balance, open positions, trade history, and equity curve.
State is saved to disk after every mutation so it survives between sessions.

Usage::

    account = SimulatedAccount.load_or_create("accounts/pullback_meta.json", 25000)
    account.open_position(...)
    account.close_position(...)
    account.snapshot(date, price_fn)
    print(account.status())
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@dataclass
class OpenPosition:
    id: str
    ticker: str
    entry_date: str          # ISO format
    close_target_date: str   # when we plan to close (14 DTE mark)
    entry_price: float
    short_strike: float
    long_strike: float
    contracts: int
    collateral: float        # locked capital
    credit_received: float   # net credit after slippage
    buffer: float
    spread_type: str = "put_credit_spread"
    entry_vol: float = 0.0
    entry_regime: str = ""
    notes: str = ""


@dataclass
class ClosedTrade:
    id: str
    ticker: str
    spread_type: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    short_strike: float
    long_strike: float
    contracts: int
    collateral: float
    credit_received: float
    close_cost: float
    pnl: float
    friction: float
    winner: bool
    exit_reason: str         # "checkpoint", "expiry", "emergency", "manual"
    buffer: float = 0.0
    entry_vol: float = 0.0


@dataclass
class EquitySnapshot:
    date: str
    equity: float
    balance: float
    locked: float
    open_positions: int
    cumulative_pnl: float


class SimulatedAccount:
    """Persistent simulated trading account.

    All state is stored in a JSON file and auto-saved on mutation.
    """

    ACCOUNT_TYPES = ("live", "backtest")

    def __init__(
        self,
        filepath: str,
        starting_capital: float = 25000,
        name: str = "",
        strategy: str = "",
        account_type: str = "live",
    ):
        self.filepath = filepath
        self.name = name or Path(filepath).stem
        self.strategy = strategy
        self.account_type = account_type if account_type in self.ACCOUNT_TYPES else "live"
        self.starting_capital = starting_capital
        self.balance = starting_capital  # available cash
        self.locked = 0.0               # collateral in open positions
        self.positions: list[OpenPosition] = []
        self.trades: list[ClosedTrade] = []
        self.equity_curve: list[EquitySnapshot] = []
        self.last_advanced_date: str = ""
        self.created_date: str = datetime.now().isoformat()
        self._next_id = 1
        self._commission = 0.65
        self._slippage_pct = 0.02

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self):
        """Save account state to JSON."""
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "name": self.name,
            "strategy": self.strategy,
            "account_type": self.account_type,
            "starting_capital": self.starting_capital,
            "balance": self.balance,
            "locked": self.locked,
            "positions": [asdict(p) for p in self.positions],
            "trades": [asdict(t) for t in self.trades],
            "equity_curve": [asdict(e) for e in self.equity_curve],
            "last_advanced_date": self.last_advanced_date,
            "created_date": self.created_date,
            "_next_id": self._next_id,
        }
        with open(self.filepath, "w") as f:
            json.dump(state, f, indent=2, default=_json_default)

    @classmethod
    def load(cls, filepath: str) -> "SimulatedAccount":
        """Load an existing account from JSON."""
        with open(filepath) as f:
            state = json.load(f)

        acct = cls(filepath, state["starting_capital"])
        acct.name = state["name"]
        acct.strategy = state.get("strategy", "")
        acct.account_type = state.get("account_type", "live")
        acct.balance = state["balance"]
        acct.locked = state["locked"]
        acct.positions = [OpenPosition(**p) for p in state["positions"]]
        acct.trades = [ClosedTrade(**t) for t in state["trades"]]
        acct.equity_curve = [EquitySnapshot(**e) for e in state["equity_curve"]]
        acct.last_advanced_date = state["last_advanced_date"]
        acct.created_date = state.get("created_date", "")
        acct._next_id = state.get("_next_id", len(acct.trades) + 1)
        return acct

    @classmethod
    def load_or_create(
        cls,
        filepath: str,
        starting_capital: float = 25000,
        name: str = "",
        strategy: str = "",
        account_type: str = "live",
    ) -> "SimulatedAccount":
        """Load if exists, otherwise create new."""
        if os.path.exists(filepath):
            return cls.load(filepath)
        acct = cls(filepath, starting_capital, name, strategy, account_type=account_type)
        acct.save()
        return acct

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def equity(self) -> float:
        return self.balance + self.locked

    @property
    def buying_power(self) -> float:
        return self.balance

    @property
    def total_pnl(self) -> float:
        return self.equity - self.starting_capital

    @property
    def total_trades_count(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.winner) / len(self.trades)

    @property
    def open_tickers(self) -> set[str]:
        return {p.ticker for p in self.positions}

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    def open_position(
        self,
        ticker: str,
        date: str,
        entry_price: float,
        short_strike: float,
        long_strike: float,
        contracts: int,
        credit_per_contract: float,
        collateral_per_contract: float,
        close_target_date: str,
        buffer: float = 0.10,
        entry_vol: float = 0.0,
        entry_regime: str = "",
        notes: str = "",
    ) -> OpenPosition | None:
        """Open a new position. Returns the position or None if insufficient funds."""
        total_collateral = collateral_per_contract * contracts
        total_credit = credit_per_contract * contracts

        # Friction
        open_commission = self._commission * 2 * contracts
        slippage = total_credit * self._slippage_pct
        net_credit = total_credit - slippage

        if total_collateral + open_commission > self.balance:
            return None

        pos_id = f"{ticker}_{self._next_id}"
        self._next_id += 1

        pos = OpenPosition(
            id=pos_id,
            ticker=ticker,
            entry_date=date,
            close_target_date=close_target_date,
            entry_price=entry_price,
            short_strike=short_strike,
            long_strike=long_strike,
            contracts=contracts,
            collateral=total_collateral,
            credit_received=net_credit,
            buffer=buffer,
            entry_vol=entry_vol,
            entry_regime=entry_regime,
            notes=notes,
        )

        self.balance -= total_collateral
        self.balance += net_credit
        self.locked += total_collateral
        self.positions.append(pos)
        self.save()
        return pos

    def close_position(
        self,
        pos_id: str,
        date: str,
        exit_price: float,
        close_cost: float,
        exit_reason: str = "checkpoint",
    ) -> ClosedTrade | None:
        """Close an open position by ID."""
        pos = None
        for p in self.positions:
            if p.id == pos_id:
                pos = p
                break
        if pos is None:
            return None

        close_commission = self._commission * 2 * pos.contracts
        close_slippage = close_cost * self._slippage_pct
        pnl = pos.credit_received - close_cost - close_slippage - close_commission
        # Friction includes entry slippage (reverse-engineered from net credit)
        # plus exit slippage plus commissions on both legs.
        entry_slippage = pos.credit_received * self._slippage_pct / (1 - self._slippage_pct)
        friction = close_commission + entry_slippage + close_slippage

        trade = ClosedTrade(
            id=pos.id,
            ticker=pos.ticker,
            spread_type=pos.spread_type,
            entry_date=pos.entry_date,
            exit_date=date,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            short_strike=pos.short_strike,
            long_strike=pos.long_strike,
            contracts=pos.contracts,
            collateral=pos.collateral,
            credit_received=pos.credit_received,
            close_cost=close_cost,
            pnl=pnl,
            friction=friction,
            winner=pnl > 0,
            exit_reason=exit_reason,
            buffer=pos.buffer,
            entry_vol=pos.entry_vol,
        )

        # Accounting on close:
        #   At open:  balance -= collateral; balance += net_credit (stored in credit_received)
        #   At close: balance += collateral (return margin); balance -= close_cost;
        #             balance -= close_slippage; balance -= close_commission
        # Net P/L across open+close = credit_received - close_cost - close_slippage - close_commission (= pnl)
        # But the credit was already added at open, so we must NOT add it again here.
        self.balance += pos.collateral - close_cost - close_slippage - close_commission
        self.locked -= pos.collateral
        self.positions = [p for p in self.positions if p.id != pos_id]
        self.trades.append(trade)
        self.save()
        return trade

    def snapshot(self, date: str):
        """Record an equity snapshot."""
        snap = EquitySnapshot(
            date=date,
            equity=self.equity,
            balance=self.balance,
            locked=self.locked,
            open_positions=len(self.positions),
            cumulative_pnl=self.total_pnl,
        )
        self.equity_curve.append(snap)
        self.last_advanced_date = date
        self.save()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def status(self) -> str:
        """Human-readable account status."""
        lines = [
            f"Account: {self.name}",
            f"Strategy: {self.strategy}",
            f"Last updated: {self.last_advanced_date}",
            f"",
            f"Equity:         ${self.equity:>12,.2f}",
            f"  Balance:      ${self.balance:>12,.2f}",
            f"  Locked:       ${self.locked:>12,.2f}",
            f"Starting cap:   ${self.starting_capital:>12,.2f}",
            f"Total P/L:      ${self.total_pnl:>+12,.2f} ({self.total_pnl / self.starting_capital:+.1%})",
            f"",
            f"Trades: {self.total_trades_count}  Win rate: {self.win_rate:.1%}",
            f"Open positions: {len(self.positions)}",
        ]

        if self.positions:
            lines.append("")
            for pos in self.positions:
                lines.append(
                    f"  {pos.ticker:<5} {pos.contracts}x  "
                    f"strikes {pos.short_strike:.2f}/{pos.long_strike:.2f}  "
                    f"col ${pos.collateral:,.0f}  "
                    f"close by {pos.close_target_date}"
                )

        if self.trades:
            recent = self.trades[-5:]
            lines.append("")
            lines.append("Recent trades:")
            for t in recent:
                marker = "W" if t.winner else "L"
                lines.append(
                    f"  [{marker}] {t.ticker:<5} {t.entry_date[:10]} -> {t.exit_date[:10]}  "
                    f"P/L ${t.pnl:>+8,.2f}  ({t.exit_reason})"
                )

        if self.equity_curve:
            peak = max(e.equity for e in self.equity_curve)
            current = self.equity_curve[-1].equity
            dd = (current - peak) / peak
            lines.append(f"\nMax equity:     ${peak:>12,.2f}")
            lines.append(f"Current DD:     {dd:>12.1%}")

        return "\n".join(lines)
