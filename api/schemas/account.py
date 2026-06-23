"""Pydantic schemas for account API."""

from pydantic import BaseModel


class AccountSummary(BaseModel):
    name: str
    strategy: str
    account_type: str  # "live" or "backtest"
    starting_capital: float
    equity: float
    balance: float
    locked: float
    total_pnl: float
    total_return_pct: float
    total_trades: int
    win_rate: float
    open_positions: int
    last_advanced_date: str
    created_date: str


class PositionDetail(BaseModel):
    id: str
    ticker: str
    entry_date: str
    entry_price: float
    short_strike: float
    long_strike: float
    contracts: int
    collateral: float
    credit_received: float
    buffer: float
    close_target_date: str
    notes: str = ""


class MtmPosition(BaseModel):
    id: str
    ticker: str
    entry_date: str
    entry_price: float
    current_price: float
    price_change: float
    short_strike: float
    buffer_remaining: float
    contracts: int
    credit: float
    close_cost: float
    unrealized_pnl: float
    dte_remaining: int
    close_target: str


class TradeDetail(BaseModel):
    id: str
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    contracts: int
    pnl: float
    winner: bool
    exit_reason: str
    buffer: float


class AccountDetail(BaseModel):
    summary: AccountSummary
    positions: list[PositionDetail]
    recent_trades: list[TradeDetail]
    equity_curve: list[dict]
    config_summary: str = ""  # human-readable config description


class AdvanceRequest(BaseModel):
    end_date: str | None = None


class BrokerScanRequest(BaseModel):
    ticker: str
    buffer: float = 0.10
    spread_pct: float = 0.02
    dte: int = 30


class BrokerTradeRequest(BaseModel):
    ticker: str
    quantity: int = 1
    buffer: float = 0.10
    spread_pct: float = 0.02
    dte: int = 30
    live: bool = False
