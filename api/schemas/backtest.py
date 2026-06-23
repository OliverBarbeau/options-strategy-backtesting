"""Pydantic schemas for backtest API."""

from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    strategy: str = Field(description="Strategy name: pullback, regime_adaptive, aggressive_pullback, conservative")
    tickers: list[str] = Field(default=["META", "AVGO", "MSFT", "GOOG", "NVDA", "CAT"])
    start_date: str = Field(default="2020-01-01")
    end_date: str = Field(default="2026-04-04")

    # Strategy parameters
    buffer: float = Field(default=0.10, ge=0.01, le=0.30)
    spread_pct: float = Field(default=0.02, ge=0.01, le=0.10)
    dte_open: int = Field(default=30, ge=7, le=90)
    dte_close: int = Field(default=14, ge=1, le=45)
    pullback_threshold: float = Field(default=0.03, ge=0.01, le=0.15)
    max_contracts: int = Field(default=10, ge=1, le=50)

    # Aggressive-only params
    deep_pullback: float = Field(default=0.05, ge=0.02, le=0.20)
    streak_bonus_threshold: int = Field(default=2, ge=1, le=10)
    streak_buffer: float = Field(default=0.06, ge=0.01, le=0.20)


class TradeEntry(BaseModel):
    date: str
    exit_date: str
    ticker: str
    entry_price: float
    exit_price: float
    pnl: float
    winner: bool
    contracts: int
    credit: float
    sigma: float
    buffer_used: float | None = None
    pullback_pct: float | None = None
    stacked: bool | None = None


class BacktestMetrics(BaseModel):
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    max_drawdown_pct: float
    best_trade: float
    worst_trade: float
    avg_winner: float
    avg_loser: float


class TickerResult(BaseModel):
    ticker: str
    trades: int
    win_rate: float
    total_pnl: float
    avg_pnl: float


class BacktestResponse(BaseModel):
    id: str
    status: str
    strategy: str
    tickers: list[str]
    parameters: dict
    date_range: dict
    metrics: BacktestMetrics | None = None
    ticker_results: list[TickerResult] = []
    trade_log: list[TradeEntry] = []
    equity_curve: list[dict] = []
    duration_seconds: float = 0


class BacktestProgress(BaseModel):
    progress: float  # 0.0 - 1.0
    message: str
    ticker: str = ""
