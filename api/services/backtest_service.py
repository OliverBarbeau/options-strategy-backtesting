"""Service layer wrapping tradelab strategies for API use."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from asyncio import Queue

from tradelab.pipeline import DataPipeline
from tradelab.options import historical_volatility
from tradelab.strategies.pullback_entry import PullbackEntryStrategy
from tradelab.strategies.regime_adaptive import RegimeAdaptiveStrategy
from tradelab.strategies.aggressive_pullback import AggressivePullbackStrategy

from api.schemas.backtest import (
    BacktestRequest,
    BacktestResponse,
    BacktestMetrics,
    TickerResult,
    TradeEntry,
)

# Shared pipeline (reuses parquet cache)
_pipe = DataPipeline()
_executor = ThreadPoolExecutor(max_workers=2)

# In-memory task store
_tasks: dict[str, dict] = {}


def _build_strategy(req: BacktestRequest):
    """Instantiate the right strategy from request params."""
    if req.strategy == "pullback":
        return PullbackEntryStrategy(
            buffer=req.buffer,
            spread_pct=req.spread_pct,
            pullback_threshold=req.pullback_threshold,
            dte_open=req.dte_open,
            dte_close=req.dte_close,
        )
    elif req.strategy == "regime_adaptive":
        return RegimeAdaptiveStrategy(
            spread_pct=req.spread_pct,
            dte_open=req.dte_open,
            dte_close=req.dte_close,
        )
    elif req.strategy == "aggressive_pullback":
        return AggressivePullbackStrategy(
            buffer=req.buffer,
            spread_pct=req.spread_pct,
            pullback_threshold=req.pullback_threshold,
            deep_pullback=req.deep_pullback,
            dte_open=req.dte_open,
            dte_close=req.dte_close,
            streak_bonus_threshold=req.streak_bonus_threshold,
            streak_buffer=req.streak_buffer,
        )
    else:  # conservative
        return PullbackEntryStrategy(
            buffer=req.buffer,
            spread_pct=req.spread_pct,
            pullback_threshold=0.0,  # no pullback filter
            dte_open=req.dte_open,
            dte_close=req.dte_close,
        )


def run_backtest_sync(task_id: str, req: BacktestRequest):
    """Run a backtest synchronously (called from thread pool)."""
    import warnings
    warnings.filterwarnings("ignore")

    task = _tasks[task_id]
    task["status"] = "running"
    start_time = time.time()

    try:
        strategy = _build_strategy(req)
        spy_vol = None
        if req.strategy == "regime_adaptive":
            spy_data = _pipe.fetch_stock("SPY", start=req.start_date, end=req.end_date)
            spy_vol = historical_volatility(spy_data["close"], window=30)

        all_trades = []
        ticker_results = []

        for i, ticker in enumerate(req.tickers):
            task["progress"] = i / len(req.tickers)
            task["message"] = f"Running {ticker}..."
            task["current_ticker"] = ticker

            try:
                df = _pipe.fetch_stock(ticker, start=req.start_date, end=req.end_date)
                if len(df) < 100:
                    continue

                if req.strategy == "regime_adaptive" and spy_vol is not None:
                    result = strategy.run(df, market_vol_series=spy_vol, max_contracts=req.max_contracts)
                else:
                    result = strategy.run(df, max_contracts=req.max_contracts)

                for t in result.trade_log:
                    entry = {
                        "date": str(t.get("date", "")),
                        "exit_date": str(t.get("exit_date", "")),
                        "ticker": ticker,
                        "entry_price": t.get("entry_price", 0),
                        "exit_price": t.get("exit_price", 0),
                        "pnl": t.get("pnl", 0),
                        "winner": bool(t.get("winner", False)),
                        "contracts": t.get("contracts", 0),
                        "credit": t.get("credit", 0),
                        "sigma": t.get("sigma", 0),
                        "buffer_used": t.get("buffer_used", None),
                        "pullback_pct": t.get("pullback_pct", None),
                        "stacked": t.get("stacked", None),
                    }
                    all_trades.append(entry)

                avg_pnl = result.total_pnl / result.total_trades if result.total_trades > 0 else 0
                ticker_results.append(TickerResult(
                    ticker=ticker,
                    trades=result.total_trades,
                    win_rate=result.win_rate,
                    total_pnl=result.total_pnl,
                    avg_pnl=avg_pnl,
                ))
            except Exception as e:
                task["message"] = f"Warning: {ticker} failed: {e}"

        # Aggregate metrics
        if all_trades:
            winners = [t for t in all_trades if t["winner"]]
            losers = [t for t in all_trades if not t["winner"]]
            pnls = [t["pnl"] for t in all_trades]

            # Compute max drawdown
            cum = 0
            peak = 0
            max_dd = 0
            for pnl in pnls:
                cum += pnl
                peak = max(peak, cum)
                max_dd = min(max_dd, cum - peak)

            metrics = BacktestMetrics(
                total_trades=len(all_trades),
                winners=len(winners),
                losers=len(losers),
                win_rate=len(winners) / len(all_trades),
                total_pnl=sum(pnls),
                avg_pnl=sum(pnls) / len(all_trades),
                max_drawdown_pct=max_dd / peak if peak > 0 else 0,
                best_trade=max(pnls),
                worst_trade=min(pnls),
                avg_winner=sum(t["pnl"] for t in winners) / len(winners) if winners else 0,
                avg_loser=sum(t["pnl"] for t in losers) / len(losers) if losers else 0,
            )

            # Build equity curve
            cum = 0
            equity_curve = []
            for t in sorted(all_trades, key=lambda x: x["date"]):
                cum += t["pnl"]
                equity_curve.append({"date": t["date"], "equity": cum})
        else:
            metrics = BacktestMetrics(
                total_trades=0, winners=0, losers=0, win_rate=0,
                total_pnl=0, avg_pnl=0, max_drawdown_pct=0,
                best_trade=0, worst_trade=0, avg_winner=0, avg_loser=0,
            )
            equity_curve = []

        duration = time.time() - start_time

        task["result"] = BacktestResponse(
            id=task_id,
            status="completed",
            strategy=req.strategy,
            tickers=req.tickers,
            parameters=req.model_dump(exclude={"tickers", "strategy", "start_date", "end_date"}),
            date_range={"start": req.start_date, "end": req.end_date},
            metrics=metrics,
            ticker_results=ticker_results,
            trade_log=[TradeEntry(**t) for t in all_trades],
            equity_curve=equity_curve,
            duration_seconds=duration,
        )
        task["status"] = "completed"
        task["progress"] = 1.0
        task["message"] = f"Completed in {duration:.1f}s"

    except Exception as e:
        task["status"] = "failed"
        task["message"] = f"Error: {e}"


def submit_backtest(req: BacktestRequest) -> str:
    """Submit a backtest to the thread pool. Returns task ID."""
    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Starting...",
        "current_ticker": "",
        "result": None,
    }
    _executor.submit(run_backtest_sync, task_id, req)
    return task_id


def get_task(task_id: str) -> dict | None:
    return _tasks.get(task_id)
