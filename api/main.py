"""TradeLab API: FastAPI backend for options backtesting and simulation.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.routers import backtests, accounts, scanner

app = FastAPI(
    title=settings.app_name,
    description="Options backtesting and simulation",
    version="0.1.0",
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(backtests.router)
app.include_router(accounts.router)
app.include_router(scanner.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "app": settings.app_name}


@app.get("/api/strategies")
async def list_strategies():
    """List available strategies and their default parameters."""
    return {
        "strategies": [
            {
                "id": "pullback",
                "name": "Pullback Entry",
                "description": "Enter on 3%+ pullback from 20-day high. Best risk-adjusted.",
                "defaults": {
                    "buffer": 0.10, "spread_pct": 0.02, "dte_open": 30,
                    "dte_close": 14, "pullback_threshold": 0.03,
                },
            },
            {
                "id": "regime_adaptive",
                "name": "Regime Adaptive",
                "description": "Adapt buffer to SPY vol regime. Highest raw returns.",
                "defaults": {
                    "buffer": 0.10, "spread_pct": 0.02, "dte_open": 30,
                    "dte_close": 14,
                },
            },
            {
                "id": "aggressive_pullback",
                "name": "Aggressive Pullback",
                "description": "Tighter buffer + stacking + streak bonus. Risk-tolerant.",
                "defaults": {
                    "buffer": 0.07, "spread_pct": 0.03, "dte_open": 30,
                    "dte_close": 14, "pullback_threshold": 0.03,
                    "deep_pullback": 0.05, "streak_bonus_threshold": 2,
                    "streak_buffer": 0.06,
                },
            },
            {
                "id": "conservative",
                "name": "Conservative",
                "description": "Fixed 10% buffer every cycle. Simplest strategy.",
                "defaults": {
                    "buffer": 0.10, "spread_pct": 0.02, "dte_open": 30,
                    "dte_close": 14,
                },
            },
        ],
        "tickers": {
            "recommended": ["META", "AVGO", "MSFT", "GOOG", "NVDA", "CAT"],
            "available": [
                "AAPL", "AMD", "AMZN", "AVGO", "BA", "BAC", "CAT", "COST",
                "CVX", "DIA", "GOOG", "HD", "IWM", "JNJ", "JPM", "META",
                "MSFT", "NVDA", "QQQ", "SPY", "UNH", "WMT", "XLE", "XOM",
            ],
        },
    }
