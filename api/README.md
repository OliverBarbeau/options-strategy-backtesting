# TradeLab API

FastAPI backend wrapping the `tradelab` library.

## Run

```bash
PYTHONPATH=. uvicorn api.main:app --reload --port 8000
```

Auto-docs at http://localhost:8000/docs.

## Structure

```
api/
├── main.py              # FastAPI app, CORS, routers
├── config.py            # Settings via pydantic-settings
├── routers/
│   ├── backtests.py     # POST backtest + SSE progress
│   ├── accounts.py      # List/detail/MTM/advance
│   ├── broker.py        # Schwab status/scan/trade
│   └── scanner.py       # Market scan endpoint
├── schemas/             # Pydantic request/response models
└── services/            # Business logic wrapping tradelab
```

## Endpoints

### Backtests
- `POST /api/backtests` — Submit a backtest. Returns `{id, status}`.
- `GET /api/backtests/{id}/stream` — SSE progress stream.
- `GET /api/backtests/{id}` — Completed result.

### Accounts
- `GET /api/accounts` — List all simulation accounts.
- `GET /api/accounts/{name}` — Full detail.
- `GET /api/accounts/{name}/mtm` — Mark-to-market open positions.
- `POST /api/accounts/{name}/advance` — Advance to date.

### Broker (Schwab)
- `GET /api/broker/status` — Schwab account summary.
- `GET /api/broker/positions` — Live option positions.
- `POST /api/broker/scan` — Find best spread from live chain.
- `POST /api/broker/trade` — Place order.

### Scanner
- `POST /api/scanner` — Market scan with optional backtest.

### Utility
- `GET /api/health` — Health check.
- `GET /api/strategies` — List strategies with default params.

## Architecture Notes

**Threading**: CPU-bound tradelab work (pandas/numpy) runs in a
`ThreadPoolExecutor` with 2 workers. FastAPI endpoints stay async for I/O
(DB queries, SSE streams).

**Task storage**: In-memory dict indexed by task ID. Tasks persist for the
lifetime of the server process. For a single-user tool this is sufficient; if
we need multi-user or restart recovery, migrate to SQLite.

**SSE**: Uses `sse-starlette` for streaming backtest progress. Client opens
`EventSource('/api/backtests/{id}/stream')`, receives `progress` events with
a percentage + message, then a final `complete` event with the full result.

**CORS**: Configured to allow `http://localhost:3000` by default. Override
with `TRADELAB_CORS_ORIGINS` env var.
