"""Backtest API routes."""

import asyncio
import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.schemas.backtest import BacktestRequest, BacktestResponse, BacktestProgress
from api.services.backtest_service import submit_backtest, get_task

router = APIRouter(prefix="/api/backtests", tags=["backtests"])


@router.post("", response_model=dict)
async def create_backtest(req: BacktestRequest):
    """Submit a new backtest. Returns task ID for streaming progress."""
    task_id = submit_backtest(req)
    return {"id": task_id, "status": "submitted"}


@router.get("/{task_id}/stream")
async def stream_backtest(task_id: str):
    """SSE stream of backtest progress and final results."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        while True:
            task = get_task(task_id)
            if task is None:
                break

            if task["status"] in ("running", "pending"):
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "progress": task["progress"],
                        "message": task["message"],
                        "ticker": task.get("current_ticker", ""),
                    }),
                }
                await asyncio.sleep(0.5)

            elif task["status"] == "completed":
                result = task["result"]
                yield {
                    "event": "complete",
                    "data": result.model_dump_json() if result else "{}",
                }
                break

            elif task["status"] == "failed":
                yield {
                    "event": "error",
                    "data": json.dumps({"message": task["message"]}),
                }
                break

            else:
                await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


@router.get("/{task_id}", response_model=BacktestResponse)
async def get_backtest(task_id: str):
    """Get backtest result (after completion)."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "completed":
        raise HTTPException(status_code=202, detail=f"Task status: {task['status']}")
    return task["result"]
