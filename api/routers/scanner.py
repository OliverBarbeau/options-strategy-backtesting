"""Scanner API routes."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/scanner", tags=["scanner"])


class ScanRequest(BaseModel):
    pullback_only: bool = False
    backtest: bool = False
    top_n: int = 15


@router.post("")
async def run_scan(req: ScanRequest):
    """Scan the market for viable put credit spread candidates."""
    import warnings
    warnings.filterwarnings("ignore")
    from tradelab.scanner import StrategyScanner

    scanner = StrategyScanner()

    if req.backtest:
        results = scanner.scan_and_backtest(n=req.top_n, pullback_only=req.pullback_only)
        return {"type": "backtest", "results": results}
    else:
        results = scanner.scan(pullback_only=req.pullback_only)
        return {
            "type": "scan",
            "total_scanned": len(results),
            "qualified": len([r for r in results if r.qualifies]),
            "results": [
                {
                    "ticker": r.ticker,
                    "sector": r.sector,
                    "price": r.price,
                    "score": r.score,
                    "hv30": r.hv30,
                    "hv30_percentile": r.hv30_percentile,
                    "pullback_pct": r.pullback_pct,
                    "credit_potential": r.credit_potential,
                    "breach_rate": r.breach_rate,
                    "has_earnings_soon": r.has_earnings_soon,
                    "qualifies": r.qualifies,
                    "flags": r.flags,
                }
                for r in results[:req.top_n]
            ],
        }
