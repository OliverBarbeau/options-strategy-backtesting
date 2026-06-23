"""Account API routes."""

from fastapi import APIRouter, HTTPException

from api.schemas.account import (
    AccountSummary,
    AccountDetail,
    MtmPosition,
    AdvanceRequest,
)
from api.services.account_service import (
    list_accounts,
    get_account,
    mark_to_market,
    advance_account,
)

router = APIRouter(prefix="/api/accounts", tags=["accounts"])


@router.get("", response_model=list[AccountSummary])
async def get_accounts():
    """List all simulation accounts."""
    return list_accounts()


@router.get("/{name}", response_model=AccountDetail)
async def get_account_detail(name: str):
    """Get full account detail including positions, trades, equity curve."""
    result = get_account(name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Account '{name}' not found")
    return result


@router.get("/{name}/mtm", response_model=list[MtmPosition])
async def get_mtm(name: str):
    """Mark-to-market all open positions with current prices."""
    return mark_to_market(name)


@router.post("/{name}/advance", response_model=AccountSummary)
async def advance(name: str, req: AdvanceRequest):
    """Advance an account to a specific date (or today)."""
    result = advance_account(name, req.end_date)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Account '{name}' not found")
    return result


@router.post("/advance-all")
async def advance_all(req: AdvanceRequest):
    """Advance all accounts to a specific date (or today)."""
    accounts = list_accounts()
    results = []
    for acct in accounts:
        try:
            result = advance_account(acct.name, req.end_date)
            if result:
                results.append({"name": acct.name, "status": "ok", "equity": result.equity})
            else:
                results.append({"name": acct.name, "status": "not_found"})
        except Exception as e:
            results.append({"name": acct.name, "status": "error", "error": str(e)})
    return {"advanced": len([r for r in results if r["status"] == "ok"]), "results": results}
