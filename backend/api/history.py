"""backend/api/history.py – History CRUD endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.core.schemas import HistoryItem, HistoryListResponse, WritingMode
from backend.db import database as db

router = APIRouter()


@router.get("/history", response_model=HistoryListResponse, summary="List history")
async def list_history(
    session_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> HistoryListResponse:
    """Retrieve paginated humanization history."""
    records, total = await db.get_history(session_id=session_id, page=page, page_size=page_size)
    items = [
        HistoryItem(
            id=r.id,
            session_id=r.session_id,
            original_snippet=r.original_snippet or "",
            humanized_snippet=r.humanized_snippet or "",
            mode=WritingMode(r.mode) if r.mode else WritingMode.NATURAL,
            intensity=r.intensity or "balanced",  # type: ignore[arg-type]
            overall_score=r.overall_score or 0.0,
            created_at=r.created_at,
        )
        for r in records
    ]
    return HistoryListResponse(items=items, total=total, page=page, page_size=page_size)


@router.get("/history/{record_id}", summary="Get a single history record")
async def get_history_item(record_id: int):
    """Retrieve one full humanization record by id."""
    record = await db.get_history_item(record_id)
    if not record:
        raise HTTPException(404, f"History record {record_id} not found.")
    return {
        "id": record.id,
        "original_text": record.original_text,
        "humanized_text": record.humanized_text,
        "mode": record.mode,
        "intensity": record.intensity,
        "overall_score": record.overall_score,
        "created_at": record.created_at,
    }


@router.delete("/history/{record_id}", summary="Delete a history record")
async def delete_history_item(record_id: int):
    deleted = await db.soft_delete_history_item(record_id)
    if not deleted:
        raise HTTPException(404, f"Record {record_id} not found.")
    return {"deleted": True, "id": record_id}


@router.delete("/history", summary="Clear all history")
async def clear_history(session_id: str | None = Query(None)):
    count = await db.delete_all_history(session_id=session_id)
    return {"deleted_count": count}
