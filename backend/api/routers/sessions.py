"""Session management endpoints conforming to Section 3.7 of docs/API_CONTRACT.md."""

from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from backend.api.deps import get_session_service
from backend.models.schemas import (
    CanonicalMeasurement,
    SessionCreateRequest,
    SessionDetailResponse,
    SessionListResponse,
    SessionResponse,
)
from backend.services.session_service import SessionService

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(
    request: SessionCreateRequest,
    service: SessionService = Depends(get_session_service),
) -> SessionResponse:
    """Create a new cardiac monitoring session."""
    return service.create_session(request)


@router.get("", response_model=SessionListResponse)
def list_sessions(
    limit: int = Query(50, ge=1, le=500, description="Max sessions to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    service: SessionService = Depends(get_session_service),
) -> SessionListResponse:
    """List monitoring sessions with pagination."""
    return service.list_sessions(limit=limit, offset=offset)


@router.get("/{session_id}", response_model=SessionDetailResponse)
def get_session(
    session_id: str,
    service: SessionService = Depends(get_session_service),
) -> SessionDetailResponse:
    """Get full details of a session including historical measurements."""
    detail = service.get_session(session_id)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return detail


@router.delete("/{session_id}")
def delete_session(
    session_id: str,
    service: SessionService = Depends(get_session_service),
) -> Dict[str, Any]:
    """Delete a session by ID."""
    deleted = service.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return {"deleted": True, "session_id": session_id}


@router.get("/{session_id}/export")
def export_session_csv(
    session_id: str,
    service: SessionService = Depends(get_session_service),
) -> Response:
    """Export session measurements as a CSV download."""
    csv_content = service.export_csv(session_id)
    if csv_content is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="session_{session_id}.csv"'},
    )


@router.post("/{session_id}/measurements", response_model=SessionResponse)
def add_session_measurement(
    session_id: str,
    measurement: CanonicalMeasurement,
    service: SessionService = Depends(get_session_service),
) -> SessionResponse:
    """Record a canonical measurement into an active session."""
    success = service.add_measurement(session_id, measurement)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    detail = service.get_session(session_id)
    return SessionResponse(
        session_id=detail.session_id,
        created_at=detail.created_at,
        mode=detail.mode,
        measurement_count=len(detail.measurements),
        avg_heart_rate_bpm=None if not detail.measurements else (
            round(sum(m.heart_rate_bpm for m in detail.measurements if m.heart_rate_bpm is not None) /
                  len([m for m in detail.measurements if m.heart_rate_bpm is not None]), 1)
            if [m for m in detail.measurements if m.heart_rate_bpm is not None] else None
        ),
        avg_pulse_rate_bpm=None if not detail.measurements else (
            round(sum(m.pulse_rate_bpm for m in detail.measurements if m.pulse_rate_bpm is not None) /
                  len([m for m in detail.measurements if m.pulse_rate_bpm is not None]), 1)
            if [m for m in detail.measurements if m.pulse_rate_bpm is not None] else None
        ),
        notes=detail.notes,
    )
