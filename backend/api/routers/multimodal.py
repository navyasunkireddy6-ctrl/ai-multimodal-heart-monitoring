"""Multimodal synchronization and dual-rate fusion endpoints."""

from typing import Any, Dict
from fastapi import APIRouter, Depends, Query

from backend.api.deps import get_multimodal_service, get_session_service
from backend.models.schemas import (
    MultimodalAnalyzeRequest,
    MultimodalAnalyzeResponse,
)
from backend.services.multimodal_service import MultimodalService
from backend.services.session_service import SessionService

router = APIRouter(prefix="/multimodal", tags=["Multimodal Fusion"])


@router.post("/analyze", response_model=MultimodalAnalyzeResponse)
def analyze_multimodal_signals(
    request: MultimodalAnalyzeRequest,
    service: MultimodalService = Depends(get_multimodal_service),
    session_service: SessionService = Depends(get_session_service),
) -> MultimodalAnalyzeResponse:
    """Analyze simultaneous ECG and optical PPG signals, compute PAT, discrepancy, and fused quality."""
    return service.analyze(request, session_service=session_service)


@router.get("/demo-frame")
def get_multimodal_demo_frame(
    start_sec: float = Query(0.0, ge=0.0, description="Start playback offset in seconds"),
    duration_sec: float = Query(5.0, ge=1.0, le=30.0, description="Window duration in seconds"),
    record_id: str = Query("100", description="PhysioNet ECG record identifier"),
    service: MultimodalService = Depends(get_multimodal_service),
) -> Dict[str, Any]:
    """Retrieve synchronized paired ECG and optical PPG frame for dashboard streaming."""
    return service.get_synchronized_demo_frame(
        start_sec=start_sec,
        duration_sec=duration_sec,
        record_id=record_id,
    )
