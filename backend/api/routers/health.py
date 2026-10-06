"""Health check endpoint conforming to Section 3.1 of docs/API_CONTRACT.md."""

from datetime import datetime, timezone
from fastapi import APIRouter

from backend.models.schemas import HealthResponse
from ml.ecg.classifier import MODEL_VERSION as ECG_MODEL_VERSION
from ml.trend.analyzer import MODEL_VERSION as TREND_MODEL_VERSION

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return backend service status, runtime metadata, and active ML model versions."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0",
        models_loaded={
            "ecg_classifier": ECG_MODEL_VERSION,
            "trend_analyzer": TREND_MODEL_VERSION,
        },
        disclaimer="Research prototype — not intended for medical diagnosis.",
    )
