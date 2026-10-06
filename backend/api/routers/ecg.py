"""ECG signal processing and arrhythmia classification endpoints.

Conforms to Sections 3.2 and 3.3 of docs/API_CONTRACT.md.
"""

from fastapi import APIRouter, Depends

from backend.api.deps import get_ecg_service
from backend.models.schemas import (
    ECGClassifyRequest,
    ECGClassifyResponse,
    ECGProcessRequest,
    ECGProcessResponse,
)
from backend.services.ecg_service import ECGService

router = APIRouter(prefix="/ecg", tags=["ECG"])


@router.post("/process", response_model=ECGProcessResponse)
def process_ecg_signal(
    request: ECGProcessRequest,
    service: ECGService = Depends(get_ecg_service),
) -> ECGProcessResponse:
    """Filter raw ECG window, detect R-peaks, calculate R-R intervals, heart rate, and SQI."""
    return service.process_window(request)


@router.post("/classify", response_model=ECGClassifyResponse)
def classify_ecg_rhythm(
    request: ECGClassifyRequest,
    service: ECGService = Depends(get_ecg_service),
) -> ECGClassifyResponse:
    """Classify an ECG segment into trained rhythm classes using extracted physiological features."""
    return service.classify_window(request)
