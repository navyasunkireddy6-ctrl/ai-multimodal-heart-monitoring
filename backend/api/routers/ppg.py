"""PPG signal processing and quality assessment endpoints.

Conforms to Sections 3.4 and 3.5 of docs/API_CONTRACT.md.
"""

from fastapi import APIRouter, Depends

from backend.api.deps import get_ppg_service
from backend.models.schemas import (
    PPGProcessRequest,
    PPGProcessResponse,
    PPGQualityRequest,
    PPGQualityResponse,
)
from backend.services.ppg_service import PPGService

router = APIRouter(prefix="/ppg", tags=["PPG"])


@router.post("/process", response_model=PPGProcessResponse)
def process_ppg_signal(
    request: PPGProcessRequest,
    service: PPGService = Depends(get_ppg_service),
) -> PPGProcessResponse:
    """Filter raw optical PPG waveform, detect systolic peaks, calculate pulse rate and quality."""
    return service.process_window(request)


@router.post("/quality", response_model=PPGQualityResponse)
def assess_ppg_quality(
    request: PPGQualityRequest,
    service: PPGService = Depends(get_ppg_service),
) -> PPGQualityResponse:
    """Evaluate optical PPG signal quality index based on periodicity, amplitude consistency, and spectral power."""
    return service.assess_quality(request)
