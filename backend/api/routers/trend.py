"""Short-term heart/pulse rate trend prediction endpoint.

Conforms to Section 3.6 of docs/API_CONTRACT.md.
"""

from fastapi import APIRouter, Depends

from backend.api.deps import get_trend_service
from backend.models.schemas import TrendPredictRequest, TrendPredictResponse
from backend.services.trend_service import TrendService

router = APIRouter(prefix="/trend", tags=["Trend"])


@router.post("/predict", response_model=TrendPredictResponse)
def predict_rate_trend(
    request: TrendPredictRequest,
    service: TrendService = Depends(get_trend_service),
) -> TrendPredictResponse:
    """Evaluate rate sequence history over a moving time window to determine rate trajectory."""
    return service.predict_trend(request)
