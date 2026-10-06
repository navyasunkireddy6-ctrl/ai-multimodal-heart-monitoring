"""Service providing rate trend trajectory estimation.

Conforms strictly to docs/API_CONTRACT.md and canonical Pydantic schemas.
"""

from typing import Optional
from backend.models.schemas import (
    TrendPredictRequest,
    TrendPredictResponse,
    TrendLabelEnum,
)
from ml.trend.analyzer import TrendAnalyzer, TrendAnalysisResult, MODEL_VERSION


class TrendService:
    """Service providing rate trend evaluation compatible with the canonical API contract."""

    def __init__(self, slope_threshold_bpm_per_min: float = 3.0):
        self.analyzer = TrendAnalyzer(slope_threshold_bpm_per_min=slope_threshold_bpm_per_min)

    def predict_trend(self, request: TrendPredictRequest) -> TrendPredictResponse:
        """Evaluate rate history sequence and return canonical response."""
        result: TrendAnalysisResult = self.analyzer.analyze(
            timestamps=request.timestamps,
            rates=request.rates,
        )

        return TrendPredictResponse(
            trend=TrendLabelEnum(result.trend),
            slope_bpm_per_min=round(float(result.slope_bpm_per_min), 2),
            confidence=round(float(result.confidence), 4),
            window_duration_seconds=round(float(result.window_duration_seconds), 1),
            disclaimer=result.disclaimer,
        )
