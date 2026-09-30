"""Service wrapping ECG signal processing and generating canonical API responses."""

from typing import List, Optional
import numpy as np

from backend.models.schemas import (
    ECGProcessRequest,
    ECGProcessResponse,
    QualityLabelEnum,
)
from ml.ecg.processor import ECGProcessor, ECGAnalysisResult


class ECGService:
    """Service providing ECG processing methods compatible with the canonical API contract."""

    def __init__(self, sampling_rate_hz: float = 360.0):
        self.sampling_rate_hz = sampling_rate_hz
        self.processor = ECGProcessor(sampling_rate_hz=sampling_rate_hz)

    def process_window(self, request: ECGProcessRequest) -> ECGProcessResponse:
        """Process an ECG window and return canonical response."""
        # Use requested sampling rate if differing from default
        if request.sampling_rate_hz != self.processor.sampling_rate_hz:
            processor = ECGProcessor(sampling_rate_hz=request.sampling_rate_hz)
        else:
            processor = self.processor

        result: ECGAnalysisResult = processor.process(request.signal)

        # Map to canonical QualityLabelEnum
        try:
            quality_enum = QualityLabelEnum(result.quality_label)
        except ValueError:
            quality_enum = QualityLabelEnum.POOR

        return ECGProcessResponse(
            heart_rate_bpm=result.heart_rate_bpm,
            signal_quality=result.signal_quality,
            quality_label=quality_enum,
            r_peaks=result.r_peaks.tolist(),
            rr_intervals_ms=[round(float(rr), 1) for rr in result.rr_intervals_ms],
            filtered_signal=[round(float(v), 4) for v in result.filtered_signal],
        )
