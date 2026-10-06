"""Service providing PPG signal processing and optical quality assessment.

Conforms strictly to the canonical API contract in docs/API_CONTRACT.md.
"""

from typing import Optional
import numpy as np

from backend.models.schemas import (
    PPGProcessRequest,
    PPGProcessResponse,
    PPGQualityRequest,
    PPGQualityResponse,
    QualityLabelEnum,
)
from ml.ppg.processor import PPGProcessor, PPGAnalysisResult


class PPGService:
    """Service providing PPG processing methods compatible with the canonical API contract."""

    def __init__(self, sampling_rate_hz: float = 30.0):
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.processor = PPGProcessor(sampling_rate_hz=self.sampling_rate_hz)

    def process_window(self, request: PPGProcessRequest) -> PPGProcessResponse:
        """Process an optical PPG window and return canonical response."""
        fs = request.sampling_rate_hz
        if fs != self.processor.sampling_rate_hz:
            processor = PPGProcessor(sampling_rate_hz=fs)
        else:
            processor = self.processor

        result: PPGAnalysisResult = processor.process(request.signal)

        # Pulse rate is None if quality is POOR per canonical contract
        pulse_rate = (
            round(float(result.pulse_rate_bpm), 1)
            if (result.pulse_rate_bpm is not None and result.quality_label != QualityLabelEnum.POOR.value)
            else None
        )

        return PPGProcessResponse(
            pulse_rate_bpm=pulse_rate,
            signal_quality=round(float(result.signal_quality), 4),
            quality_label=QualityLabelEnum(result.quality_label),
            peaks=[int(p) for p in result.peaks],
            filtered_signal=[round(float(v), 4) for v in result.filtered_signal],
        )

    def assess_quality(self, request: PPGQualityRequest) -> PPGQualityResponse:
        """Assess signal quality, SNR, and usability for optical PPG window."""
        fs = request.sampling_rate_hz
        if fs != self.processor.sampling_rate_hz:
            processor = PPGProcessor(sampling_rate_hz=fs)
        else:
            processor = self.processor

        result: PPGAnalysisResult = processor.process(request.signal)

        return PPGQualityResponse(
            signal_quality=round(float(result.signal_quality), 4),
            quality_label=QualityLabelEnum(result.quality_label),
            snr_db=round(float(result.snr_db), 2),
            is_usable=result.is_usable,
            message=result.message,
        )
