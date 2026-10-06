"""Service wrapping ECG signal processing and generating canonical API responses."""

from typing import List, Optional
import numpy as np

from backend.models.schemas import (
    ECGProcessRequest,
    ECGProcessResponse,
    ECGClassifyRequest,
    ECGClassifyResponse,
    QualityLabelEnum,
)
from ml.ecg.processor import ECGProcessor, ECGAnalysisResult
from ml.ecg.classifier import ECGClassifier, ECGClassificationResult


class ECGService:
    """Service providing ECG processing and ML classification compatible with the canonical API contract."""

    def __init__(self, sampling_rate_hz: float = 360.0, model_path: Optional[str] = None):
        self.sampling_rate_hz = sampling_rate_hz
        self.processor = ECGProcessor(sampling_rate_hz=sampling_rate_hz)
        self.classifier = ECGClassifier(
            model_path=model_path,
            sampling_rate_hz=sampling_rate_hz
        )

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

    def classify_window(self, request: ECGClassifyRequest) -> ECGClassifyResponse:
        """Classify ECG rhythm using active ML classifier pipeline."""
        res: ECGClassificationResult = self.classifier.predict(
            signal=request.signal,
            sampling_rate_hz=request.sampling_rate_hz,
            rr_intervals_ms=request.rr_intervals_ms,
        )

        return ECGClassifyResponse(
            rhythm_class=res.rhythm_class,
            rhythm_confidence=round(float(res.rhythm_confidence), 4),
            model_version=res.model_version,
            features={k: round(float(v), 3) for k, v in res.features.items()},
            disclaimer=res.disclaimer,
        )
