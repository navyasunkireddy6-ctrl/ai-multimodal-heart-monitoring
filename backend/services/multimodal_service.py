"""Service providing multimodal ECG and optical PPG synchronization and comparative analysis.

Conforms strictly to docs/API_CONTRACT.md and canonical Pydantic schemas.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from backend.models.schemas import (
    CanonicalMeasurement,
    ModeEnum,
    MultimodalAnalyzeRequest,
    MultimodalAnalyzeResponse,
    QualityLabelEnum,
    TrendLabelEnum,
)
from backend.services.session_service import SessionService
from ml.ecg.classifier import ECGClassifier
from ml.ecg.dataset import ECGDatasetLoader
from ml.ecg.processor import ECGProcessor
from ml.fusion.synchronizer import (
    MultimodalSynchronizer,
    MultimodalSyncResult,
    MODEL_VERSION,
)
from ml.ppg.processor import PPGProcessor
from ml.ppg.synthetic import generate_synthetic_ppg
from ml.trend.analyzer import TrendAnalyzer


class MultimodalService:
    """Service orchestrating simultaneous ECG and PPG processing, synchronization, and fusion."""

    def __init__(
        self,
        ecg_sampling_rate_hz: float = 360.0,
        ppg_sampling_rate_hz: float = 30.0,
        session_service: Optional[SessionService] = None,
    ):
        self.ecg_processor = ECGProcessor(sampling_rate_hz=ecg_sampling_rate_hz)
        self.ecg_classifier = ECGClassifier(sampling_rate_hz=ecg_sampling_rate_hz)
        self.ppg_processor = PPGProcessor(sampling_rate_hz=ppg_sampling_rate_hz)
        self.synchronizer = MultimodalSynchronizer()
        self.trend_analyzer = TrendAnalyzer()
        self.dataset_loader = ECGDatasetLoader()
        self.session_service = session_service

    def analyze(
        self, request: MultimodalAnalyzeRequest, session_service: Optional[SessionService] = None
    ) -> MultimodalAnalyzeResponse:
        """Execute end-to-end multimodal analysis across simultaneous ECG and PPG windows."""
        # 1. ECG Signal Processing
        if request.ecg_sampling_rate_hz != self.ecg_processor.sampling_rate_hz:
            ecg_proc = ECGProcessor(sampling_rate_hz=request.ecg_sampling_rate_hz)
        else:
            ecg_proc = self.ecg_processor

        ecg_res = ecg_proc.process(request.ecg_signal)

        # 2. ECG Arrhythmia Rhythm Classification
        clf_res = self.ecg_classifier.predict(
            signal=request.ecg_signal,
            sampling_rate_hz=request.ecg_sampling_rate_hz,
            rr_intervals_ms=ecg_res.rr_intervals_ms,
        )

        # 3. Optical PPG Signal Processing & Quality Assessment
        if request.ppg_sampling_rate_hz != self.ppg_processor.sampling_rate_hz:
            ppg_proc = PPGProcessor(sampling_rate_hz=request.ppg_sampling_rate_hz)
        else:
            ppg_proc = self.ppg_processor

        ppg_res = ppg_proc.process(request.ppg_signal)

        # 4. Multimodal Temporal Alignment, PAT Estimation & SQI Fusion
        sync_res = self.synchronizer.synchronize(
            ecg_result=ecg_res,
            ppg_result=ppg_res,
            ecg_sampling_rate_hz=request.ecg_sampling_rate_hz,
            ppg_sampling_rate_hz=request.ppg_sampling_rate_hz,
        )

        # 5. Canonical Measurement Generation
        mode = ModeEnum.OFFLINE_DEMO if request.record_id else ModeEnum.SMARTPHONE_PPG
        now_iso = datetime.now(timezone.utc).isoformat()
        canonical_meas = self.synchronizer.to_canonical_measurement(
            sync_result=sync_res,
            mode=mode,
            rhythm_class=clf_res.rhythm_class,
            rhythm_confidence=round(float(clf_res.rhythm_confidence), 4),
            trend=TrendLabelEnum.STABLE,
            timestamp=now_iso,
        )

        # 6. Session Recording
        target_service = session_service or self.session_service
        if request.session_id and target_service:
            target_service.add_measurement(request.session_id, canonical_meas)

        return MultimodalAnalyzeResponse(
            ecg_heart_rate_bpm=sync_res.ecg_heart_rate_bpm,
            ppg_pulse_rate_bpm=sync_res.ppg_pulse_rate_bpm,
            rate_discrepancy_bpm=sync_res.rate_discrepancy_bpm,
            relative_discrepancy_pct=sync_res.relative_discrepancy_pct,
            agreement_level=sync_res.agreement_level,
            pulse_arrival_time_ms=sync_res.pulse_arrival_time_ms,
            pat_variability_ms=sync_res.pat_variability_ms,
            ecg_beat_count=sync_res.ecg_beat_count,
            ppg_beat_count=sync_res.ppg_beat_count,
            pulse_deficit_detected=sync_res.pulse_deficit_detected,
            ecg_signal_quality=sync_res.ecg_signal_quality,
            ppg_signal_quality=sync_res.ppg_signal_quality,
            fused_signal_quality=sync_res.fused_signal_quality,
            fused_quality_label=QualityLabelEnum(sync_res.fused_quality_label),
            rhythm_class=clf_res.rhythm_class,
            rhythm_confidence=round(float(clf_res.rhythm_confidence), 4),
            trend=TrendLabelEnum.STABLE,
            recommended_primary_modality=sync_res.recommended_primary_modality,
            canonical_measurement=canonical_meas,
            model_version=MODEL_VERSION,
            disclaimer=sync_res.disclaimer,
        )

    def get_synchronized_demo_frame(
        self,
        start_sec: float = 0.0,
        duration_sec: float = 5.0,
        record_id: str = "100",
    ) -> Dict[str, Any]:
        """Generate synchronized paired ECG and optical PPG playback frame for interactive viewers."""
        # Load real ECG slice from MIT-BIH dataset
        sig, fs, lead = self.dataset_loader.load_record(
            record_id, start_sec=start_sec, duration_sec=duration_sec
        )
        ecg_proc = ECGProcessor(sampling_rate_hz=fs)
        ecg_res = ecg_proc.process(sig)

        hr_target = ecg_res.heart_rate_bpm or 75.0

        # Synthesize matching optical PPG pulse train with physiological PAT delay ~220ms
        r_peaks_sec = ecg_res.r_peaks / float(fs) if len(ecg_res.r_peaks) > 0 else None
        _, ppg_raw = generate_synthetic_ppg(
            duration_sec=duration_sec,
            sampling_rate_hz=30.0,
            pulse_rate_bpm=hr_target,
            noise_amplitude=0.02,
            r_peaks_sec=r_peaks_sec,
            pat_delay_sec=0.22,
        )
        ppg_proc = PPGProcessor(sampling_rate_hz=30.0)
        ppg_res = ppg_proc.process(ppg_raw)

        # Run synchronization
        sync_res = self.synchronizer.synchronize(
            ecg_result=ecg_res,
            ppg_result=ppg_res,
            ecg_sampling_rate_hz=fs,
            ppg_sampling_rate_hz=30.0,
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": ModeEnum.OFFLINE_DEMO.value,
            "record_id": record_id,
            "start_sec": start_sec,
            "duration_sec": duration_sec,
            "ecg": {
                "sampling_rate_hz": fs,
                "lead_name": lead,
                "raw_waveform": [round(float(v), 4) for v in sig[: int(fs * duration_sec)]],
                "filtered_waveform": [round(float(v), 4) for v in ecg_res.filtered_signal],
                "r_peaks": ecg_res.r_peaks.tolist(),
                "heart_rate_bpm": ecg_res.heart_rate_bpm,
                "signal_quality": ecg_res.signal_quality,
                "quality_label": ecg_res.quality_label,
            },
            "ppg": {
                "sampling_rate_hz": 30.0,
                "raw_waveform": [round(float(v), 4) for v in ppg_raw],
                "filtered_waveform": [round(float(v), 4) for v in ppg_res.filtered_signal],
                "peaks": [int(p) for p in ppg_res.peaks],
                "pulse_rate_bpm": ppg_res.pulse_rate_bpm,
                "signal_quality": ppg_res.signal_quality,
                "quality_label": ppg_res.quality_label,
            },
            "fusion": sync_res.to_dict(),
        }
