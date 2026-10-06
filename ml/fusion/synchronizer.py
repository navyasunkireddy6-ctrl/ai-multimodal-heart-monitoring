"""Multimodal Cardiac Synchronization, Dual-Rate Fusion, and Transit Time Analysis.

Implements physiological alignment between ECG electrical activity and optical PPG peripheral
pulsatile flow:
1. Dual-rate comparative analysis: ECG Heart Rate (HR) vs PPG Pulse Rate (PR).
2. Pulse Arrival Time (PAT) / Pulse Transit Time (PTT) beat-by-beat latency estimation.
3. Pulse Deficit detection: identifying hemodynamically ineffective contractions or arrhythmias.
4. Multimodal reliability fusion and failover logic based on ECG & optical SQI.
5. Canonical measurement transformation adhering strictly to docs/API_CONTRACT.md.

Disclaimer:
    Research prototype — not intended for medical diagnosis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from backend.models.schemas import (
    CanonicalMeasurement,
    ModeEnum,
    QualityLabelEnum,
    TrendLabelEnum,
)
from ml.ecg.processor import ECGAnalysisResult
from ml.ppg.processor import PPGAnalysisResult

MODEL_VERSION = "multimodal-fusion-v1.0.0"
RESEARCH_DISCLAIMER = "Research prototype — not intended for medical diagnosis."


@dataclass
class MultimodalSyncResult:
    """Consolidated outcome of synchronized multimodal ECG and PPG evaluation."""

    ecg_heart_rate_bpm: Optional[float]
    ppg_pulse_rate_bpm: Optional[float]
    rate_discrepancy_bpm: Optional[float]
    relative_discrepancy_pct: Optional[float]
    agreement_level: str  # EXCELLENT, ACCEPTABLE, DISCREPANT, UNAVAILABLE
    pulse_arrival_time_ms: Optional[float]
    pat_variability_ms: Optional[float]
    ecg_beat_count: int
    ppg_beat_count: int
    pulse_deficit_detected: bool
    ecg_signal_quality: float
    ppg_signal_quality: float
    fused_signal_quality: float
    fused_quality_label: str  # GOOD, FAIR, POOR
    recommended_primary_modality: str  # BOTH, ECG, PPG
    model_version: str = MODEL_VERSION
    disclaimer: str = RESEARCH_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ecg_heart_rate_bpm": self.ecg_heart_rate_bpm,
            "ppg_pulse_rate_bpm": self.ppg_pulse_rate_bpm,
            "rate_discrepancy_bpm": self.rate_discrepancy_bpm,
            "relative_discrepancy_pct": self.relative_discrepancy_pct,
            "agreement_level": self.agreement_level,
            "pulse_arrival_time_ms": self.pulse_arrival_time_ms,
            "pat_variability_ms": self.pat_variability_ms,
            "ecg_beat_count": self.ecg_beat_count,
            "ppg_beat_count": self.ppg_beat_count,
            "pulse_deficit_detected": self.pulse_deficit_detected,
            "ecg_signal_quality": self.ecg_signal_quality,
            "ppg_signal_quality": self.ppg_signal_quality,
            "fused_signal_quality": self.fused_signal_quality,
            "fused_quality_label": self.fused_quality_label,
            "recommended_primary_modality": self.recommended_primary_modality,
            "model_version": self.model_version,
            "disclaimer": self.disclaimer,
        }


class MultimodalSynchronizer:
    """Engine performing multimodal alignment, transit time estimation, and fusion."""

    def __init__(
        self,
        excellent_threshold_bpm: float = 3.0,
        acceptable_threshold_bpm: float = 8.0,
        min_pat_ms: float = 80.0,
        max_pat_ms: float = 450.0,
        deficit_beat_threshold: int = 2,
    ):
        self.excellent_threshold_bpm = float(excellent_threshold_bpm)
        self.acceptable_threshold_bpm = float(acceptable_threshold_bpm)
        self.min_pat_ms = float(min_pat_ms)
        self.max_pat_ms = float(max_pat_ms)
        self.deficit_beat_threshold = int(deficit_beat_threshold)

    def compute_rate_agreement(
        self, hr: Optional[float], pr: Optional[float]
    ) -> Tuple[Optional[float], Optional[float], str]:
        """Compute absolute discrepancy, relative percentage error, and qualitative agreement."""
        if hr is None or pr is None:
            return None, None, "UNAVAILABLE"

        delta = round(abs(hr - pr), 2)
        rel_pct = round((delta / hr) * 100.0, 2) if hr > 0 else 0.0

        if delta <= self.excellent_threshold_bpm:
            agreement = "EXCELLENT"
        elif delta <= self.acceptable_threshold_bpm:
            agreement = "ACCEPTABLE"
        else:
            agreement = "DISCREPANT"

        return delta, rel_pct, agreement

    def compute_pulse_arrival_time(
        self,
        ecg_r_peaks_sec: np.ndarray,
        ppg_peaks_sec: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate Pulse Arrival Time (PAT in ms) for consecutive R-peak and systolic peak pairs."""
        if len(ecg_r_peaks_sec) == 0 or len(ppg_peaks_sec) == 0:
            return None, None

        pat_list: List[float] = []

        for r_t in ecg_r_peaks_sec:
            # Find the first systolic peak following the R-peak
            following = ppg_peaks_sec[ppg_peaks_sec > r_t]
            if len(following) == 0:
                continue

            first_peak = following[0]
            delay_ms = (first_peak - r_t) * 1000.0

            # Physiological validation filter: transit time typically 100 - 450 ms
            if self.min_pat_ms <= delay_ms <= self.max_pat_ms:
                pat_list.append(delay_ms)

        if not pat_list:
            return None, None

        mean_pat = round(float(np.mean(pat_list)), 1)
        std_pat = round(float(np.std(pat_list)), 1) if len(pat_list) > 1 else 0.0
        return mean_pat, std_pat

    def fuse_quality(
        self,
        ecg_sqi: float,
        ppg_sqi: float,
        ecg_weight: float = 0.6,
    ) -> Tuple[float, str, str]:
        """Calculate weighted fused SQI and determine primary recommended modality."""
        ecg_sqi = float(np.clip(ecg_sqi, 0.0, 1.0))
        ppg_sqi = float(np.clip(ppg_sqi, 0.0, 1.0))

        # Dynamically bias weight if one sensor has significantly degraded contact
        if ecg_sqi < 0.50 and ppg_sqi >= 0.70:
            fused_sqi = 0.25 * ecg_sqi + 0.75 * ppg_sqi
            recommended = "PPG"
        elif ppg_sqi < 0.50 and ecg_sqi >= 0.70:
            fused_sqi = 0.75 * ecg_sqi + 0.25 * ppg_sqi
            recommended = "ECG"
        else:
            w_ecg = ecg_weight
            w_ppg = 1.0 - ecg_weight
            fused_sqi = (w_ecg * ecg_sqi) + (w_ppg * ppg_sqi)
            recommended = "BOTH" if (ecg_sqi >= 0.60 and ppg_sqi >= 0.60) else (
                "ECG" if ecg_sqi >= ppg_sqi else "PPG"
            )

        fused_sqi = round(float(np.clip(fused_sqi, 0.0, 1.0)), 3)

        if fused_sqi >= 0.80:
            label = "GOOD"
        elif fused_sqi >= 0.55:
            label = "FAIR"
        else:
            label = "POOR"

        return fused_sqi, label, recommended

    def synchronize(
        self,
        ecg_result: ECGAnalysisResult,
        ppg_result: PPGAnalysisResult,
        ecg_sampling_rate_hz: float = 360.0,
        ppg_sampling_rate_hz: float = 30.0,
    ) -> MultimodalSyncResult:
        """Perform comprehensive multimodal synchronization on processed ECG and PPG results."""
        # Convert peak indices to timestamps in seconds
        r_peaks_sec = ecg_result.r_peaks / float(ecg_sampling_rate_hz) if len(ecg_result.r_peaks) > 0 else np.array([])
        ppg_peaks_sec = ppg_result.peaks / float(ppg_sampling_rate_hz) if len(ppg_result.peaks) > 0 else np.array([])

        ecg_beat_count = int(len(r_peaks_sec))
        ppg_beat_count = int(len(ppg_peaks_sec))

        # Check for pulse deficit: ventricular contractions failing to generate palpable pulse
        pulse_deficit = bool(
            (ecg_beat_count - ppg_beat_count) >= self.deficit_beat_threshold
            and ecg_result.quality_label != "POOR"
            and ppg_result.quality_label != "POOR"
        )

        # Rate agreement
        delta, rel_pct, agreement = self.compute_rate_agreement(
            ecg_result.heart_rate_bpm, ppg_result.pulse_rate_bpm
        )

        # Pulse Arrival Time (PAT)
        mean_pat, std_pat = self.compute_pulse_arrival_time(r_peaks_sec, ppg_peaks_sec)

        # Fused SQI
        fused_sqi, fused_label, recommended = self.fuse_quality(
            ecg_result.signal_quality, ppg_result.signal_quality
        )

        return MultimodalSyncResult(
            ecg_heart_rate_bpm=ecg_result.heart_rate_bpm,
            ppg_pulse_rate_bpm=ppg_result.pulse_rate_bpm,
            rate_discrepancy_bpm=delta,
            relative_discrepancy_pct=rel_pct,
            agreement_level=agreement,
            pulse_arrival_time_ms=mean_pat,
            pat_variability_ms=std_pat,
            ecg_beat_count=ecg_beat_count,
            ppg_beat_count=ppg_beat_count,
            pulse_deficit_detected=pulse_deficit,
            ecg_signal_quality=round(float(ecg_result.signal_quality), 3),
            ppg_signal_quality=round(float(ppg_result.signal_quality), 3),
            fused_signal_quality=fused_sqi,
            fused_quality_label=fused_label,
            recommended_primary_modality=recommended,
            model_version=MODEL_VERSION,
            disclaimer=RESEARCH_DISCLAIMER,
        )

    def to_canonical_measurement(
        self,
        sync_result: MultimodalSyncResult,
        mode: ModeEnum = ModeEnum.OFFLINE_DEMO,
        rhythm_class: Optional[str] = None,
        rhythm_confidence: Optional[float] = None,
        trend: Optional[TrendLabelEnum] = None,
        timestamp: Optional[str] = None,
    ) -> CanonicalMeasurement:
        """Construct canonical measurement object conforming to docs/API_CONTRACT.md."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        # PPG rate reported as null if POOR quality per canonical contract
        pr_val = (
            sync_result.ppg_pulse_rate_bpm
            if (sync_result.fused_quality_label != "POOR" and sync_result.ppg_signal_quality >= 0.50)
            else None
        )

        return CanonicalMeasurement(
            timestamp=ts,
            mode=mode,
            heart_rate_bpm=sync_result.ecg_heart_rate_bpm,
            pulse_rate_bpm=pr_val,
            signal_quality=sync_result.fused_signal_quality,
            quality_label=QualityLabelEnum(sync_result.fused_quality_label),
            rhythm_class=rhythm_class,
            rhythm_confidence=rhythm_confidence,
            trend=trend,
            model_version=sync_result.model_version,
        )
