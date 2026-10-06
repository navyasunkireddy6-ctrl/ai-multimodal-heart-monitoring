"""Smartphone and Optical Photoplethysmogram (PPG) Signal Processing.

Provides filtering, systolic peak detection, pulse rate estimation,
and optical Signal Quality Index (SQI) assessment aligned with the canonical API contract.

Disclaimer:
    Research prototype — not intended for medical diagnosis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks

from backend.models.schemas import QualityLabelEnum


@dataclass
class PPGAnalysisResult:
    """Structured output from optical PPG waveform analysis."""
    sampling_rate_hz: float
    raw_signal: np.ndarray
    filtered_signal: np.ndarray
    peaks: np.ndarray
    ibi_intervals_ms: np.ndarray
    pulse_rate_bpm: Optional[float]
    signal_quality: float
    quality_label: str
    snr_db: float
    is_usable: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "pulse_rate_bpm": round(float(self.pulse_rate_bpm), 1) if self.pulse_rate_bpm is not None else None,
            "signal_quality": round(float(self.signal_quality), 4),
            "quality_label": self.quality_label,
            "peaks": [int(p) for p in self.peaks],
            "ibi_intervals_ms": [round(float(ibi), 1) for ibi in self.ibi_intervals_ms],
            "filtered_signal": [round(float(v), 4) for v in self.filtered_signal],
            "snr_db": round(float(self.snr_db), 2),
            "is_usable": self.is_usable,
            "message": self.message,
        }


class PPGProcessor:
    """Processor for smartphone camera and optical PPG pulse signals."""

    def __init__(
        self,
        sampling_rate_hz: float = 30.0,
        lowcut_hz: float = 0.5,
        highcut_hz: float = 4.0,
        filter_order: int = 2,
        min_bpm: float = 40.0,
        max_bpm: float = 200.0,
    ):
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.lowcut_hz = float(lowcut_hz)
        self.highcut_hz = float(highcut_hz)
        self.filter_order = int(filter_order)
        self.min_bpm = float(min_bpm)
        self.max_bpm = float(max_bpm)

        # Physiological bounds on Inter-Beat Intervals (IBI)
        self.min_ibi_sec = 60.0 / self.max_bpm   # 0.30s at 200 BPM
        self.max_ibi_sec = 60.0 / self.min_bpm   # 1.50s at 40 BPM

    def filter_signal(self, signal: List[float] | np.ndarray) -> np.ndarray:
        """Apply zero-phase Butterworth bandpass filter to isolate arterial pulse pulsations."""
        arr = np.asarray(signal, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if len(arr) < 18:
            return arr - np.mean(arr) if len(arr) > 0 else arr

        nyquist = 0.5 * self.sampling_rate_hz
        low = max(0.1, self.lowcut_hz) / nyquist
        high = min(self.highcut_hz, nyquist - 0.5) / nyquist

        if low >= high or low <= 0 or high >= 1:
            return arr - np.mean(arr)

        try:
            sos = butter(self.filter_order, [low, high], btype="bandpass", output="sos")
            filtered = sosfiltfilt(sos, arr)
            return filtered
        except Exception:
            return arr - np.mean(arr)

    def detect_peaks(self, filtered_signal: np.ndarray) -> np.ndarray:
        """Detect systolic pulse peaks using adaptive thresholding and refractory constraints."""
        signal = np.asarray(filtered_signal, dtype=np.float64)
        n = len(signal)
        min_distance = max(1, int(self.sampling_rate_hz * self.min_ibi_sec))

        if n < min_distance * 2:
            return np.array([], dtype=int)

        sig_std = float(np.std(signal))
        sig_range = float(np.percentile(signal, 95) - np.percentile(signal, 5))
        if sig_std < 1e-6 or sig_range < 1e-5:
            return np.array([], dtype=int)

        # Prominence threshold isolating true systolic pulse upstroke from dicrotic/diastolic notch
        prominence = max(0.01, 0.35 * sig_range)

        peaks, _ = find_peaks(
            signal,
            distance=min_distance,
            prominence=prominence,
            height=np.median(signal),
        )
        return peaks

    def extract_ibi_intervals(self, peaks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract consecutive Inter-Beat Intervals (IBIs) in milliseconds.
        
        Returns:
            all_ibi_ms: All peak-to-peak differences in milliseconds.
            valid_ibi_ms: Physiological IBIs within [min_ibi, max_ibi] bounds.
        """
        if len(peaks) < 2:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        diffs_samples = np.diff(peaks)
        all_ibi_ms = (diffs_samples / self.sampling_rate_hz) * 1000.0

        min_ms = self.min_ibi_sec * 1000.0
        max_ms = self.max_ibi_sec * 1000.0
        valid_mask = (all_ibi_ms >= min_ms) & (all_ibi_ms <= max_ms)
        valid_ibi_ms = all_ibi_ms[valid_mask]

        # Additional outlier filter: exclude values >40% away from median
        if len(valid_ibi_ms) >= 3:
            med = float(np.median(valid_ibi_ms))
            ratio_mask = (valid_ibi_ms >= med * 0.60) & (valid_ibi_ms <= med * 1.40)
            valid_ibi_ms = valid_ibi_ms[ratio_mask]

        return all_ibi_ms, valid_ibi_ms

    def calculate_pulse_rate(self, valid_ibi_ms: np.ndarray) -> Optional[float]:
        """Compute estimated pulse rate in beats per minute (BPM) from valid IBIs."""
        if len(valid_ibi_ms) == 0:
            return None

        # Robust estimate using median IBI
        med_ibi = float(np.median(valid_ibi_ms))
        if med_ibi <= 0:
            return None

        pr = 60000.0 / med_ibi
        return round(float(np.clip(pr, self.min_bpm, self.max_bpm)), 1)

    def assess_signal_quality(
        self,
        raw_signal: np.ndarray,
        filtered_signal: np.ndarray,
        peaks: np.ndarray,
        valid_ibi_ms: np.ndarray,
    ) -> Tuple[float, str, float, bool, str]:
        """Evaluate optical Signal Quality Index (SQI), SNR, usability, and user guidance.
        
        Returns:
            sqi: Normalized score between 0.0 and 1.0.
            quality_label: 'GOOD', 'FAIR', or 'POOR'.
            snr_db: Estimated optical pulse SNR in decibels.
            is_usable: Boolean flag indicating if pulse rate estimation is clinically reliable.
            message: Informative guidance for client and user.
        """
        n = len(filtered_signal)
        if n < int(self.sampling_rate_hz * 1.5):
            return 0.20, QualityLabelEnum.POOR.value, -5.0, False, "Signal window too short for pulse analysis."

        sig_std = float(np.std(filtered_signal))
        raw_std = float(np.std(raw_signal))
        if sig_std < 1e-5 or raw_std < 1e-5:
            return 0.05, QualityLabelEnum.POOR.value, -15.0, False, "Flat or inactive optical signal detected."

        # 1. Optical Signal-to-Noise Ratio (SNR) in raw signal: cardiac pulse band (0.7-3.5 Hz) vs noise
        raw_ac = raw_signal - np.mean(raw_signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sampling_rate_hz)
        raw_power = np.abs(np.fft.rfft(raw_ac)) ** 2
        total_raw_p = np.sum(raw_power) + 1e-12

        cardiac_band = (freqs >= 0.7) & (freqs <= 3.5)
        in_band_raw_p = np.sum(raw_power[cardiac_band])
        out_band_raw_p = total_raw_p - in_band_raw_p

        snr_linear = in_band_raw_p / (out_band_raw_p + 1e-12)
        snr_db = float(10.0 * np.log10(max(1e-4, snr_linear)))

        # 2. Spectral peak concentration in filtered signal (sharpness of cardiac pulse)
        filt_power = np.abs(np.fft.rfft(filtered_signal)) ** 2
        total_filt_p = np.sum(filt_power) + 1e-12
        max_bin = int(np.argmax(filt_power))
        bin_start = max(0, max_bin - 2)
        bin_end = min(len(filt_power), max_bin + 3)
        peak_power = np.sum(filt_power[bin_start:bin_end])
        spectral_concentration = float(peak_power / total_filt_p)

        # 3. Beat Interval Regularity (coefficient of variation of valid IBIs)
        if len(valid_ibi_ms) >= 2:
            mean_ibi = float(np.mean(valid_ibi_ms))
            std_ibi = float(np.std(valid_ibi_ms))
            cv_ibi = std_ibi / mean_ibi if mean_ibi > 0 else 1.0
            regularity_score = float(np.clip(1.0 - cv_ibi * 2.5, 0.0, 1.0))
        elif len(peaks) >= 2:
            regularity_score = 0.40
        else:
            regularity_score = 0.0

        # 4. Expected systolic peaks count
        min_expected_peaks = max(1, int((n / self.sampling_rate_hz) * (self.min_bpm / 60.0) * 0.75))
        peak_count_ratio = min(1.0, len(peaks) / min_expected_peaks)

        # 5. Composite SQI
        snr_norm = float(np.clip((snr_db + 5.0) / 20.0, 0.0, 1.0))
        sqi = 0.40 * snr_norm + 0.30 * spectral_concentration + 0.20 * regularity_score + 0.10 * peak_count_ratio
        sqi = float(np.clip(sqi, 0.0, 1.0))

        # Determine label and usability
        if sqi >= 0.70 and snr_db >= 3.0:
            quality_label = QualityLabelEnum.GOOD.value
            is_usable = True
            message = "Signal quality sufficient for reliable pulse rate estimation."
        elif sqi >= 0.40 and snr_db >= -3.0 and len(valid_ibi_ms) >= 2:
            quality_label = QualityLabelEnum.FAIR.value
            is_usable = True
            message = "Moderate optical noise. Maintain steady fingertip contact."
        else:
            quality_label = QualityLabelEnum.POOR.value
            is_usable = False
            message = "Poor optical contact or motion artifact. Keep finger steady on camera."

        return round(sqi, 4), quality_label, round(snr_db, 2), is_usable, message

    def process(self, signal: List[float] | np.ndarray) -> PPGAnalysisResult:
        """End-to-end processing of a photoplethysmogram window.
        
        Filters waveform, locates systolic peaks, measures IBIs,
        determines pulse rate, and evaluates optical SQI.
        """
        raw = np.asarray(signal, dtype=np.float64)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        filtered = self.filter_signal(raw)
        peaks = self.detect_peaks(filtered)
        all_ibi_ms, valid_ibi_ms = self.extract_ibi_intervals(peaks)

        sqi, quality_label, snr_db, is_usable, message = self.assess_signal_quality(
            raw_signal=raw,
            filtered_signal=filtered,
            peaks=peaks,
            valid_ibi_ms=valid_ibi_ms,
        )

        # Per canonical contract: pulse rate must be null if quality is POOR
        if quality_label == QualityLabelEnum.POOR.value or not is_usable:
            pulse_rate_bpm = None
        else:
            pulse_rate_bpm = self.calculate_pulse_rate(valid_ibi_ms)

        return PPGAnalysisResult(
            sampling_rate_hz=self.sampling_rate_hz,
            raw_signal=raw,
            filtered_signal=filtered,
            peaks=peaks,
            ibi_intervals_ms=valid_ibi_ms,
            pulse_rate_bpm=pulse_rate_bpm,
            signal_quality=sqi,
            quality_label=quality_label,
            snr_db=snr_db,
            is_usable=is_usable,
            message=message,
        )
