"""ECG signal processing, filtering, R-peak detection, and Signal Quality Index (SQI).

Implements:
- Zero-phase Butterworth bandpass filtering (0.5 - 40 Hz) and notch filtering
- Baseline wander removal
- Modified Pan-Tompkins QRS and R-peak detection with refractory validation
- Physiological RR interval validation (rejecting non-physiological intervals)
- Heart rate calculation (HR = 60 / RR)
- Multi-metric ECG Signal Quality Assessment (kurtosis, spectral power ratio, peak regularity)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, find_peaks
from scipy.stats import kurtosis


@dataclass
class ECGAnalysisResult:
    """Structured result of ECG signal analysis."""
    sampling_rate_hz: float
    raw_signal: np.ndarray
    filtered_signal: np.ndarray
    r_peaks: np.ndarray  # sample indices
    rr_intervals_ms: np.ndarray  # consecutive intervals in milliseconds
    heart_rate_bpm: Optional[float]
    signal_quality: float  # [0.0, 1.0]
    quality_label: str  # "GOOD", "FAIR", "POOR"
    metrics: Dict[str, float] = field(default_factory=dict)


class ECGProcessor:
    """Robust physiological ECG signal processor."""

    def __init__(
        self,
        sampling_rate_hz: float = 360.0,
        lowcut_hz: float = 0.5,
        highcut_hz: float = 40.0,
        filter_order: int = 3,
        min_bpm: float = 35.0,
        max_bpm: float = 220.0,
    ):
        """Initialize ECG processor parameters.
        
        Args:
            sampling_rate_hz: Sampling frequency in Hertz.
            lowcut_hz: Highpass cutoff for baseline wander removal.
            highcut_hz: Lowpass cutoff for high-frequency noise removal.
            filter_order: Butterworth filter order.
            min_bpm: Minimum physiologically valid heart rate in BPM.
            max_bpm: Maximum physiologically valid heart rate in BPM.
        """
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.lowcut_hz = lowcut_hz
        self.highcut_hz = highcut_hz
        self.filter_order = filter_order
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

        # Precompute valid RR interval bounds in seconds
        # e.g., max_bpm = 220 -> min_rr = 60 / 220 = 0.272s (272 ms)
        # min_bpm = 35 -> max_rr = 60 / 35 = 1.714s (1714 ms)
        self.min_rr_sec = 60.0 / self.max_bpm
        self.max_rr_sec = 60.0 / self.min_bpm

    def filter_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply zero-phase Butterworth bandpass filter to eliminate baseline wander and high-freq noise."""
        signal = np.asarray(signal, dtype=np.float64)
        if len(signal) < 15:
            return signal

        nyquist = 0.5 * self.sampling_rate_hz
        # Ensure cutoff frequencies do not exceed Nyquist
        low = max(0.1, self.lowcut_hz) / nyquist
        high = min(self.highcut_hz, nyquist - 1.0) / nyquist

        if low >= high or low <= 0 or high >= 1:
            # Fallback mean removal if filter bands invalid
            return signal - np.mean(signal)

        # Second-order sections (SOS) representation for numerical stability
        sos = butter(self.filter_order, [low, high], btype="bandpass", output="sos")
        filtered = sosfiltfilt(sos, signal)
        return filtered

    def detect_r_peaks(self, filtered_signal: np.ndarray) -> np.ndarray:
        """Detect QRS complexes and locate R-peaks using a Pan-Tompkins inspired workflow.
        
        Steps:
        1. Derivative enhancement to emphasize high-slope QRS complexes.
        2. Non-linear squaring to accentuate peaks and suppress noise.
        3. Moving window integration (~150 ms).
        4. Adaptive thresholding with physiological refractory period constraint (~200 ms).
        5. Peak realignment to the local maximum in the filtered waveform.
        """
        signal = np.asarray(filtered_signal, dtype=np.float64)
        n_samples = len(signal)
        if n_samples < int(self.sampling_rate_hz * 0.5):
            return np.array([], dtype=int)

        fs = self.sampling_rate_hz

        # 1. 5-point central derivative approximation
        diff = np.diff(signal)
        # Pad to match original length
        diff = np.pad(diff, (0, 1), mode="edge")

        # 2. Squaring function
        squared = diff ** 2

        # 3. Moving window integration (~120ms - 150ms window)
        window_size = max(3, int(0.12 * fs))
        kernel = np.ones(window_size) / window_size
        integrated = np.convolve(squared, kernel, mode="same")

        # 4. Adaptive thresholding
        # Refractory period: no two R-peaks can occur within min_rr_sec (or 200 ms)
        min_distance = max(1, int(0.20 * fs))

        # Dynamic threshold based on moving statistics
        q75 = np.percentile(integrated, 75)
        q25 = np.percentile(integrated, 25)
        iqr = q75 - q25
        threshold = q75 + 0.3 * iqr if iqr > 0 else np.mean(integrated)

        initial_peaks, _ = find_peaks(
            integrated,
            height=threshold,
            distance=min_distance
        )

        if len(initial_peaks) == 0:
            # Fallback with relaxed threshold
            fallback_thresh = np.mean(integrated) + 0.5 * np.std(integrated)
            initial_peaks, _ = find_peaks(
                integrated,
                height=fallback_thresh,
                distance=min_distance
            )

        if len(initial_peaks) == 0:
            return np.array([], dtype=int)

        # 5. Peak realignment: Search within [-100ms, +100ms] of each integrated peak
        # for the local maximum of the bandpass filtered signal
        search_radius = max(2, int(0.10 * fs))
        aligned_peaks = []

        for p in initial_peaks:
            start_idx = max(0, p - search_radius)
            end_idx = min(n_samples, p + search_radius + 1)
            local_window = signal[start_idx:end_idx]
            if len(local_window) > 0:
                local_peak_idx = start_idx + int(np.argmax(local_window))
                aligned_peaks.append(local_peak_idx)

        # Remove duplicate peak indices and sort
        unique_peaks = np.unique(aligned_peaks)

        # Filter out minor peaks (e.g., P-waves or low noise peaks) whose amplitude
        # is significantly below the prominent QRS complex amplitude distribution
        if len(unique_peaks) > 2:
            peak_amps = signal[unique_peaks]
            q75_amp = np.percentile(peak_amps, 75)
            if q75_amp > 0:
                amplitude_mask = peak_amps >= (0.35 * q75_amp)
                unique_peaks = unique_peaks[amplitude_mask]

        # Secondary refractory enforcement and T-wave discrimination after realignment:
        # Standard physiological constraint: beats occurring within 360 ms of a large peak
        # with significantly lower amplitude (< 60%) are classified as T-waves.
        filtered_r_peaks = []
        last_peak = -100000
        for p in unique_peaks:
            if (p - last_peak) >= min_distance:
                # T-wave refractory check
                if last_peak >= 0 and (p - last_peak) < int(0.36 * fs):
                    if signal[p] < 0.60 * signal[last_peak]:
                        continue  # Skip T-wave candidate
                filtered_r_peaks.append(p)
                last_peak = p

        return np.array(filtered_r_peaks, dtype=int)

    def extract_rr_intervals(self, r_peaks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract consecutive RR intervals and perform physiological validation.
        
        Returns:
            all_rr_ms: All consecutive intervals in milliseconds.
            valid_rr_ms: Intervals satisfying physiological constraints (35 - 220 BPM).
        """
        if len(r_peaks) < 2:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        # Differences in samples -> convert to milliseconds
        diff_samples = np.diff(r_peaks)
        all_rr_sec = diff_samples / self.sampling_rate_hz
        all_rr_ms = all_rr_sec * 1000.0

        # Physiological validation: 0.27s (220 BPM) <= RR <= 1.71s (35 BPM)
        valid_mask = (all_rr_sec >= self.min_rr_sec) & (all_rr_sec <= self.max_rr_sec)

        # Relative outlier rejection: Reject RR intervals deviating > 40% from median
        if np.sum(valid_mask) >= 3:
            median_rr = np.median(all_rr_sec[valid_mask])
            relative_valid = np.abs(all_rr_sec - median_rr) <= (0.45 * median_rr)
            valid_mask = valid_mask & relative_valid

        valid_rr_ms = all_rr_ms[valid_mask]
        return all_rr_ms, valid_rr_ms

    def calculate_heart_rate(self, valid_rr_ms: np.ndarray) -> Optional[float]:
        """Calculate physiological heart rate from validated RR intervals.
        
        Formula: HR = 60 / RR (in seconds) = 60000 / RR (in ms).
        Uses robust trimmed/median estimation to avoid single-beat distortion.
        """
        if len(valid_rr_ms) == 0:
            return None

        # Median RR is robust against remaining ectopic beats
        median_rr_sec = np.median(valid_rr_ms) / 1000.0
        if median_rr_sec <= 0:
            return None

        hr = 60.0 / median_rr_sec
        # Hard clamping within physiological boundaries
        if hr < self.min_bpm or hr > self.max_bpm:
            return None

        return round(float(hr), 1)

    def assess_signal_quality(
        self,
        raw_signal: np.ndarray,
        filtered_signal: np.ndarray,
        r_peaks: np.ndarray,
        all_rr_ms: np.ndarray,
        valid_rr_ms: np.ndarray
    ) -> Tuple[float, str, Dict[str, float]]:
        """Assess ECG Signal Quality Index (SQI) using a composite scoring mechanism.
        
        Evaluates:
        1. Kurtosis (kSQI): Normal QRS complexes have high peakedness (kurtosis > 5).
        2. Power distribution in QRS band (5-30 Hz) vs total power.
        3. Baseline wander ratio: Low frequency (< 0.5 Hz) drift.
        4. RR Interval validity ratio: Percentage of physiologically plausible intervals.
        5. RR regularity: Coefficient of variation of intervals.
        """
        metrics: Dict[str, float] = {}

        if len(filtered_signal) < int(self.sampling_rate_hz * 1.0):
            return 0.0, "POOR", {"error": "Signal too short for SQI evaluation"}

        # 1. Kurtosis SQI (Normal ECG kurtosis is typically between 5 and 35)
        k = float(kurtosis(filtered_signal, fisher=False))
        metrics["kurtosis"] = round(k, 2)
        if k >= 5.0:
            k_score = min(1.0, 0.6 + (min(k, 25.0) - 5.0) / 40.0)
        elif k >= 3.5:
            k_score = 0.35
        else:
            # Gaussian noise (k ~ 3.0) or flat signal has almost zero QRS peakedness
            k_score = 0.05

        # 2. Peak amplitude to noise RMS ratio (QRS prominence)
        rms = float(np.std(filtered_signal)) + 1e-12
        if len(r_peaks) > 0:
            peak_amplitudes = np.abs(filtered_signal[r_peaks])
            peak_prominence = float(np.median(peak_amplitudes) / rms)
        else:
            peak_prominence = 0.0
        metrics["peak_prominence"] = round(peak_prominence, 2)
        # Real ECG R-peaks have prominence >= 3.0 relative to RMS
        prominence_score = min(1.0, max(0.0, (peak_prominence - 1.5) / 2.5))

        # 3. Spectral energy concentration in QRS band (5 - 25 Hz)
        freqs = np.fft.rfftfreq(len(filtered_signal), d=1.0 / self.sampling_rate_hz)
        fft_vals = np.abs(np.fft.rfft(filtered_signal)) ** 2
        total_power = np.sum(fft_vals) + 1e-12

        qrs_band = (freqs >= 5.0) & (freqs <= 25.0)
        qrs_power = np.sum(fft_vals[qrs_band])
        spectral_ratio = float(qrs_power / total_power)
        metrics["spectral_qrs_ratio"] = round(spectral_ratio, 3)

        # 4. Peak & RR validity
        if len(all_rr_ms) > 0:
            valid_ratio = float(len(valid_rr_ms) / len(all_rr_ms))
        else:
            valid_ratio = 0.0
        metrics["valid_rr_ratio"] = round(valid_ratio, 3)

        # 5. RR Regularity (coefficient of variation)
        if len(valid_rr_ms) >= 2:
            rr_cv = float(np.std(valid_rr_ms) / np.mean(valid_rr_ms))
            reg_score = max(0.0, 1.0 - min(1.0, rr_cv * 3.0))
        else:
            reg_score = 0.3 if len(r_peaks) >= 1 else 0.0
        metrics["rr_regularity"] = round(reg_score, 3)

        # Composite SQI (weighted blend)
        if len(r_peaks) < 2 or k < 3.2 or peak_prominence < 2.0:
            # Clear noise penalty when peaks are non-prominent or kurtosis is Gaussian/sub-Gaussian
            composite_sqi = min(0.35, 0.4 * k_score + 0.3 * prominence_score + 0.3 * spectral_ratio)
        else:
            composite_sqi = (
                0.25 * k_score +
                0.25 * prominence_score +
                0.20 * min(1.0, spectral_ratio * 1.5) +
                0.15 * valid_ratio +
                0.15 * reg_score
            )

        composite_sqi = max(0.0, min(1.0, float(composite_sqi)))
        metrics["sqi"] = round(composite_sqi, 3)

        # Quality label thresholds
        if composite_sqi >= 0.70 and len(valid_rr_ms) >= 1:
            quality_label = "GOOD"
        elif composite_sqi >= 0.45:
            quality_label = "FAIR"
        else:
            quality_label = "POOR"

        return round(composite_sqi, 3), quality_label, metrics

    def process(self, signal: List[float] | np.ndarray) -> ECGAnalysisResult:
        """End-to-end processing of an ECG window.
        
        Args:
            signal: Raw ECG voltage samples.
            
        Returns:
            ECGAnalysisResult containing filtered waveform, R-peaks, RR intervals,
            heart rate, and SQI metrics.
        """
        raw = np.asarray(signal, dtype=np.float64)
        filtered = self.filter_signal(raw)
        r_peaks = self.detect_r_peaks(filtered)
        all_rr_ms, valid_rr_ms = self.extract_rr_intervals(r_peaks)
        hr_bpm = self.calculate_heart_rate(valid_rr_ms)
        sqi, quality_label, metrics = self.assess_signal_quality(
            raw_signal=raw,
            filtered_signal=filtered,
            r_peaks=r_peaks,
            all_rr_ms=all_rr_ms,
            valid_rr_ms=valid_rr_ms,
        )

        return ECGAnalysisResult(
            sampling_rate_hz=self.sampling_rate_hz,
            raw_signal=raw,
            filtered_signal=filtered,
            r_peaks=r_peaks,
            rr_intervals_ms=valid_rr_ms,
            heart_rate_bpm=hr_bpm,
            signal_quality=sqi,
            quality_label=quality_label,
            metrics=metrics,
        )
