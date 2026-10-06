"""Synthetic Photoplethysmogram (PPG) Waveform Generation.

Generates realistic human optical arterial pulse waveforms with systolic upstroke,
dicrotic notch, diastolic peak, respiratory baseline modulation, and sensor noise.
Optionally supports ECG phase-locking via r_peaks_sec for multimodal hemodynamic synchronization.
"""

from typing import Tuple, Optional
import numpy as np


def generate_synthetic_ppg(
    duration_sec: float = 10.0,
    sampling_rate_hz: float = 30.0,
    pulse_rate_bpm: float = 72.0,
    noise_amplitude: float = 0.02,
    motion_amplitude: float = 0.0,
    dc_offset: float = 128.0,
    ac_amplitude: float = 12.0,
    seed: Optional[int] = None,
    r_peaks_sec: Optional[np.ndarray] = None,
    pat_delay_sec: float = 0.22,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a realistic optical fingertip PPG signal.

    Args:
        duration_sec: Signal duration in seconds.
        sampling_rate_hz: Optical camera sampling rate in Hz (typically 30 FPS).
        pulse_rate_bpm: Target cardiovascular pulse rate in beats per minute.
        noise_amplitude: Normalized Gaussian noise level.
        motion_amplitude: Magnitude of motion artifacts and optical baseline disruption.
        dc_offset: Mean baseline optical intensity (e.g., camera red channel ~128).
        ac_amplitude: Pulsatile AC component amplitude.
        seed: Random seed for reproducible waveform synthesis.
        r_peaks_sec: Optional array of ECG R-peak timestamps in seconds for phase-locked multimodal synthesis.
        pat_delay_sec: Physiological Pulse Arrival Time latency in seconds (default 0.22s = 220ms).

    Returns:
        t: Array of time sample timestamps in seconds.
        signal: Optical intensity series values.
    """
    n_samples = int(duration_sec * sampling_rate_hz)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    ibi_sec = 60.0 / pulse_rate_bpm

    normalized_pulse = np.zeros_like(t)

    # Place individual pulse beats across time window
    if r_peaks_sec is not None and len(r_peaks_sec) > 0:
        beat_times = np.array(r_peaks_sec) + pat_delay_sec
        if len(r_peaks_sec) > 1:
            ibi_sec = float(np.mean(np.diff(r_peaks_sec)))
    else:
        beat_times = np.arange(0.2, duration_sec + ibi_sec, ibi_sec)

    for b_t in beat_times:
        # 1. Primary systolic wave (rapid ventricular ejection)
        s_width = 0.075 * (ibi_sec / 0.833)
        normalized_pulse += 1.0 * np.exp(-((t - b_t) ** 2) / (2 * (s_width ** 2)))

        # 2. Secondary diastolic wave (aortic valve closure / dicrotic reflection)
        d_delay = 0.28 * (ibi_sec / 0.833)
        d_width = 0.10 * (ibi_sec / 0.833)
        normalized_pulse += 0.35 * np.exp(-((t - (b_t + d_delay)) ** 2) / (2 * (d_width ** 2)))

    # 3. Respiratory baseline wander (~0.22 Hz breathing modulation)
    respiration_drift = 0.15 * np.sin(2 * np.pi * 0.22 * t)

    # 4. Sensor and photon noise
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_amplitude, n_samples)

    # 5. Optional motion artifacts (sudden optical baseline disruption and tremor)
    motion = np.zeros_like(t)
    if motion_amplitude > 0.0:
        artifact_center = duration_sec * 0.5
        envelope = np.exp(-((t - artifact_center) ** 2) / 2.0)
        # Baseline surge (~0.25 Hz) + optical fluctuation (~5.5 Hz)
        motion = motion_amplitude * (
            1.2 * np.sin(2 * np.pi * 0.25 * t) +
            0.8 * np.sin(2 * np.pi * 5.5 * t)
        ) * envelope

    # Scale to typical optical red-channel intensity
    pulse_signal = dc_offset + ac_amplitude * (normalized_pulse + respiration_drift + noise + motion)
    return t, pulse_signal
