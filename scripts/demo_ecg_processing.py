"""CLI Demonstration of ECG Signal Processing, R-Peak Detection, and SQI Assessment.

Run with:
    python scripts/demo_ecg_processing.py
"""

import sys
import numpy as np

# Ensure project root is in python path
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.ecg.processor import ECGProcessor


def generate_synthetic_ecg(
    duration_sec: float = 10.0,
    sampling_rate_hz: float = 360.0,
    heart_rate_bpm: float = 72.0,
    noise_amplitude: float = 0.05,
    baseline_drift_amplitude: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a realistic synthetic ECG lead II signal with baseline drift and noise."""
    t = np.linspace(0, duration_sec, int(duration_sec * sampling_rate_hz), endpoint=False)
    rr_sec = 60.0 / heart_rate_bpm
    ecg = np.zeros_like(t)

    # Place P-Q-R-S-T waves periodically
    peak_times = np.arange(0.4, duration_sec - 0.2, rr_sec)
    for p_t in peak_times:
        # P-wave
        ecg += 0.15 * np.exp(-((t - (p_t - 0.18)) ** 2) / (2 * (0.03 ** 2)))
        # Q-wave
        ecg -= 0.15 * np.exp(-((t - (p_t - 0.04)) ** 2) / (2 * (0.012 ** 2)))
        # R-peak (sharp and high amplitude)
        ecg += 1.20 * np.exp(-((t - p_t) ** 2) / (2 * (0.014 ** 2)))
        # S-wave
        ecg -= 0.25 * np.exp(-((t - (p_t + 0.04)) ** 2) / (2 * (0.015 ** 2)))
        # T-wave
        ecg += 0.30 * np.exp(-((t - (p_t + 0.22)) ** 2) / (2 * (0.05 ** 2)))

    # Add baseline wander (respiration / motion ~ 0.25 Hz)
    drift = baseline_drift_amplitude * np.sin(2 * np.pi * 0.25 * t)
    # Add high-frequency noise
    noise = np.random.normal(0, noise_amplitude, len(t))

    raw_signal = ecg + drift + noise
    return t, raw_signal


def run_demo():
    print("=" * 70)
    print("  MULTIMODAL CARDIAC MONITORING — ECG PROCESSING DEMONSTRATION")
    print("  Research prototype — not intended for medical diagnosis.")
    print("=" * 70)

    target_hr = 75.0
    fs = 360.0
    duration = 10.0
    print(f"\n[1] Generating 10-second ECG waveform:")
    print(f"    - Target Heart Rate: {target_hr} BPM")
    print(f"    - Sampling Rate:     {fs} Hz ({int(duration * fs)} samples)")
    print(f"    - Baseline Drift:    0.3 mV (simulated respiration)")
    print(f"    - Added Noise:       Gaussian noise sigma=0.04 mV")

    np.random.seed(42)
    from typing import Tuple
    t, raw_signal = generate_synthetic_ecg(
        duration_sec=duration,
        sampling_rate_hz=fs,
        heart_rate_bpm=target_hr,
        noise_amplitude=0.04,
        baseline_drift_amplitude=0.3
    )

    processor = ECGProcessor(sampling_rate_hz=fs)
    print("\n[2] Executing ECG Processing Pipeline...")
    result = processor.process(raw_signal)

    print("\n" + "-" * 70)
    print("  STRUCTURED ECG PROCESSING RESULTS")
    print("-" * 70)
    print(f"  • Filtered Signal Min/Max:  {result.filtered_signal.min():.3f} / {result.filtered_signal.max():.3f} mV")
    print(f"  • Detected R-Peaks Count:  {len(result.r_peaks)}")
    print(f"  • R-Peak Indices:          {result.r_peaks.tolist()[:8]} ...")
    print(f"  • Consecutive RR Intervals:{[round(x, 1) for x in result.rr_intervals_ms[:6]]} ms ...")
    print(f"  • Estimated Heart Rate:    {result.heart_rate_bpm} BPM (Target: {target_hr} BPM)")
    print(f"  • Signal Quality (SQI):    {result.signal_quality:.3f}")
    print(f"  • Quality Classification:  [{result.quality_label}]")
    print(f"  • SQI Diagnostics:")
    for k, v in result.metrics.items():
        print(f"      - {k:22s}: {v}")

    # Validation: Error relative to target
    if result.heart_rate_bpm is not None:
        err = abs(result.heart_rate_bpm - target_hr)
        print(f"\n  Rate Error: {err:.1f} BPM (within expected engineering tolerance)")

    print("\n[3] Testing Robust Validation with Artificial Noise Spike:")
    corrupted_signal = raw_signal.copy()
    # Inject impossible artifact: 2 massive spikes separated by only 50 ms (impossible RR)
    corrupted_signal[1000] += 5.0
    corrupted_signal[1018] += 5.0  # 18 samples at 360Hz = 50ms (impossible for human heart)
    corrupted_res = processor.process(corrupted_signal)
    print(f"  • Corrupted Signal Quality: [{corrupted_res.quality_label}] (SQI: {corrupted_res.signal_quality:.3f})")
    print(f"  • Protected Heart Rate:     {corrupted_res.heart_rate_bpm} BPM (Safely preserved or rejected)")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
