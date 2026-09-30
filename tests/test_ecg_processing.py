"""Automated unit and validation tests for ECG signal processing."""

import numpy as np
import pytest

from ml.ecg.processor import ECGProcessor, ECGAnalysisResult
from backend.services.ecg_service import ECGService
from backend.models.schemas import ECGProcessRequest, ECGProcessResponse, QualityLabelEnum
from scripts.demo_ecg_processing import generate_synthetic_ecg


def test_ecg_processor_initialization():
    """Verify processor parameters and RR limits."""
    proc = ECGProcessor(sampling_rate_hz=360.0, min_bpm=40.0, max_bpm=200.0)
    assert proc.sampling_rate_hz == 360.0
    assert abs(proc.min_rr_sec - (60.0 / 200.0)) < 1e-4  # 0.30s
    assert abs(proc.max_rr_sec - (60.0 / 40.0)) < 1e-4   # 1.50s


def test_ecg_filtering_baseline_wander():
    """Verify that zero-phase Butterworth filter removes large low-frequency baseline drift."""
    proc = ECGProcessor(sampling_rate_hz=360.0)
    fs = 360.0
    t = np.linspace(0, 5, int(5 * fs), endpoint=False)
    # Heavy low-frequency drift (0.2 Hz, 2.0 mV amplitude) + DC offset
    drift = 2.0 * np.sin(2 * np.pi * 0.2 * t) + 1.5
    filtered = proc.filter_signal(drift)
    # After 0.5 Hz highpass, the 0.2 Hz drift amplitude should be attenuated by > 80%
    assert np.max(np.abs(filtered)) < 0.5
    assert abs(np.mean(filtered)) < 0.05


def test_r_peak_detection_and_hr_estimation_60_bpm():
    """Verify accurate detection of 60 BPM ECG (1000 ms interval)."""
    fs = 360.0
    _, raw = generate_synthetic_ecg(
        duration_sec=8.0,
        sampling_rate_hz=fs,
        heart_rate_bpm=60.0,
        noise_amplitude=0.02,
        baseline_drift_amplitude=0.1
    )
    proc = ECGProcessor(sampling_rate_hz=fs)
    result = proc.process(raw)

    assert result.heart_rate_bpm is not None
    # Tolerance: within 1.0 BPM of target 60 BPM
    assert abs(result.heart_rate_bpm - 60.0) <= 1.0
    assert result.signal_quality >= 0.70
    assert result.quality_label == "GOOD"
    # Consecutive RR intervals should be ~1000 ms
    for rr in result.rr_intervals_ms:
        assert abs(rr - 1000.0) < 50.0  # within 50 ms


def test_r_peak_detection_and_hr_estimation_90_bpm():
    """Verify accurate detection of 90 BPM ECG (666.7 ms interval)."""
    fs = 360.0
    _, raw = generate_synthetic_ecg(
        duration_sec=8.0,
        sampling_rate_hz=fs,
        heart_rate_bpm=90.0,
        noise_amplitude=0.02,
        baseline_drift_amplitude=0.1
    )
    proc = ECGProcessor(sampling_rate_hz=fs)
    result = proc.process(raw)

    assert result.heart_rate_bpm is not None
    assert abs(result.heart_rate_bpm - 90.0) <= 1.5
    assert result.signal_quality >= 0.70
    assert result.quality_label == "GOOD"


def test_impossible_rr_intervals_rejected():
    """Verify that impossible intervals (too short or too long) are rejected."""
    proc = ECGProcessor(sampling_rate_hz=360.0)
    # Peak list with an impossible 100ms interval (sample distance 36)
    # and a massive pause (3000ms, sample distance 1080)
    r_peaks = np.array([360, 396, 756, 1836, 2196])  # 36 samples = 100ms, 1080 samples = 3000ms
    all_rr, valid_rr = proc.extract_rr_intervals(r_peaks)

    # 100ms and 3000ms must be absent from valid_rr
    for rr in valid_rr:
        assert rr >= (proc.min_rr_sec * 1000.0)
        assert rr <= (proc.max_rr_sec * 1000.0)


def test_pure_noise_signal_quality_poor():
    """Verify that pure Gaussian noise results in low SQI and POOR quality."""
    proc = ECGProcessor(sampling_rate_hz=360.0)
    np.random.seed(99)
    noise = np.random.normal(0, 1.0, 360 * 5)
    result = proc.process(noise)

    # Pure noise should not produce confident GOOD ECG classification
    assert result.signal_quality < 0.60
    assert result.quality_label in ["POOR", "FAIR"]


def test_short_signal_safety():
    """Verify that processing very short or empty signals handles safely without crashing."""
    proc = ECGProcessor(sampling_rate_hz=360.0)
    res_empty = proc.process([])
    assert res_empty.heart_rate_bpm is None
    assert res_empty.quality_label == "POOR"
    assert len(res_empty.r_peaks) == 0

    res_short = proc.process([0.1, 0.2, 0.3])
    assert res_short.heart_rate_bpm is None
    assert res_short.quality_label == "POOR"


def test_ecg_service_canonical_contract():
    """Verify ECGService returns structured results conforming to canonical schema."""
    service = ECGService(sampling_rate_hz=360.0)
    _, raw = generate_synthetic_ecg(
        duration_sec=6.0,
        sampling_rate_hz=360.0,
        heart_rate_bpm=72.0,
        noise_amplitude=0.02,
        baseline_drift_amplitude=0.1
    )

    req = ECGProcessRequest(
        sampling_rate_hz=360.0,
        signal=raw.tolist(),
        record_id="test_record_01"
    )

    resp: ECGProcessResponse = service.process_window(req)
    assert isinstance(resp, ECGProcessResponse)
    assert resp.quality_label == QualityLabelEnum.GOOD
    assert resp.signal_quality >= 0.70
    assert resp.heart_rate_bpm is not None
    assert abs(resp.heart_rate_bpm - 72.0) <= 1.5
    assert len(resp.r_peaks) >= 5
    assert len(resp.rr_intervals_ms) >= 4
    assert len(resp.filtered_signal) == len(raw)
