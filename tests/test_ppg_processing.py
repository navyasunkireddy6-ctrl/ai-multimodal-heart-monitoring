"""Automated unit and integration tests for Phase 6 PPG Signal Processing."""

import numpy as np
import pytest

from ml.ppg.processor import PPGProcessor, PPGAnalysisResult
from ml.ppg.synthetic import generate_synthetic_ppg
from backend.services.ppg_service import PPGService
from backend.models.schemas import (
    PPGProcessRequest,
    PPGProcessResponse,
    PPGQualityRequest,
    PPGQualityResponse,
    QualityLabelEnum,
)


def test_ppg_processor_initialization():
    """Verify processor parameters and IBI bounds."""
    proc = PPGProcessor(sampling_rate_hz=30.0, min_bpm=40.0, max_bpm=200.0)
    assert proc.sampling_rate_hz == 30.0
    assert abs(proc.min_ibi_sec - 0.30) < 1e-4
    assert abs(proc.max_ibi_sec - 1.50) < 1e-4


def test_ppg_bandpass_filtering_removes_dc_and_high_freq():
    """Verify bandpass filter eliminates large DC offset and high-frequency noise."""
    fs = 30.0
    proc = PPGProcessor(sampling_rate_hz=fs)
    t = np.linspace(0, 6, int(6 * fs), endpoint=False)

    # 128 DC offset + 1.0 Hz pulse + 12.0 Hz flicker noise
    dc_offset = 135.0
    sig_1hz = 2.0 * np.sin(2 * np.pi * 1.0 * t)
    flicker = 1.5 * np.sin(2 * np.pi * 12.0 * t)
    raw = dc_offset + sig_1hz + flicker

    filtered = proc.filter_signal(raw)

    # DC offset should be centered near 0.0
    assert abs(np.mean(filtered)) < 0.20
    # Flicker noise should be heavily attenuated
    assert np.max(np.abs(filtered)) < 2.5


def test_ppg_pulse_rate_estimation_60_bpm():
    """Verify accurate pulse rate estimation for 60.0 BPM (1000 ms IBI)."""
    fs = 30.0
    _, raw = generate_synthetic_ppg(
        duration_sec=10.0,
        sampling_rate_hz=fs,
        pulse_rate_bpm=60.0,
        noise_amplitude=0.015,
        seed=10,
    )
    proc = PPGProcessor(sampling_rate_hz=fs)
    res = proc.process(raw)

    assert res.pulse_rate_bpm is not None
    assert abs(res.pulse_rate_bpm - 60.0) <= 1.5
    assert res.signal_quality >= 0.70
    assert res.quality_label == "GOOD"
    assert res.is_usable is True

    # Check IBIs
    for ibi in res.ibi_intervals_ms:
        assert abs(ibi - 1000.0) < 70.0


def test_ppg_pulse_rate_estimation_90_bpm():
    """Verify accurate pulse rate estimation for 90.0 BPM (667 ms IBI)."""
    fs = 30.0
    _, raw = generate_synthetic_ppg(
        duration_sec=10.0,
        sampling_rate_hz=fs,
        pulse_rate_bpm=90.0,
        noise_amplitude=0.015,
        seed=20,
    )
    proc = PPGProcessor(sampling_rate_hz=fs)
    res = proc.process(raw)

    assert res.pulse_rate_bpm is not None
    assert abs(res.pulse_rate_bpm - 90.0) <= 1.5
    assert res.signal_quality >= 0.70
    assert res.quality_label == "GOOD"
    assert res.is_usable is True


def test_ppg_pulse_rate_estimation_120_bpm():
    """Verify accurate pulse rate estimation for 120.0 BPM (500 ms IBI)."""
    fs = 30.0
    _, raw = generate_synthetic_ppg(
        duration_sec=8.0,
        sampling_rate_hz=fs,
        pulse_rate_bpm=120.0,
        noise_amplitude=0.02,
        seed=30,
    )
    proc = PPGProcessor(sampling_rate_hz=fs)
    res = proc.process(raw)

    assert res.pulse_rate_bpm is not None
    assert abs(res.pulse_rate_bpm - 120.0) <= 2.0


def test_ppg_motion_artifact_suppresses_rate_as_poor():
    """Verify heavy motion artifacts result in POOR quality and null pulse rate."""
    fs = 30.0
    _, raw = generate_synthetic_ppg(
        duration_sec=8.0,
        sampling_rate_hz=fs,
        pulse_rate_bpm=72.0,
        noise_amplitude=0.40,
        motion_amplitude=5.0,
        seed=40,
    )
    proc = PPGProcessor(sampling_rate_hz=fs)
    res = proc.process(raw)

    assert res.quality_label == "POOR"
    assert res.is_usable is False
    assert res.pulse_rate_bpm is None
    assert "Poor optical contact" in res.message or "noise" in res.message.lower()


def test_ppg_flat_signal_handling():
    """Verify flat constant signal returns POOR quality without error."""
    proc = PPGProcessor(sampling_rate_hz=30.0)
    flat_signal = [128.0] * 150

    res = proc.process(flat_signal)
    assert res.quality_label == "POOR"
    assert res.is_usable is False
    assert res.pulse_rate_bpm is None
    assert len(res.peaks) == 0


def test_ppg_short_and_empty_signals_safety():
    """Verify processor does not crash on empty or very short arrays."""
    proc = PPGProcessor(sampling_rate_hz=30.0)

    # Empty
    res_empty = proc.process([])
    assert res_empty.pulse_rate_bpm is None
    assert res_empty.quality_label == "POOR"
    assert res_empty.is_usable is False

    # Short
    res_short = proc.process([120.0] * 10)
    assert res_short.pulse_rate_bpm is None
    assert res_short.quality_label == "POOR"
    assert res_short.is_usable is False


def test_ppg_nan_and_inf_safety():
    """Verify corrupted signals with NaNs and infs are sanitized gracefully."""
    proc = PPGProcessor(sampling_rate_hz=30.0)
    corrupted = [float("nan")] * 15 + [float("inf")] * 5 + [128.0] * 80

    res = proc.process(corrupted)
    assert isinstance(res, PPGAnalysisResult)
    assert np.all(np.isfinite(res.filtered_signal))
    assert 0.0 <= res.signal_quality <= 1.0


def test_ppg_service_process_window_canonical_contract():
    """Verify PPGService.process_window produces compliant PPGProcessResponse."""
    fs = 30.0
    _, raw = generate_synthetic_ppg(
        duration_sec=8.0,
        sampling_rate_hz=fs,
        pulse_rate_bpm=75.0,
        noise_amplitude=0.015,
        seed=50,
    )
    service = PPGService(sampling_rate_hz=fs)
    req = PPGProcessRequest(
        sampling_rate_hz=fs,
        signal=raw.tolist(),
        source="smartphone_camera",
    )
    resp = service.process_window(req)

    assert isinstance(resp, PPGProcessResponse)
    assert resp.pulse_rate_bpm is not None
    assert abs(resp.pulse_rate_bpm - 75.0) <= 2.0
    assert 0.0 <= resp.signal_quality <= 1.0
    assert resp.quality_label in [QualityLabelEnum.GOOD, QualityLabelEnum.FAIR]
    assert len(resp.peaks) > 0
    assert len(resp.filtered_signal) == len(raw)


def test_ppg_service_assess_quality_canonical_contract():
    """Verify PPGService.assess_quality produces compliant PPGQualityResponse."""
    fs = 30.0
    _, raw = generate_synthetic_ppg(
        duration_sec=8.0,
        sampling_rate_hz=fs,
        pulse_rate_bpm=70.0,
        noise_amplitude=0.015,
        seed=60,
    )
    service = PPGService(sampling_rate_hz=fs)
    req = PPGQualityRequest(
        sampling_rate_hz=fs,
        signal=raw.tolist(),
    )
    resp = service.assess_quality(req)

    assert isinstance(resp, PPGQualityResponse)
    assert 0.0 <= resp.signal_quality <= 1.0
    assert resp.quality_label == QualityLabelEnum.GOOD
    assert resp.snr_db is not None and resp.snr_db > 0.0
    assert resp.is_usable is True
    assert resp.message is not None


def test_synthetic_ppg_generator_parameters():
    """Verify synthetic waveform generator generates correct signal properties."""
    t, sig = generate_synthetic_ppg(
        duration_sec=5.0,
        sampling_rate_hz=30.0,
        pulse_rate_bpm=80.0,
        dc_offset=130.0,
        ac_amplitude=10.0,
        seed=70,
    )
    assert len(t) == 150
    assert len(sig) == 150
    assert np.all(np.isfinite(sig))
    assert abs(np.mean(sig) - 130.0) < 5.0
    assert np.min(sig) > 0.0
