"""Automated unit and integration tests for Phase 9 Multimodal Synchronization & Fusion."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.api.deps import get_session_service
from backend.models.schemas import (
    CanonicalMeasurement,
    ModeEnum,
    MultimodalAnalyzeRequest,
    MultimodalAnalyzeResponse,
    QualityLabelEnum,
)
from ml.ecg.dataset import ECGDatasetLoader
from ml.ecg.processor import ECGProcessor
from ml.fusion.synchronizer import (
    MultimodalSynchronizer,
    MultimodalSyncResult,
    MODEL_VERSION,
)
from ml.ppg.processor import PPGProcessor
from ml.ppg.synthetic import generate_synthetic_ppg


@pytest.fixture
def client():
    """Create test client and reset session repository."""
    session_service = get_session_service()
    session_service.clear()
    return TestClient(app)


def test_synchronizer_initialization():
    """Verify default parameters and thresholds of MultimodalSynchronizer."""
    sync = MultimodalSynchronizer(
        excellent_threshold_bpm=3.0,
        acceptable_threshold_bpm=8.0,
        min_pat_ms=80.0,
        max_pat_ms=450.0,
        deficit_beat_threshold=2,
    )
    assert sync.excellent_threshold_bpm == 3.0
    assert sync.acceptable_threshold_bpm == 8.0
    assert sync.min_pat_ms == 80.0
    assert sync.max_pat_ms == 450.0
    assert sync.deficit_beat_threshold == 2


def test_rate_agreement_computation():
    """Verify rate discrepancy calculation and qualitative agreement labeling."""
    sync = MultimodalSynchronizer()

    # Excellent agreement (< 3.0 BPM)
    delta, rel, label = sync.compute_rate_agreement(72.0, 73.5)
    assert delta == 1.5
    assert abs(rel - 2.08) < 0.1
    assert label == "EXCELLENT"

    # Acceptable agreement (3.0 - 8.0 BPM)
    delta, rel, label = sync.compute_rate_agreement(70.0, 75.0)
    assert delta == 5.0
    assert label == "ACCEPTABLE"

    # Discrepant agreement (> 8.0 BPM)
    delta, rel, label = sync.compute_rate_agreement(70.0, 85.0)
    assert delta == 15.0
    assert label == "DISCREPANT"

    # Missing / None rates
    delta, rel, label = sync.compute_rate_agreement(72.0, None)
    assert delta is None
    assert label == "UNAVAILABLE"


def test_pulse_arrival_time_estimation():
    """Verify beat-to-beat Pulse Arrival Time calculation under known 220ms phase delay."""
    sync = MultimodalSynchronizer()

    # ECG R-peaks at 0.5s, 1.3s, 2.1s, 2.9s
    r_peaks_sec = np.array([0.50, 1.30, 2.10, 2.90])
    # PPG systolic peaks delayed by exactly 220ms
    ppg_peaks_sec = r_peaks_sec + 0.220

    pat_ms, pat_std = sync.compute_pulse_arrival_time(r_peaks_sec, ppg_peaks_sec)
    assert pat_ms is not None
    assert abs(pat_ms - 220.0) < 1.0
    assert pat_std == 0.0


def test_pulse_deficit_detection():
    """Verify that missed peripheral beats are accurately flagged as pulse deficits."""
    sync = MultimodalSynchronizer(deficit_beat_threshold=2)
    fs_ecg, fs_ppg = 360.0, 30.0

    # Clean ECG with 6 beats
    loader = ECGDatasetLoader()
    sig_ecg, fs_ecg, _ = loader.load_record("100", start_sec=0.0, duration_sec=5.0)
    ecg_res = ECGProcessor(sampling_rate_hz=fs_ecg).process(sig_ecg)

    # PPG pulse train with only 4 beats (significant deficit: 6 vs 4)
    _, short_ppg = generate_synthetic_ppg(
        duration_sec=5.0,
        sampling_rate_hz=fs_ppg,
        pulse_rate_bpm=45.0,
    )
    ppg_res = PPGProcessor(sampling_rate_hz=fs_ppg).process(short_ppg)

    result = sync.synchronize(
        ecg_result=ecg_res,
        ppg_result=ppg_res,
        ecg_sampling_rate_hz=fs_ecg,
        ppg_sampling_rate_hz=fs_ppg,
    )

    assert result.pulse_deficit_detected is True
    assert result.ecg_beat_count > result.ppg_beat_count


def test_fused_quality_and_sensor_failover():
    """Verify dynamic SQI weighting and sensor failover under degraded contact."""
    sync = MultimodalSynchronizer()

    # Both sensors clean
    sqi_both, label_both, rec_both = sync.fuse_quality(0.95, 0.90)
    assert sqi_both >= 0.85
    assert label_both == "GOOD"
    assert rec_both == "BOTH"

    # Optical contact degraded, ECG clean -> failover to ECG
    sqi_ecg_fav, label_ecg, rec_ecg = sync.fuse_quality(ecg_sqi=0.95, ppg_sqi=0.30)
    assert label_ecg in ("GOOD", "FAIR")
    assert rec_ecg == "ECG"

    # ECG degraded (motion), optical PPG clean -> failover to PPG
    sqi_ppg_fav, label_ppg, rec_ppg = sync.fuse_quality(ecg_sqi=0.30, ppg_sqi=0.92)
    assert label_ppg in ("GOOD", "FAIR")
    assert rec_ppg == "PPG"


def test_canonical_measurement_transformation():
    """Verify conversion from MultimodalSyncResult to CanonicalMeasurement schema."""
    sync = MultimodalSynchronizer()
    sync_res = MultimodalSyncResult(
        ecg_heart_rate_bpm=74.5,
        ppg_pulse_rate_bpm=74.0,
        rate_discrepancy_bpm=0.5,
        relative_discrepancy_pct=0.67,
        agreement_level="EXCELLENT",
        pulse_arrival_time_ms=218.0,
        pat_variability_ms=5.0,
        ecg_beat_count=6,
        ppg_beat_count=6,
        pulse_deficit_detected=False,
        ecg_signal_quality=0.95,
        ppg_signal_quality=0.88,
        fused_signal_quality=0.92,
        fused_quality_label="GOOD",
        recommended_primary_modality="BOTH",
    )

    meas = sync.to_canonical_measurement(
        sync_result=sync_res,
        mode=ModeEnum.OFFLINE_DEMO,
        rhythm_class="Normal Sinus Rhythm",
        rhythm_confidence=0.96,
    )

    assert isinstance(meas, CanonicalMeasurement)
    assert meas.mode == ModeEnum.OFFLINE_DEMO
    assert meas.heart_rate_bpm == 74.5
    assert meas.pulse_rate_bpm == 74.0
    assert meas.signal_quality == 0.92
    assert meas.quality_label == QualityLabelEnum.GOOD
    assert meas.rhythm_class == "Normal Sinus Rhythm"
    assert meas.model_version == MODEL_VERSION


def test_multimodal_analyze_api_endpoint(client):
    """Verify POST /api/v1/multimodal/analyze processes paired inputs and records to session."""
    # 1. Create a monitoring session
    sess_resp = client.post("/api/v1/sessions", json={"mode": "offline_demo"})
    session_id = sess_resp.json()["session_id"]

    loader = ECGDatasetLoader()
    sig_ecg, fs_ecg, _ = loader.load_record("100", start_sec=0.0, duration_sec=5.0)

    _, ppg_raw = generate_synthetic_ppg(
        duration_sec=5.0,
        sampling_rate_hz=30.0,
        pulse_rate_bpm=75.0,
        noise_amplitude=0.01,
    )

    req_payload = {
        "ecg_signal": sig_ecg.tolist(),
        "ecg_sampling_rate_hz": fs_ecg,
        "ppg_signal": ppg_raw.tolist(),
        "ppg_sampling_rate_hz": 30.0,
        "record_id": "100",
        "session_id": session_id,
    }

    response = client.post("/api/v1/multimodal/analyze", json=req_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["ecg_heart_rate_bpm"] is not None
    assert data["ppg_pulse_rate_bpm"] is not None
    assert data["rate_discrepancy_bpm"] is not None
    assert data["agreement_level"] in ("EXCELLENT", "ACCEPTABLE")
    assert data["fused_quality_label"] in ("GOOD", "FAIR")
    assert data["canonical_measurement"]["mode"] == "offline_demo"

    # Verify session received measurement
    detail = client.get(f"/api/v1/sessions/{session_id}").json()
    assert len(detail["measurements"]) == 1


def test_multimodal_demo_frame_api_endpoint(client):
    """Verify GET /api/v1/multimodal/demo-frame returns synchronized waveform data."""
    response = client.get("/api/v1/multimodal/demo-frame?start_sec=5.0&duration_sec=5.0&record_id=100")
    assert response.status_code == 200
    data = response.json()

    assert data["mode"] == "offline_demo"
    assert data["record_id"] == "100"
    assert "ecg" in data
    assert "ppg" in data
    assert "fusion" in data
    assert len(data["ecg"]["filtered_waveform"]) == int(360 * 5.0)
    assert len(data["ppg"]["filtered_waveform"]) == int(30 * 5.0)
    assert data["fusion"]["agreement_level"] in ("EXCELLENT", "ACCEPTABLE", "DISCREPANT")
