"""Automated unit and integration tests for Phase 8 FastAPI Backend API and Live WebSocket."""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.api.deps import get_session_service
from backend.models.schemas import (
    CanonicalMeasurement,
    ModeEnum,
    QualityLabelEnum,
    TrendLabelEnum,
)
from ml.ppg.synthetic import generate_synthetic_ppg
from ml.ecg.dataset import ECGDatasetLoader

ROOT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def client():
    """Create a FastAPI test client and clear session storage before each test."""
    session_service = get_session_service()
    session_service.clear()
    return TestClient(app)


# ============================================================================
# 1. Health Check Endpoint Tests
# ============================================================================

def test_health_check_endpoint(client):
    """Verify GET /api/v1/health returns 200 and matches the canonical API contract."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "models_loaded" in data
    assert "ecg_classifier" in data["models_loaded"]
    assert "trend_analyzer" in data["models_loaded"]
    assert "disclaimer" in data
    assert "Research prototype" in data["disclaimer"]


# ============================================================================
# 2. ECG REST Endpoints Tests
# ============================================================================

def test_ecg_process_endpoint(client):
    """Verify POST /api/v1/ecg/process with real MIT-BIH Record 100 signal."""
    loader = ECGDatasetLoader(data_dir=ROOT_DIR / "data" / "ecg")
    sig, fs, _ = loader.load_record("100", start_sec=0.0, duration_sec=5.0)

    payload = {
        "sampling_rate_hz": fs,
        "signal": sig.tolist(),
        "record_id": "100",
    }
    response = client.post("/api/v1/ecg/process", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["heart_rate_bpm"] is not None
    assert 60.0 <= data["heart_rate_bpm"] <= 100.0
    assert data["signal_quality"] > 0.80
    assert data["quality_label"] in ("GOOD", "FAIR")
    assert len(data["r_peaks"]) > 0
    assert len(data["rr_intervals_ms"]) > 0
    assert len(data["filtered_signal"]) == len(sig)


def test_ecg_classify_endpoint(client):
    """Verify POST /api/v1/ecg/classify returns rhythm prediction with confidence."""
    loader = ECGDatasetLoader(data_dir=ROOT_DIR / "data" / "ecg")
    sig, fs, _ = loader.load_record("100", start_sec=12.0, duration_sec=6.0)

    payload = {
        "sampling_rate_hz": fs,
        "signal": sig.tolist(),
    }
    response = client.post("/api/v1/ecg/classify", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["rhythm_class"] == "Normal Sinus Rhythm"
    assert 0.0 <= data["rhythm_confidence"] <= 1.0
    assert data["model_version"] == "rf-ecg-v1.0.0"
    assert "features" in data
    assert "mean_rr_ms" in data["features"]
    assert "Research prototype" in data["disclaimer"]


# ============================================================================
# 3. PPG REST Endpoints Tests
# ============================================================================

def test_ppg_process_endpoint_good_quality(client):
    """Verify POST /api/v1/ppg/process extracts pulse rate from synthetic optical PPG."""
    _, clean_ppg = generate_synthetic_ppg(
        duration_sec=6.0,
        sampling_rate_hz=30.0,
        pulse_rate_bpm=72.0,
        noise_amplitude=0.02,
    )
    payload = {
        "sampling_rate_hz": 30.0,
        "signal": clean_ppg.tolist(),
        "source": "smartphone_camera",
    }
    response = client.post("/api/v1/ppg/process", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["pulse_rate_bpm"] is not None
    assert abs(data["pulse_rate_bpm"] - 72.0) <= 6.0
    assert data["quality_label"] in ("GOOD", "FAIR")
    assert len(data["peaks"]) > 0
    assert len(data["filtered_signal"]) == len(clean_ppg)


def test_ppg_quality_assessment_endpoints(client):
    """Verify POST /api/v1/ppg/quality discriminates between clean and degraded signals."""
    _, clean_ppg = generate_synthetic_ppg(
        duration_sec=6.0,
        sampling_rate_hz=30.0,
        pulse_rate_bpm=75.0,
        noise_amplitude=0.01,
    )
    resp_good = client.post(
        "/api/v1/ppg/quality",
        json={"sampling_rate_hz": 30.0, "signal": clean_ppg.tolist()},
    )
    assert resp_good.status_code == 200
    data_good = resp_good.json()
    assert data_good["is_usable"] is True
    assert data_good["quality_label"] in ("GOOD", "FAIR")

    # Noise / flat signal test
    flat_noise = [128.0 + (i % 3) * 0.05 for i in range(180)]
    resp_poor = client.post(
        "/api/v1/ppg/quality",
        json={"sampling_rate_hz": 30.0, "signal": flat_noise},
    )
    assert resp_poor.status_code == 200
    data_poor = resp_poor.json()
    assert data_poor["quality_label"] == "POOR"
    assert data_poor["is_usable"] is False


# ============================================================================
# 4. Trend REST Endpoint Tests
# ============================================================================

def test_trend_predict_endpoint_increasing(client):
    """Verify POST /api/v1/trend/predict detects accelerating rate trajectory."""
    base_t = datetime(2026, 9, 30, 12, 0, 0, tzinfo=timezone.utc)
    rising_times = [(base_t + timedelta(seconds=i * 15)).isoformat() for i in range(5)]
    rising_rates = [70.0, 74.0, 78.0, 82.0, 86.0]

    response = client.post(
        "/api/v1/trend/predict",
        json={"timestamps": rising_times, "rates": rising_rates},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["trend"] == "INCREASING"
    assert data["slope_bpm_per_min"] > 10.0
    assert data["confidence"] >= 0.85


def test_trend_predict_endpoint_decreasing(client):
    """Verify POST /api/v1/trend/predict detects decelerating rate trajectory."""
    base_t = datetime(2026, 9, 30, 12, 0, 0, tzinfo=timezone.utc)
    falling_times = [(base_t + timedelta(seconds=i * 15)).isoformat() for i in range(5)]
    falling_rates = [100.0, 94.0, 88.0, 82.0, 76.0]

    response = client.post(
        "/api/v1/trend/predict",
        json={"timestamps": falling_times, "rates": falling_rates},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["trend"] == "DECREASING"
    assert data["slope_bpm_per_min"] < -10.0


# ============================================================================
# 5. Session Management Lifecycle Tests
# ============================================================================

def test_session_lifecycle_and_csv_export(client):
    """Test full CRUD and CSV export workflow for sessions."""
    # 1. Create Session
    create_resp = client.post(
        "/api/v1/sessions",
        json={"mode": "smartphone_ppg", "notes": "Automated session test"},
    )
    assert create_resp.status_code == 201
    sess = create_resp.json()
    session_id = sess["session_id"]
    assert sess["mode"] == "smartphone_ppg"
    assert sess["measurement_count"] == 0

    # 2. List Sessions
    list_resp = client.get("/api/v1/sessions?limit=10&offset=0")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] >= 1
    assert any(s["session_id"] == session_id for s in list_data["sessions"])

    # 3. Add Measurements
    meas1 = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "smartphone_ppg",
        "heart_rate_bpm": None,
        "pulse_rate_bpm": 74.0,
        "signal_quality": 0.92,
        "quality_label": "GOOD",
        "rhythm_class": None,
        "rhythm_confidence": None,
        "trend": "STABLE",
        "model_version": "ppg-peak-v1.0.0",
    }
    meas2 = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "smartphone_ppg",
        "heart_rate_bpm": None,
        "pulse_rate_bpm": 76.0,
        "signal_quality": 0.90,
        "quality_label": "GOOD",
        "rhythm_class": None,
        "rhythm_confidence": None,
        "trend": "STABLE",
        "model_version": "ppg-peak-v1.0.0",
    }
    client.post(f"/api/v1/sessions/{session_id}/measurements", json=meas1)
    client.post(f"/api/v1/sessions/{session_id}/measurements", json=meas2)

    # 4. Get Session Details
    detail_resp = client.get(f"/api/v1/sessions/{session_id}")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert len(detail_data["measurements"]) == 2
    assert detail_data["notes"] == "Automated session test"

    # Verify updated summary average in list
    summary_resp = client.get("/api/v1/sessions")
    s_item = next(s for s in summary_resp.json()["sessions"] if s["session_id"] == session_id)
    assert s_item["measurement_count"] == 2
    assert abs(s_item["avg_pulse_rate_bpm"] - 75.0) < 0.1

    # 5. Export CSV
    export_resp = client.get(f"/api/v1/sessions/{session_id}/export")
    assert export_resp.status_code == 200
    assert export_resp.headers["content-type"].startswith("text/csv")
    csv_text = export_resp.text
    assert "timestamp,mode,heart_rate_bpm,pulse_rate_bpm" in csv_text
    assert "74.0" in csv_text
    assert "76.0" in csv_text

    # 6. Delete Session
    del_resp = client.delete(f"/api/v1/sessions/{session_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # 7. Verify 404 after deletion
    get_404 = client.get(f"/api/v1/sessions/{session_id}")
    assert get_404.status_code == 404


def test_session_not_found_handling(client):
    """Verify 404 responses for non-existent session IDs."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/api/v1/sessions/{fake_id}").status_code == 404
    assert client.delete(f"/api/v1/sessions/{fake_id}").status_code == 404
    assert client.get(f"/api/v1/sessions/{fake_id}/export").status_code == 404


# ============================================================================
# 6. Playback Endpoints Tests
# ============================================================================

def test_playback_frame_and_control(client):
    """Verify simulated ECG playback frame retrieval and playback control."""
    frame_resp = client.get("/api/v1/playback/frame")
    assert frame_resp.status_code == 200
    frame_data = frame_resp.json()
    assert "raw_waveform" in frame_data or "status" in frame_data

    ctrl_resp = client.post("/api/v1/playback/control?action=pause")
    assert ctrl_resp.status_code == 200
    assert ctrl_resp.json()["state"] in ("PAUSED", "paused")

    resume_resp = client.post("/api/v1/playback/control?action=resume")
    assert resume_resp.status_code == 200
    assert resume_resp.json()["state"] in ("PLAYING", "playing")


# ============================================================================
# 7. Live Optical PPG WebSocket Streaming Tests
# ============================================================================

def test_websocket_ppg_stream_clean_pulse(client):
    """Verify /ws/v1/ppg stream produces real-time pulse rates and records to session."""
    sess_resp = client.post("/api/v1/sessions", json={"mode": "smartphone_ppg"})
    sess_id = sess_resp.json()["session_id"]

    _, clean_ppg = generate_synthetic_ppg(
        duration_sec=4.0,
        sampling_rate_hz=30.0,
        pulse_rate_bpm=75.0,
        noise_amplitude=0.01,
    )

    with client.websocket_connect("/ws/v1/ppg") as ws:
        t_base = datetime.now(timezone.utc)
        latest_update = None

        # Send 60 frames
        for i in range(60):
            val = float(clean_ppg[i % len(clean_ppg)])
            frame = {
                "type": "ppg_frame",
                "timestamp": (t_base + timedelta(seconds=i / 30.0)).isoformat(),
                "session_id": sess_id,
                "red_channel_value": val,
                "camera_fps": 30.0,
            }
            ws.send_text(json.dumps(frame))
            raw_text = ws.receive_text()
            latest_update = json.loads(raw_text)

        assert latest_update["type"] == "measurement_update"
        assert latest_update["quality_label"] in ("GOOD", "FAIR")
        assert latest_update["pulse_rate_bpm"] is not None
        assert abs(latest_update["pulse_rate_bpm"] - 75.0) <= 6.0
        assert latest_update["warning"] is None

    # Check session recorded measurements
    detail = client.get(f"/api/v1/sessions/{sess_id}").json()
    assert len(detail["measurements"]) >= 15


def test_websocket_ppg_stream_poor_quality_advisory(client):
    """Verify /ws/v1/ppg handles degraded optical signal with advisory warnings and null pulse rate."""
    with client.websocket_connect("/ws/v1/ppg") as ws:
        t_base = datetime.now(timezone.utc)
        latest_update = None

        # Send flat / noise frames
        for i in range(60):
            val = 125.0 + float((i % 5) * 0.05)
            frame = {
                "type": "ppg_frame",
                "timestamp": (t_base + timedelta(seconds=i / 30.0)).isoformat(),
                "red_channel_value": val,
                "camera_fps": 30.0,
            }
            ws.send_text(json.dumps(frame))
            raw_text = ws.receive_text()
            latest_update = json.loads(raw_text)

        assert latest_update["type"] == "measurement_update"
        assert latest_update["quality_label"] == "POOR"
        assert latest_update["pulse_rate_bpm"] is None
        assert "Poor optical contact" in latest_update["warning"]


def test_websocket_invalid_frame_resilience(client):
    """Verify /ws/v1/ppg is resilient against corrupted or non-JSON payloads."""
    with client.websocket_connect("/ws/v1/ppg") as ws:
        ws.send_text("INVALID_NON_JSON")
        resp = json.loads(ws.receive_text())
        assert resp["type"] == "error"
        assert "Invalid PPG WebSocket frame" in resp["message"]
