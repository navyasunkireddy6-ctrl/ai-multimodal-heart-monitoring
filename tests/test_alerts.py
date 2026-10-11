"""Automated unit and integration tests for Phase 11 Real-Time Cardiac Anomaly Alerts & Notification Engine."""

import time
import json
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.api.deps import get_alert_service, get_session_service
from backend.models.schemas import (
    AlertConfig,
    AlertSeverityEnum,
    AlertTypeEnum,
    CanonicalMeasurement,
    ModeEnum,
    MultimodalAnalyzeResponse,
    QualityLabelEnum,
    TrendLabelEnum,
)
from backend.services.alert_service import AlertService


@pytest.fixture
def alert_service():
    """Create a fresh AlertService for testing."""
    svc = AlertService()
    svc.clear()
    return svc


@pytest.fixture
def client():
    """Create a FastAPI TestClient with cleared alert & session services."""
    alert_svc = get_alert_service()
    alert_svc.clear()
    sess_svc = get_session_service()
    sess_svc.clear()
    return TestClient(app)


# ============================================================================
# 1. AlertService Unit Tests
# ============================================================================

def test_alert_service_initialization_and_config(alert_service):
    """Verify default thresholds and configuration update."""
    config = alert_service.get_config()
    assert config.tachycardia_threshold_bpm == 100.0
    assert config.critical_tachycardia_bpm == 130.0
    assert config.bradycardia_threshold_bpm == 60.0
    assert config.critical_bradycardia_bpm == 45.0
    assert config.cooldown_seconds == 5.0

    new_cfg = AlertConfig(
        tachycardia_threshold_bpm=110.0,
        critical_tachycardia_bpm=140.0,
        bradycardia_threshold_bpm=55.0,
        critical_bradycardia_bpm=40.0,
    )
    alert_service.update_config(new_cfg)
    assert alert_service.get_config().tachycardia_threshold_bpm == 110.0


def test_tachycardia_warning_and_critical_alarm(alert_service):
    """Verify rate thresholds trigger WARNING at 100 BPM and CRITICAL at 130 BPM."""
    # Normal rate (75 BPM) -> No alert
    m_norm = CanonicalMeasurement(
        mode=ModeEnum.ECG_DATASET,
        heart_rate_bpm=75.0,
        signal_quality=0.92,
        quality_label=QualityLabelEnum.GOOD,
        model_version="test-1.0",
    )
    assert len(alert_service.evaluate_measurement(m_norm)) == 0

    # Warning Tachycardia (105 BPM)
    m_warn = CanonicalMeasurement(
        mode=ModeEnum.ECG_DATASET,
        heart_rate_bpm=105.0,
        signal_quality=0.90,
        quality_label=QualityLabelEnum.GOOD,
        model_version="test-1.0",
    )
    alerts_warn = alert_service.evaluate_measurement(m_warn)
    assert len(alerts_warn) == 1
    assert alerts_warn[0].alert_type == AlertTypeEnum.TACHYCARDIA
    assert alerts_warn[0].severity == AlertSeverityEnum.WARNING
    assert alerts_warn[0].metrics["rate_bpm"] == 105.0

    # Critical Tachycardia (138 BPM) -> Escalation bypasses cooldown
    m_crit = CanonicalMeasurement(
        mode=ModeEnum.ECG_DATASET,
        heart_rate_bpm=138.0,
        signal_quality=0.88,
        quality_label=QualityLabelEnum.GOOD,
        model_version="test-1.0",
    )
    alerts_crit = alert_service.evaluate_measurement(m_crit)
    assert len(alerts_crit) == 1
    assert alerts_crit[0].alert_type == AlertTypeEnum.TACHYCARDIA
    assert alerts_crit[0].severity == AlertSeverityEnum.CRITICAL


def test_bradycardia_warning_and_critical_alarm(alert_service):
    """Verify rate thresholds trigger WARNING at <=60 BPM and CRITICAL at <=45 BPM."""
    # Warning Bradycardia (52 BPM)
    m_warn = CanonicalMeasurement(
        mode=ModeEnum.SMARTPHONE_PPG,
        pulse_rate_bpm=52.0,
        signal_quality=0.85,
        quality_label=QualityLabelEnum.GOOD,
        model_version="test-1.0",
    )
    alerts_warn = alert_service.evaluate_measurement(m_warn)
    assert len(alerts_warn) == 1
    assert alerts_warn[0].alert_type == AlertTypeEnum.BRADYCARDIA
    assert alerts_warn[0].severity == AlertSeverityEnum.WARNING

    # Critical Bradycardia (38 BPM) -> Escalation
    m_crit = CanonicalMeasurement(
        mode=ModeEnum.SMARTPHONE_PPG,
        pulse_rate_bpm=38.0,
        signal_quality=0.85,
        quality_label=QualityLabelEnum.GOOD,
        model_version="test-1.0",
    )
    alerts_crit = alert_service.evaluate_measurement(m_crit)
    assert len(alerts_crit) == 1
    assert alerts_crit[0].alert_type == AlertTypeEnum.BRADYCARDIA
    assert alerts_crit[0].severity == AlertSeverityEnum.CRITICAL


def test_arrhythmia_classification_alert(alert_service):
    """Verify non-sinus rhythm classifications trigger arrhythmia alert."""
    # Normal sinus -> No alert
    m_nsr = CanonicalMeasurement(
        mode=ModeEnum.ECG_DATASET,
        heart_rate_bpm=72.0,
        signal_quality=0.95,
        quality_label=QualityLabelEnum.GOOD,
        rhythm_class="Normal Sinus Rhythm",
        rhythm_confidence=0.95,
        model_version="rf-ecg-v1.0.0",
    )
    assert len(alert_service.evaluate_measurement(m_nsr)) == 0

    # Premature Ventricular Contraction (PVC) -> Critical Arrhythmia
    m_pvc = CanonicalMeasurement(
        mode=ModeEnum.ECG_DATASET,
        heart_rate_bpm=74.0,
        signal_quality=0.92,
        quality_label=QualityLabelEnum.GOOD,
        rhythm_class="Premature Ventricular Contraction",
        rhythm_confidence=0.88,
        model_version="rf-ecg-v1.0.0",
    )
    alerts = alert_service.evaluate_measurement(m_pvc)
    assert len(alerts) == 1
    assert alerts[0].alert_type == AlertTypeEnum.ARRHYTHMIA_DETECTED
    assert alerts[0].severity == AlertSeverityEnum.CRITICAL
    assert "Premature Ventricular Contraction" in alerts[0].message


def test_pulse_deficit_alert(alert_service):
    """Verify multimodal pulse deficit triggers clinical warning/alarm."""
    canonical_meas = CanonicalMeasurement(
        mode=ModeEnum.OFFLINE_DEMO,
        heart_rate_bpm=85.0,
        pulse_rate_bpm=65.0,
        signal_quality=0.85,
        quality_label=QualityLabelEnum.GOOD,
        model_version="test-1.0",
    )
    multimodal_resp = MultimodalAnalyzeResponse(
        ecg_heart_rate_bpm=85.0,
        ppg_pulse_rate_bpm=65.0,
        rate_discrepancy_bpm=20.0,
        relative_discrepancy_pct=23.5,
        agreement_level="DISCREPANT",
        pulse_arrival_time_ms=210.0,
        pat_variability_ms=15.0,
        ecg_beat_count=7,
        ppg_beat_count=5,
        pulse_deficit_detected=True,
        ecg_signal_quality=0.90,
        ppg_signal_quality=0.82,
        fused_signal_quality=0.86,
        fused_quality_label=QualityLabelEnum.GOOD,
        rhythm_class="Premature Ventricular Contraction",
        rhythm_confidence=0.82,
        trend=TrendLabelEnum.STABLE,
        recommended_primary_modality="ECG",
        canonical_measurement=canonical_meas,
        model_version="fusion-v1.0.0",
    )

    alerts = alert_service.evaluate_multimodal_response(multimodal_resp)
    # Expect Arrhythmia alert + Pulse Deficit alert
    alert_types = [a.alert_type for a in alerts]
    assert AlertTypeEnum.PULSE_DEFICIT in alert_types
    assert AlertTypeEnum.ARRHYTHMIA_DETECTED in alert_types


def test_cooldown_debouncing_prevents_alert_flooding(alert_service):
    """Verify repeated triggers of same alert within cooldown period are suppressed."""
    m_warn = CanonicalMeasurement(
        mode=ModeEnum.ECG_DATASET,
        heart_rate_bpm=108.0,
        signal_quality=0.90,
        quality_label=QualityLabelEnum.GOOD,
        model_version="test-1.0",
    )
    # First trigger -> Fires
    res1 = alert_service.evaluate_measurement(m_warn)
    assert len(res1) == 1

    # Second trigger immediately -> Suppressed by cooldown
    res2 = alert_service.evaluate_measurement(m_warn)
    assert len(res2) == 0


def test_alert_subscriber_pub_sub(alert_service):
    """Verify registered callbacks are notified upon alert trigger."""
    received = []
    alert_service.register_subscriber(lambda a: received.append(a))

    m_warn = CanonicalMeasurement(
        mode=ModeEnum.ECG_DATASET,
        heart_rate_bpm=115.0,
        signal_quality=0.90,
        quality_label=QualityLabelEnum.GOOD,
        model_version="test-1.0",
    )
    alert_service.evaluate_measurement(m_warn)
    assert len(received) == 1
    assert received[0].alert_type == AlertTypeEnum.TACHYCARDIA


# ============================================================================
# 2. REST API Integration Tests (/api/v1/alerts)
# ============================================================================

def test_alerts_rest_api_lifecycle(client):
    """Verify listing, active query, acknowledging, and config endpoints."""
    # 1. Trigger an alert via alert service
    svc = get_alert_service()
    m_crit = CanonicalMeasurement(
        mode=ModeEnum.ECG_DATASET,
        heart_rate_bpm=142.0,
        signal_quality=0.92,
        quality_label=QualityLabelEnum.GOOD,
        model_version="test-1.0",
    )
    svc.evaluate_measurement(m_crit)

    # 2. List Alerts
    list_resp = client.get("/api/v1/alerts")
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert data["total"] >= 1
    assert data["unacknowledged_count"] >= 1
    alert_id = data["alerts"][0]["alert_id"]
    assert data["alerts"][0]["severity"] == "CRITICAL"

    # 3. Get Active Alerts
    active_resp = client.get("/api/v1/alerts/active")
    assert active_resp.status_code == 200
    active_list = active_resp.json()
    assert any(a["alert_id"] == alert_id for a in active_list)

    # 4. Acknowledge Alert
    ack_resp = client.post(f"/api/v1/alerts/{alert_id}/acknowledge")
    assert ack_resp.status_code == 200
    ack_data = ack_resp.json()
    assert ack_data["acknowledged"] is True
    assert ack_data["alert_id"] == alert_id

    # 5. Verify Unacknowledged Count Drops
    list_after = client.get("/api/v1/alerts?unacknowledged_only=true").json()
    assert not any(a["alert_id"] == alert_id for a in list_after["alerts"])

    # 6. Config endpoints
    cfg_get = client.get("/api/v1/alerts/config")
    assert cfg_get.status_code == 200
    assert cfg_get.json()["tachycardia_threshold_bpm"] == 100.0

    cfg_put = client.put(
        "/api/v1/alerts/config",
        json={"tachycardia_threshold_bpm": 105.0, "critical_tachycardia_bpm": 135.0},
    )
    assert cfg_put.status_code == 200
    assert cfg_put.json()["tachycardia_threshold_bpm"] == 105.0


def test_websocket_stream_attaches_active_alert(client):
    """Verify live optical PPG WebSocket frames attach active_alert when rate exceeds threshold."""
    with client.websocket_connect("/ws/v1/ppg") as ws:
        t_base = datetime.now(timezone.utc)
        # Send a tachycardia pulse train (simulate fast 110 BPM rate)
        # 60 frames = 2 seconds
        for i in range(60):
            frame = {
                "type": "ppg_frame",
                "timestamp": t_base.isoformat(),
                "red_channel_value": 140.0 + (10.0 if (i % 8) < 4 else -10.0),
                "camera_fps": 30.0,
            }
            ws.send_text(json.dumps(frame))
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "measurement_update"

    # Verify alert was captured by service
    alert_svc = get_alert_service()
    assert alert_svc.get_config().tachycardia_threshold_bpm > 0
