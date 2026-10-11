"""Automated unit and integration tests for Phase 12 Automated Clinical Diagnostic Report Generation."""

from datetime import datetime, timezone, timedelta
import pytest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.api.deps import get_report_service, get_session_service, get_alert_service
from backend.models.schemas import (
    AlertSeverityEnum,
    AlertTypeEnum,
    CardiacAlert,
    CanonicalMeasurement,
    ClinicalReportSummary,
    ModeEnum,
    QualityLabelEnum,
    TrendLabelEnum,
)
from backend.services.report_service import ReportService


@pytest.fixture
def client():
    """Create a test client with cleared session and alert services."""
    get_session_service().clear()
    get_alert_service().clear()
    return TestClient(app)


def test_report_service_generation_from_measurements():
    """Verify statistical aggregation and clinical impression generation from measurements."""
    service = ReportService()
    t_base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

    measurements = [
        CanonicalMeasurement(
            timestamp=(t_base + timedelta(seconds=i)).isoformat(),
            mode=ModeEnum.ECG_DATASET,
            heart_rate_bpm=72.0 + (i % 3),
            pulse_rate_bpm=72.0 + (i % 3),
            signal_quality=0.92,
            quality_label=QualityLabelEnum.GOOD,
            rhythm_class="Normal Sinus Rhythm",
            rhythm_confidence=0.95,
            trend=TrendLabelEnum.STABLE,
            model_version="test-1.0",
        )
        for i in range(10)
    ]

    report = service.build_report_from_measurements(measurements, mode="ecg_dataset")
    assert isinstance(report, ClinicalReportSummary)
    assert report.total_measurements == 10
    assert report.duration_seconds >= 9.0
    assert report.mean_heart_rate_bpm is not None
    assert 72.0 <= report.mean_heart_rate_bpm <= 75.0
    assert report.dominant_rhythm == "Normal Sinus Rhythm"
    assert report.quality_distribution["GOOD"] == 100.0
    assert len(report.clinical_impressions) >= 2
    assert "Research prototype" in report.disclaimer


def test_report_service_arrhythmia_and_alerts_aggregation():
    """Verify arrhythmia distribution calculation and emergency alert aggregation."""
    service = ReportService()
    t_base = datetime(2026, 10, 1, 12, 0, 0, tzinfo=timezone.utc)

    measurements = [
        CanonicalMeasurement(
            timestamp=(t_base + timedelta(seconds=i)).isoformat(),
            mode=ModeEnum.OFFLINE_DEMO,
            heart_rate_bpm=110.0 if i > 5 else 75.0,
            pulse_rate_bpm=110.0 if i > 5 else 75.0,
            signal_quality=0.88,
            quality_label=QualityLabelEnum.GOOD,
            rhythm_class="Premature Ventricular Contraction" if i == 7 else "Normal Sinus Rhythm",
            rhythm_confidence=0.89,
            model_version="test-1.0",
        )
        for i in range(10)
    ]

    mock_alert = CardiacAlert(
        alert_id="test-alert-1",
        severity=AlertSeverityEnum.CRITICAL,
        alert_type=AlertTypeEnum.TACHYCARDIA,
        message="Acute tachycardia detected at 110.0 BPM",
    )

    report = service.build_report_from_measurements(
        measurements,
        alerts=[mock_alert],
        patient_notes="Exercise stress test recovery window",
        mode="offline_demo",
    )
    assert report.total_alerts == 1
    assert report.critical_alerts_count == 1
    assert report.rhythm_distribution.get("Premature Ventricular Contraction") == 1
    assert any("Tachycardia" in imp for imp in report.clinical_impressions)
    assert any("Cardiac ectopy" in imp for imp in report.clinical_impressions)
    assert any("Exercise stress" in imp for imp in report.clinical_impressions)


def test_report_service_empty_measurements_safety():
    """Verify graceful handling when measurements list is empty."""
    service = ReportService()
    report = service.build_report_from_measurements([])
    assert report.total_measurements == 0
    assert report.duration_seconds == 0.0
    assert report.overall_sqi == 0.0
    assert "Insufficient" in report.clinical_impressions[0]


def test_report_service_html_rendering():
    """Verify HTML diagnostic report contains valid tags, statistics, and disclaimers."""
    service = ReportService()
    report = service.build_report_from_measurements([
        CanonicalMeasurement(
            timestamp="2026-10-01T12:00:00Z",
            mode=ModeEnum.ECG_DATASET,
            heart_rate_bpm=72.0,
            signal_quality=0.95,
            quality_label=QualityLabelEnum.GOOD,
            model_version="test-1.0",
        )
    ])
    html = service.render_html_report(report)
    assert "<!DOCTYPE html>" in html
    assert "BioQuest Multimodal Cardiac Telemetry" in html
    assert "72.0 BPM" in html
    assert "Research prototype — not intended for medical diagnosis." in html


def test_reports_rest_api_lifecycle(client):
    """Verify REST API endpoints for session summary and HTML export."""
    # 1. Create a session and add measurements
    sess_resp = client.post("/api/v1/sessions", json={"mode": "smartphone_ppg", "notes": "Clinical test session"})
    assert sess_resp.status_code == 201
    sess_id = sess_resp.json()["session_id"]

    for i in range(5):
        m = {
            "timestamp": f"2026-10-01T12:00:0{i}Z",
            "mode": "smartphone_ppg",
            "heart_rate_bpm": None,
            "pulse_rate_bpm": 74.0 + i,
            "signal_quality": 0.90,
            "quality_label": "GOOD",
            "model_version": "ppg-peak-v1.0.0",
        }
        client.post(f"/api/v1/sessions/{sess_id}/measurements", json=m)

    # 2. Get JSON summary
    summary_resp = client.get(f"/api/v1/reports/{sess_id}/summary")
    assert summary_resp.status_code == 200
    summary_data = summary_resp.json()
    assert summary_data["session_id"] == sess_id
    assert summary_data["total_measurements"] == 5
    assert summary_data["mean_pulse_rate_bpm"] is not None

    # 3. Get HTML report
    html_resp = client.get(f"/api/v1/reports/{sess_id}/html")
    assert html_resp.status_code == 200
    assert "text/html" in html_resp.headers["content-type"]
    assert "BioQuest" in html_resp.text

    # 4. Non-existent session returns 404
    fake_id = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/api/v1/reports/{fake_id}/summary").status_code == 404
    assert client.get(f"/api/v1/reports/{fake_id}/html").status_code == 404


def test_reports_rest_api_adhoc_generate(client):
    """Verify POST /api/v1/reports/generate endpoint."""
    payload = {
        "mode": "offline_demo",
        "measurements": [
            {
                "timestamp": "2026-10-01T12:00:00Z",
                "mode": "offline_demo",
                "heart_rate_bpm": 80.0,
                "pulse_rate_bpm": 80.0,
                "signal_quality": 0.94,
                "quality_label": "GOOD",
                "rhythm_class": "Normal Sinus Rhythm",
                "model_version": "test-1.0",
            }
        ],
        "patient_notes": "Ad-hoc test analysis",
    }
    resp = client.post("/api/v1/reports/generate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_measurements"] == 1
    assert data["mean_heart_rate_bpm"] == 80.0
    assert any("Ad-hoc test" in imp for imp in data["clinical_impressions"])
