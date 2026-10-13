"""Unit and integration test suite for Phase 15: Clinical Audit Compliance, Cryptographic Integrity & System Diagnostics."""

from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.models.schemas import (
    AuditIntegrityManifest,
    CanonicalMeasurement,
    ModeEnum,
    QualityLabelEnum,
    SessionCreateRequest,
    TrendLabelEnum,
)
from backend.services.audit_service import AuditService
from backend.services.session_service import SessionService


def test_audit_service_deterministic_checksum():
    """Verify cryptographic SHA-256 checksum is deterministic and repeatable."""
    service = AuditService()
    t_now = datetime.now(timezone.utc).isoformat()

    measurements = [
        CanonicalMeasurement(
            timestamp=t_now,
            mode=ModeEnum.OFFLINE_DEMO,
            heart_rate_bpm=72.0,
            pulse_rate_bpm=72.0,
            signal_quality=0.95,
            quality_label=QualityLabelEnum.GOOD,
            rhythm_class="Normal Sinus Rhythm",
            trend=TrendLabelEnum.STABLE,
            model_version="test-1.0",
        )
    ]

    manifest1 = service.compute_session_manifest("test-sess-1", measurements)
    manifest2 = service.compute_session_manifest("test-sess-1", measurements)

    assert isinstance(manifest1, AuditIntegrityManifest)
    assert len(manifest1.sha256_checksum) == 64
    assert manifest1.sha256_checksum == manifest2.sha256_checksum
    assert manifest1.record_hash == manifest2.record_hash
    assert manifest1.tamper_verified is True


def test_audit_service_tamper_detection():
    """Verify that any tampering with recorded telemetry triggers verification failure."""
    service = AuditService()
    t_now = datetime.now(timezone.utc).isoformat()

    m_original = [
        CanonicalMeasurement(
            timestamp=t_now,
            mode=ModeEnum.OFFLINE_DEMO,
            heart_rate_bpm=75.0,
            pulse_rate_bpm=75.0,
            signal_quality=0.90,
            quality_label=QualityLabelEnum.GOOD,
            rhythm_class="Normal Sinus Rhythm",
            model_version="test-1.0",
        )
    ]
    manifest = service.compute_session_manifest("test-sess-tamper", m_original)

    # Corrupt data
    m_tampered = [
        CanonicalMeasurement(
            timestamp=t_now,
            mode=ModeEnum.OFFLINE_DEMO,
            heart_rate_bpm=140.0,  # Falsified heart rate
            pulse_rate_bpm=75.0,
            signal_quality=0.90,
            quality_label=QualityLabelEnum.GOOD,
            rhythm_class="Normal Sinus Rhythm",
            model_version="test-1.0",
        )
    ]

    is_valid = service.verify_session_integrity("test-sess-tamper", manifest.sha256_checksum, m_tampered)
    assert is_valid is False, "Tampered telemetry must fail integrity verification!"


def test_audit_deep_system_diagnostics():
    """Verify deep multi-module diagnostic self-check reports all subsystems operational."""
    service = AuditService()
    diag = service.run_deep_system_diagnostics()

    assert diag.system_status == "ALL_SYSTEMS_OPERATIONAL"
    assert len(diag.subsystems) >= 10

    subsystem_names = [s.subsystem for s in diag.subsystems]
    assert "ECG_Pan_Tompkins_Engine" in subsystem_names
    assert "PPG_Optical_Processor" in subsystem_names
    assert "Arrhythmia_Classifier_Model" in subsystem_names
    assert "Trend_Forecasting_Engine" in subsystem_names
    assert "Multimodal_Fusion_Synchronizer" in subsystem_names
    assert "Emergency_Alerts_Engine" in subsystem_names
    assert "Clinical_Reporting_Engine" in subsystem_names
    assert "Concurrency_Benchmark_Engine" in subsystem_names
    assert "Session_Storage_Engine" in subsystem_names
    assert "Cryptographic_Audit_Integrity" in subsystem_names

    for s in diag.subsystems:
        assert s.status == "OPERATIONAL"
        assert s.latency_ms >= 0.0


def test_audit_compliance_statement():
    """Verify clinical research compliance statement and standards references."""
    service = AuditService()
    comp = service.get_compliance_statement()

    assert comp.audit_logging_enabled is True
    assert any("IEC 62304" in std for std in comp.standards_referenced)
    assert any("ISO 13485" in std for std in comp.standards_referenced)
    assert "SHA-256" in comp.data_integrity
    assert "Research prototype" in comp.disclaimer


def test_audit_rest_api_lifecycle():
    """Verify REST API audit endpoints: /integrity/{id}, /system-diagnostics, and /compliance."""
    client = TestClient(app)

    # 1. Create a monitoring session
    res_sess = client.post("/api/v1/sessions", json={"mode": "offline_demo", "metadata": {"unit": "ICU"}})
    assert res_sess.status_code == 201
    sess_id = res_sess.json()["session_id"]

    # 2. Check session integrity manifest
    res_audit = client.get(f"/api/v1/audit/integrity/{sess_id}")
    assert res_audit.status_code == 200
    manifest = res_audit.json()
    assert manifest["session_id"] == sess_id
    assert len(manifest["sha256_checksum"]) == 64
    assert manifest["tamper_verified"] is True

    # 3. Check deep diagnostics endpoint
    res_diag = client.get("/api/v1/audit/system-diagnostics")
    assert res_diag.status_code == 200
    diag = res_diag.json()
    assert diag["system_status"] == "ALL_SYSTEMS_OPERATIONAL"
    assert len(diag["subsystems"]) >= 10

    # 4. Check compliance endpoint
    res_comp = client.get("/api/v1/audit/compliance")
    assert res_comp.status_code == 200
    comp = res_comp.json()
    assert comp["audit_logging_enabled"] is True
    assert len(comp["standards_referenced"]) >= 3


def test_audit_html_dashboard_endpoint():
    """Verify GET /api/v1/audit/html serves valid interactive audit HTML."""
    client = TestClient(app)
    res_html = client.get("/api/v1/audit/html")

    assert res_html.status_code == 200
    assert "text/html" in res_html.headers.get("content-type", "")
    html = res_html.text
    assert "BioQuest Clinical Audit Compliance & System Diagnostics" in html
    assert "ALL SYSTEMS OPERATIONAL" in html
    assert "ECG_Pan_Tompkins_Engine" in html
    assert "Cryptographic_Audit_Integrity" in html
    assert "IEC 62304" in html

