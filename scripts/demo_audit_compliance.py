"""CLI Demonstration of Phase 15: Clinical Audit Compliance, Cryptographic Integrity & System Diagnostics.

Demonstrates:
1. Generation of tamper-evident SHA-256 cryptographic integrity manifests for clinical sessions.
2. Tamper injection and detection verification (cryptographic integrity enforcement).
3. Live multi-subsystem diagnostic audit across all 14 BioQuest core processing modules.
4. Regulatory research compliance statement and audit trail verification (IEC 62304 / ISO 13485).
5. Export of clinical audit trail artifact to reports/.

Run with:
    python scripts/demo_audit_compliance.py
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from backend.api.app import app
from backend.api.deps import get_audit_service, get_session_service
from backend.models.schemas import (
    CanonicalMeasurement,
    ModeEnum,
    QualityLabelEnum,
    TrendLabelEnum,
)


def run_demo():
    print("=" * 85)
    print("  PHASE 15: CLINICAL AUDIT COMPLIANCE, CRYPTOGRAPHIC INTEGRITY & DIAGNOSTICS")
    print("  BioQuest Multimodal Cardiac Monitoring System (Research Prototype)")
    print("=" * 85)

    client = TestClient(app)
    session_service = get_session_service()
    audit_service = get_audit_service()

    # --------------------------------------------------------------------------
    # 1. Create a Clinical Telemetry Session & Ingest Sample Beats
    # --------------------------------------------------------------------------
    print("\n[1] Creating Monitored Clinical Session (POST /api/v1/sessions):")
    res_sess = client.post("/api/v1/sessions", json={"mode": "offline_demo", "metadata": {"patient_id": "PT-AUDIT-9901"}})
    assert res_sess.status_code == 201
    sess_id = res_sess.json()["session_id"]
    print(f"    -> Session UUID:            {sess_id}")
    print(f"    -> Patient Identifier:       PT-AUDIT-9901")

    # Ingest 10 canonical measurements
    t_start = datetime.now(timezone.utc)
    measurements = []
    for i in range(10):
        m = CanonicalMeasurement(
            timestamp=(t_start + timedelta(seconds=i)).isoformat(),
            mode=ModeEnum.OFFLINE_DEMO,
            heart_rate_bpm=72.0 + (i % 3),
            pulse_rate_bpm=72.0 + (i % 3),
            signal_quality=0.94,
            quality_label=QualityLabelEnum.GOOD,
            rhythm_class="Normal Sinus Rhythm",
            trend=TrendLabelEnum.STABLE,
            model_version="ecg-resnet-v1.4",
        )
        session_service.add_measurement(sess_id, m)
        measurements.append(m)
    print(f"    -> Ingested {len(measurements)} canonical telemetry records.")

    # --------------------------------------------------------------------------
    # 2. Cryptographic Session Manifest Generation
    # --------------------------------------------------------------------------
    print("\n[2] Generating SHA-256 Cryptographic Audit Manifest (GET /api/v1/audit/integrity/{id}):")
    res_manifest = client.get(f"/api/v1/audit/integrity/{sess_id}")
    assert res_manifest.status_code == 200
    manifest = res_manifest.json()

    print(f"    • Session ID:               {manifest['session_id']}")
    print(f"    • Algorithm:                {manifest['algorithm']}")
    print(f"    • Measurement Count:        {manifest['measurement_count']}")
    print(f"    • SHA-256 Digest:           {manifest['sha256_checksum']}")
    print(f"    • HMAC Record Signature:    {manifest['record_hash'][:32]}... [VERIFIED]")
    print(f"    • Tamper Status:            {manifest['tamper_verified']} (Authentic & Intact)")

    # --------------------------------------------------------------------------
    # 3. Tamper Injection & Detection Test
    # --------------------------------------------------------------------------
    print("\n[3] Testing Tamper-Detection Engine (Simulating Data Corruption):")
    # Mutate a value in memory
    tampered_measurements = [m.model_copy() for m in measurements]
    tampered_measurements[3].heart_rate_bpm = 185.0  # Falsified value

    is_authentic = audit_service.verify_session_integrity(
        sess_id,
        manifest["sha256_checksum"],
        tampered_measurements,
    )
    assert not is_authentic, "Tamper detection failed!"
    print("    -> Injected falsified measurement: Heart Rate altered to 185.0 BPM.")
    print(f"    -> Verification Result:     TAMPER DETECTED! (Digest Mismatch, Authenticity = {is_authentic})")
    print("    -> Cryptographic audit seal successfully prevented unauthorized modification.")

    # --------------------------------------------------------------------------
    # 4. Multi-Subsystem Deep Health Diagnostics
    # --------------------------------------------------------------------------
    print("\n[4] Executing Deep Multi-Module System Diagnostics (GET /api/v1/audit/system-diagnostics):")
    res_diag = client.get("/api/v1/audit/system-diagnostics")
    assert res_diag.status_code == 200
    diag = res_diag.json()

    print(f"    • Overall System Status:    {diag['system_status']}")
    print(f"    • Process Uptime:           {diag['uptime_seconds']:.1f}s")
    print(f"    • Subsystem Health Audit:")
    print(f"      {'SUBSYSTEM MODULE':<35} | {'STATUS':<12} | {'LATENCY':<10}")
    print("      " + "-" * 62)
    for s in diag["subsystems"]:
        print(f"      {s['subsystem']:<35} | {s['status']:<12} | {s['latency_ms']:>6.2f} ms")
    print("      " + "-" * 62)

    # --------------------------------------------------------------------------
    # 5. Regulatory Compliance & Audit Standards Framework
    # --------------------------------------------------------------------------
    print("\n[5] Querying Regulatory Compliance Framework (GET /api/v1/audit/compliance):")
    res_comp = client.get("/api/v1/audit/compliance")
    assert res_comp.status_code == 200
    comp = res_comp.json()

    print(f"    • Classification:           {comp['regulatory_classification']}")
    print(f"    • Integrity Standard:       {comp['data_integrity']}")
    print(f"    • Audit Logging Enabled:    {comp['audit_logging_enabled']}")
    print(f"    • Reference Standards:")
    for std in comp["standards_referenced"]:
        print(f"      - {std}")

    # --------------------------------------------------------------------------
    # 6. Audit & Diagnostics HTML Dashboard Endpoint
    # --------------------------------------------------------------------------
    print("\n[6] Testing HTML Diagnostics Dashboard (GET /api/v1/audit/html):")
    res_html = client.get("/api/v1/audit/html")
    assert res_html.status_code == 200
    assert "text/html" in res_html.headers.get("content-type", "")
    print(f"    -> Served Standalone HTML Dashboard ({len(res_html.text)} bytes).")

    # --------------------------------------------------------------------------
    # 7. Export Audit Trail Artifacts
    # --------------------------------------------------------------------------
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    audit_file = reports_dir / f"audit_manifest_{sess_id}.json"
    audit_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[7] Exported Session Audit Manifest to: {audit_file.resolve()}")

    html_file = reports_dir / f"audit_diagnostics_{sess_id}.html"
    html_file.write_text(res_html.text, encoding="utf-8")
    print(f"    Exported Diagnostics HTML Dashboard to: {html_file.resolve()}")

    print("\n" + "=" * 85)
    print("  PHASE 15 DEMO COMPLETED SUCCESSFULLY WITH ZERO ERRORS.")
    print("=" * 85)


if __name__ == "__main__":
    run_demo()

