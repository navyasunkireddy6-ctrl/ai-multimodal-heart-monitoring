"""CLI Demonstration of Phase 12: Automated Clinical Diagnostic Report Generation & Session Analytics Export.

Demonstrates:
1. Ingestion of multimodal session measurements (ECG + Optical PPG + PAT + Arrhythmia).
2. Aggregation of vital rate statistics (Mean, Min, Max Heart Rate and Pulse Rate).
3. Signal quality index (SQI) distribution across GOOD, FAIR, and POOR categories.
4. Arrhythmia classification distribution and ectopic beat frequency.
5. Emergency anomaly alerts breakdown and clinical incidence tracking.
6. Generation of structured clinical diagnostic summary via REST API (GET /api/v1/reports/{session_id}/summary).
7. Rendering and export of a self-contained, publication-quality HTML Diagnostic Report with styling and print support.

Run with:
    python scripts/demo_clinical_reporting.py
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
from backend.api.deps import get_alert_service, get_session_service
from backend.models.schemas import (
    AlertSeverityEnum,
    AlertTypeEnum,
    CanonicalMeasurement,
    CardiacAlert,
    ModeEnum,
    QualityLabelEnum,
    SessionCreateRequest,
    TrendLabelEnum,
)


def run_demo():
    print("=" * 80)
    print("  PHASE 12: AUTOMATED CLINICAL DIAGNOSTIC REPORT GENERATION & EXPORT")
    print("  BioQuest Multimodal Cardiac Monitoring System (Research Prototype)")
    print("=" * 80)

    client = TestClient(app)
    session_service = get_session_service()
    alert_service = get_alert_service()
    alert_service.clear()

    # --------------------------------------------------------------------------
    # 1. Create a Clinical Telemetry Monitoring Session
    # --------------------------------------------------------------------------
    print("\n[1] Initializing Clinical Monitoring Session (POST /api/v1/sessions):")
    sess_req = SessionCreateRequest(
        mode=ModeEnum.OFFLINE_DEMO,
        metadata={
            "patient_id": "PT-88340",
            "patient_age": 58,
            "patient_gender": "Female",
            "indication": "Post-infarct telemetry with ambulatory optical pulse check",
            "department": "Cardiology Observation Unit 3B",
        },
    )
    res_sess = client.post("/api/v1/sessions", json=sess_req.model_dump())
    assert res_sess.status_code in (200, 201), f"Session create failed: {res_sess.text}"
    sess_data = res_sess.json()
    session_id = sess_data["session_id"]
    print(f"    -> Session successfully initialized: {session_id}")
    print(f"    -> Patient ID: PT-88340 (Age 58, Female)")
    print(f"    -> Clinical Indication: Post-infarct telemetry observation")

    # --------------------------------------------------------------------------
    # 2. Ingest 30-Second Multimodal Telemetry Data
    # --------------------------------------------------------------------------
    print("\n[2] Ingesting 30-Second Multimodal Cardiac Telemetry Stream:")
    t_start = datetime.now(timezone.utc) - timedelta(seconds=30)

    # Sequence profile:
    # 0-14s: Normal Sinus Rhythm (72-75 BPM)
    # 15-20s: Premature Ventricular Contractions (PVC ectopy)
    # 21-26s: Transient Tachycardia (108-115 BPM)
    # 27-29s: Recovery back to Sinus Rhythm (80-84 BPM)
    measurements = []
    for i in range(30):
        t_cur = (t_start + timedelta(seconds=i)).isoformat()
        if i < 15:
            hr = 72.0 + (i % 4)
            pr = hr + 0.5
            rhythm = "Normal Sinus Rhythm"
            conf = 0.98
            sqi = 0.94
            qlabel = QualityLabelEnum.GOOD
            trend = TrendLabelEnum.STABLE
        elif 15 <= i <= 20:
            hr = 88.0 + (i % 3) * 2
            pr = hr - 6.0  # slight deficit during ectopy
            rhythm = "Premature Ventricular Contraction"
            conf = 0.91
            sqi = 0.86
            qlabel = QualityLabelEnum.FAIR
            trend = TrendLabelEnum.STABLE
        elif 21 <= i <= 26:
            hr = 108.0 + (i - 21) * 1.5
            pr = hr + 0.2
            rhythm = "Normal Sinus Rhythm"
            conf = 0.95
            sqi = 0.92
            qlabel = QualityLabelEnum.GOOD
            trend = TrendLabelEnum.INCREASING
        else:
            hr = 82.0 - (i - 27) * 1.0
            pr = hr
            rhythm = "Normal Sinus Rhythm"
            conf = 0.96
            sqi = 0.90
            qlabel = QualityLabelEnum.GOOD
            trend = TrendLabelEnum.DECREASING

        m = CanonicalMeasurement(
            timestamp=t_cur,
            mode=ModeEnum.OFFLINE_DEMO,
            heart_rate_bpm=round(hr, 1),
            pulse_rate_bpm=round(pr, 1),
            signal_quality=round(sqi, 3),
            quality_label=qlabel,
            rhythm_class=rhythm,
            rhythm_confidence=round(conf, 2),
            trend=trend,
            model_version="ecg-resnet-v1.4",
        )
        session_service.add_measurement(session_id, m)
        measurements.append(m)

    print(f"    -> Ingested {len(measurements)} canonical multimodal cardiac frames.")
    print(f"    -> Baseline rhythm: Normal Sinus Rhythm (72-76 BPM)")
    print(f"    -> Ectopic window: 6 PVC events at 15-20s")
    print(f"    -> Rate excursion: Transient Tachycardia peak at 115.5 BPM")

    # --------------------------------------------------------------------------
    # 3. Simulate Clinical Alerts Triggered During Session
    # --------------------------------------------------------------------------
    print("\n[3] Triggering Emergency Anomaly Alerts into Engine:")
    alert_tachy = CardiacAlert(
        alert_id="ALT-DEMO-001",
        severity=AlertSeverityEnum.WARNING,
        alert_type=AlertTypeEnum.TACHYCARDIA,
        message="Transient Tachycardia observed with peak rate 115.5 BPM",
        timestamp=(t_start + timedelta(seconds=23)).isoformat(),
        metrics={"peak_hr": 115.5, "threshold": 100.0},
    )
    alert_arrhy = CardiacAlert(
        alert_id="ALT-DEMO-002",
        severity=AlertSeverityEnum.CRITICAL,
        alert_type=AlertTypeEnum.ARRHYTHMIA_DETECTED,
        message="Premature Ventricular Contraction detected (91% confidence)",
        timestamp=(t_start + timedelta(seconds=17)).isoformat(),
        metrics={"confidence": 0.91, "rhythm": "Premature Ventricular Contraction"},
    )
    alert_service._alerts.append(alert_tachy)
    alert_service._alerts.append(alert_arrhy)
    print(f"    -> Alert ALT-DEMO-001 [WARNING]:  {alert_tachy.message}")
    print(f"    -> Alert ALT-DEMO-002 [CRITICAL]: {alert_arrhy.message}")

    # --------------------------------------------------------------------------
    # 4. Generate Structured Clinical Diagnostic Report (REST API)
    # --------------------------------------------------------------------------
    print("\n[4] Requesting Automated Diagnostic Summary (GET /api/v1/reports/{session_id}/summary):")
    res_rep = client.get(
        f"/api/v1/reports/{session_id}/summary?patient_notes=Telemetry%20window%20evaluated%20by%20On-Duty%20Cardiologist"
    )
    assert res_rep.status_code == 200, f"Report query failed: {res_rep.text}"
    rep = res_rep.json()

    print("\n" + "-" * 75)
    print("                    CLINICAL DIAGNOSTIC REPORT SUMMARY")
    print("-" * 75)
    print(f" Report ID:           {rep['report_id']}")
    print(f" Session ID:          {rep['session_id']}")
    print(f" Generation Time:     {rep['generated_at']}")
    print(f" Observation Window:  {rep['duration_seconds']:.1f} seconds ({rep['total_measurements']} samples)")
    print(f" Modality / Mode:     {rep['mode']}")
    print("-" * 75)
    print(" VITAL SIGNS & RATE ANALYSIS:")
    print(f"   • Mean Heart Rate:    {rep['mean_heart_rate_bpm']} BPM (Range: {rep['min_heart_rate_bpm']} - {rep['max_heart_rate_bpm']} BPM)")
    print(f"   • Mean Optical Pulse: {rep['mean_pulse_rate_bpm']} BPM (Range: {rep['min_pulse_rate_bpm']} - {rep['max_pulse_rate_bpm']} BPM)")
    print(f"   • Overall SQI Score:  {rep['overall_sqi']:.3f}")
    print(f"   • Quality Breakdown:  GOOD: {rep['quality_distribution'].get('GOOD', 0)}%, FAIR: {rep['quality_distribution'].get('FAIR', 0)}%, POOR: {rep['quality_distribution'].get('POOR', 0)}%")
    print("-" * 75)
    print(" RHYTHM & ARRHYTHMIA CLASSIFICATION:")
    for rhythm_type, cnt in rep["rhythm_distribution"].items():
        pct = (cnt / rep["total_measurements"]) * 100
        print(f"   • {rhythm_type:<35} : {cnt:>2} beats ({pct:>5.1f}%)")
    print("-" * 75)
    print(" EMERGENCY INCIDENTS & ANOMALY ALERTS:")
    print(f"   • Total Active Incidents:   {rep['total_alerts']}")
    print(f"   • Critical Alarms:          {rep['critical_alerts_count']}")
    warning_cnt = rep['total_alerts'] - rep['critical_alerts_count']
    print(f"   • Warning Notifications:    {warning_cnt}")
    print("-" * 75)
    print(" AUTOMATED CARDIOLOGY IMPRESSIONS:")
    for idx, impression in enumerate(rep["clinical_impressions"], 1):
        print(f"   {idx}. {impression}")
    print("-" * 75)

    # --------------------------------------------------------------------------
    # 5. Export Printable HTML Clinical Diagnostic Report
    # --------------------------------------------------------------------------
    print("\n[5] Fetching Printable HTML Clinical Diagnostic Report (GET /api/v1/reports/{session_id}/html):")
    res_html = client.get(
        f"/api/v1/reports/{session_id}/html?patient_notes=Ambulatory%20cardiac%20ward%20telemetry"
    )
    assert res_html.status_code == 200
    assert "text/html" in res_html.headers.get("content-type", "")
    html_content = res_html.text

    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"clinical_report_{session_id}.html"
    report_file.write_text(html_content, encoding="utf-8")

    print(f"    -> Exported printable HTML report to: {report_file.resolve()}")
    print(f"    -> HTML Report Size: {len(html_content):,} bytes")
    print(f"    -> Contains embedded responsive CSS, SVG diagnostic badges, and print media queries.")

    print("\n" + "=" * 80)
    print("  PHASE 12 DEMO COMPLETED SUCCESSFULLY WITH ZERO ERRORS.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
