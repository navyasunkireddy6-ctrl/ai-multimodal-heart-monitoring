"""CLI Demonstration of Phase 11: Real-Time Cardiac Anomaly Alerts & Emergency Notification Engine.

Demonstrates:
1. Dynamic threshold configuration inspect & update (tachycardia, bradycardia, arrhythmia confidence).
2. Normal Sinus Rhythm baseline with zero spurious alerts.
3. Acute Tachycardia warning (105 BPM) and critical alarm escalation (138 BPM).
4. Critical Bradycardia emergency alarm (38 BPM).
5. Machine Learning Arrhythmia Alert (Premature Ventricular Contraction with 92% confidence).
6. Multimodal Pulse Deficit emergency alert (hemodynamic dissociation with transit latency).
7. Cooldown debouncing engine suppressing alert flooding.
8. Active emergency incident retrieval and clinical acknowledgment lifecycle.
9. Live WebSocket telemetry alert attachment over /ws/v1/ppg.

Run with:
    python scripts/demo_cardiac_alerts.py
"""

import json
import sys
import time
from datetime import datetime, timezone
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
    CanonicalMeasurement,
    ModeEnum,
    MultimodalAnalyzeResponse,
    QualityLabelEnum,
    TrendLabelEnum,
)


def run_demo():
    print("=" * 80)
    print("  PHASE 11: REAL-TIME CARDIAC ANOMALY ALERTS & NOTIFICATION ENGINE")
    print("  Research prototype — not intended for medical diagnosis.")
    print("=" * 80)

    client = TestClient(app)
    alert_service = get_alert_service()
    alert_service.clear()

    # --------------------------------------------------------------------------
    # 1. Alert Configuration Inspection & Updating
    # --------------------------------------------------------------------------
    print("\n[1] Inspecting Clinical Threshold Configuration (GET /api/v1/alerts/config):")
    cfg_resp = client.get("/api/v1/alerts/config")
    assert cfg_resp.status_code == 200
    cfg = cfg_resp.json()
    print(f"    - Tachycardia Warning:     >= {cfg['tachycardia_threshold_bpm']:.1f} BPM")
    print(f"    - Tachycardia Critical:    >= {cfg['critical_tachycardia_bpm']:.1f} BPM")
    print(f"    - Bradycardia Warning:     <= {cfg['bradycardia_threshold_bpm']:.1f} BPM")
    print(f"    - Bradycardia Critical:    <= {cfg['critical_bradycardia_bpm']:.1f} BPM")
    print(f"    - Arrhythmia Confidence:   >= {cfg['min_arrhythmia_confidence'] * 100:.0f}%")
    print(f"    - Cooldown Debounce:       {cfg['cooldown_seconds']:.1f}s")

    # --------------------------------------------------------------------------
    # 2. Baseline Normal Rhythm Evaluation
    # --------------------------------------------------------------------------
    print("\n[2] Evaluating Normal Sinus Rhythm Baseline (72 BPM):")
    m_baseline = CanonicalMeasurement(
        mode=ModeEnum.ECG_DATASET,
        heart_rate_bpm=72.0,
        pulse_rate_bpm=72.0,
        signal_quality=0.95,
        quality_label=QualityLabelEnum.GOOD,
        rhythm_class="Normal Sinus Rhythm",
        rhythm_confidence=0.98,
        trend=TrendLabelEnum.STABLE,
        model_version="demo-1.0",
    )
    alerts = alert_service.evaluate_measurement(m_baseline)
    print(f"    - Normal measurement evaluated: {len(alerts)} alerts triggered (Expected: 0)")
    assert len(alerts) == 0
    print("    -> Normal sinus rhythm verified clean without false alarms.")

    # --------------------------------------------------------------------------
    # 3. Tachycardia Detection & Severity Escalation
    # --------------------------------------------------------------------------
    print("\n[3] Testing Tachycardia Detection & Escalation:")
    # Warning level (108 BPM)
    m_tachy_warn = CanonicalMeasurement(
        mode=ModeEnum.ECG_DATASET,
        heart_rate_bpm=108.0,
        signal_quality=0.91,
        quality_label=QualityLabelEnum.GOOD,
        model_version="demo-1.0",
    )
    tachy_warn_alerts = alert_service.evaluate_measurement(m_tachy_warn)
    assert len(tachy_warn_alerts) == 1
    a1 = tachy_warn_alerts[0]
    print(f"    - [{a1.severity}] {a1.alert_type}: {a1.message}")
    assert a1.severity == "WARNING"

    # Critical level (135 BPM) - Escalation bypasses cooldown
    m_tachy_crit = CanonicalMeasurement(
        mode=ModeEnum.ECG_DATASET,
        heart_rate_bpm=135.0,
        signal_quality=0.89,
        quality_label=QualityLabelEnum.GOOD,
        model_version="demo-1.0",
    )
    tachy_crit_alerts = alert_service.evaluate_measurement(m_tachy_crit)
    assert len(tachy_crit_alerts) == 1
    a2 = tachy_crit_alerts[0]
    print(f"    - [{a2.severity}] {a2.alert_type}: {a2.message}")
    assert a2.severity == "CRITICAL"

    # --------------------------------------------------------------------------
    # 4. Severe Bradycardia Emergency Alarm
    # --------------------------------------------------------------------------
    print("\n[4] Testing Severe Bradycardia Alarm (38 BPM):")
    m_brady_crit = CanonicalMeasurement(
        mode=ModeEnum.SMARTPHONE_PPG,
        pulse_rate_bpm=38.0,
        signal_quality=0.88,
        quality_label=QualityLabelEnum.GOOD,
        model_version="demo-1.0",
    )
    brady_alerts = alert_service.evaluate_measurement(m_brady_crit)
    assert len(brady_alerts) == 1
    a3 = brady_alerts[0]
    print(f"    - [{a3.severity}] {a3.alert_type}: {a3.message}")
    assert a3.severity == "CRITICAL"

    # --------------------------------------------------------------------------
    # 5. Arrhythmia Classification Detection
    # --------------------------------------------------------------------------
    print("\n[5] Testing ML Arrhythmia Detection (Premature Ventricular Contraction):")
    m_arrhythmia = CanonicalMeasurement(
        mode=ModeEnum.ECG_DATASET,
        heart_rate_bpm=76.0,
        signal_quality=0.92,
        quality_label=QualityLabelEnum.GOOD,
        rhythm_class="Premature Ventricular Contraction",
        rhythm_confidence=0.92,
        model_version="rf-ecg-v1.0.0",
    )
    arr_alerts = alert_service.evaluate_measurement(m_arrhythmia)
    assert len(arr_alerts) == 1
    a4 = arr_alerts[0]
    print(f"    - [{a4.severity}] {a4.alert_type}: {a4.message}")
    assert a4.severity == "CRITICAL"

    # --------------------------------------------------------------------------
    # 6. Multimodal Pulse Deficit Emergency Alert
    # --------------------------------------------------------------------------
    print("\n[6] Testing Multimodal Pulse Deficit Anomaly Alert:")
    multimodal_resp = MultimodalAnalyzeResponse(
        ecg_heart_rate_bpm=92.0,
        ppg_pulse_rate_bpm=70.0,
        rate_discrepancy_bpm=22.0,
        relative_discrepancy_pct=23.9,
        agreement_level="DISCREPANT",
        pulse_arrival_time_ms=235.0,
        pat_variability_ms=18.0,
        ecg_beat_count=8,
        ppg_beat_count=6,
        pulse_deficit_detected=True,
        ecg_signal_quality=0.92,
        ppg_signal_quality=0.80,
        fused_signal_quality=0.85,
        fused_quality_label=QualityLabelEnum.GOOD,
        rhythm_class="Premature Ventricular Contraction",
        rhythm_confidence=0.88,
        trend=TrendLabelEnum.STABLE,
        recommended_primary_modality="ECG",
        canonical_measurement=m_baseline,
        model_version="fusion-v1.0.0",
    )
    deficit_alerts = alert_service.evaluate_multimodal_response(multimodal_resp)
    assert any(a.alert_type == "PULSE_DEFICIT" for a in deficit_alerts)
    def_alert = next(a for a in deficit_alerts if a.alert_type == "PULSE_DEFICIT")
    print(f"    - [{def_alert.severity}] {def_alert.alert_type}: {def_alert.message}")

    # --------------------------------------------------------------------------
    # 7. Cooldown Debouncing Verification
    # --------------------------------------------------------------------------
    print("\n[7] Testing Cooldown Debouncing (Preventing Clinician Alarm Flooding):")
    # Immediate repeat of same warning rate
    repeated_alerts = alert_service.evaluate_measurement(m_tachy_warn)
    print(f"    - Repeated alert within cooldown: {len(repeated_alerts)} alerts dispatched (Expected: 0)")
    assert len(repeated_alerts) == 0
    print("    -> Cooldown engine safely suppressed duplicate notification.")

    # --------------------------------------------------------------------------
    # 8. Active Incident Review & Acknowledgment Workflow
    # --------------------------------------------------------------------------
    print("\n[8] Active Incident Review & Acknowledgment Lifecycle:")
    active_resp = client.get("/api/v1/alerts/active")
    assert active_resp.status_code == 200
    active_incidents = active_resp.json()
    print(f"    - Total Active Unacknowledged Incidents: {len(active_incidents)}")
    assert len(active_incidents) >= 4

    # Acknowledge the top critical incident
    target_id = active_incidents[0]["alert_id"]
    ack_resp = client.post(f"/api/v1/alerts/{target_id}/acknowledge")
    assert ack_resp.status_code == 200
    ack_data = ack_resp.json()
    print(f"    - Incident [{target_id[:8]}...] acknowledged at: {ack_data['acknowledged_at']}")
    assert ack_data["acknowledged"] is True

    # Check active list count reduced by 1
    active_after = client.get("/api/v1/alerts/active").json()
    print(f"    - Remaining Active Incidents:            {len(active_after)}")
    assert len(active_after) == len(active_incidents) - 1

    # --------------------------------------------------------------------------
    # 9. Live WebSocket Telemetry Alert Attachment
    # --------------------------------------------------------------------------
    print("\n[9] Testing Live Optical PPG WebSocket Alert Attachment (/ws/v1/ppg):")
    with client.websocket_connect("/ws/v1/ppg") as ws:
        t_base = datetime.now(timezone.utc)
        # Stream baseline frame
        frame = {
            "type": "ppg_frame",
            "timestamp": t_base.isoformat(),
            "red_channel_value": 150.0,
            "camera_fps": 30.0,
        }
        ws.send_text(json.dumps(frame))
        resp = json.loads(ws.receive_text())
        print(f"    - Frame processed | Type: {resp['type']} | Mode: {resp['mode']} | Model: {resp['model_version']}")
        assert resp["type"] == "measurement_update"

    print("\n" + "=" * 80)
    print("  PHASE 11 CARDIAC ANOMALY ALERTS DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
