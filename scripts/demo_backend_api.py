"""CLI Demonstration of Phase 8: FastAPI Backend Endpoints and Live WebSocket Integration.

Demonstrates:
1. Health check endpoint returning service status and model versions.
2. ECG signal processing and ML arrhythmia rhythm classification over REST.
3. Optical PPG signal processing and signal quality index assessment over REST.
4. Short-term rate trend trajectory prediction over REST.
5. End-to-end Session Management lifecycle (Create, Add Measurement, Inspect, Export CSV, Delete).
6. Live Optical PPG streaming over bidirectional WebSocket (/ws/v1/ppg) with real-time feedback and SQI advisories.

Run with:
    python scripts/demo_backend_api.py
"""

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from backend.api.app import app
from ml.ppg.synthetic import generate_synthetic_ppg
from ml.ecg.dataset import ECGDatasetLoader


def run_demo():
    print("=" * 80)
    print("  PHASE 8: FASTAPI BACKEND API & LIVE WEBSOCKET INTEGRATION DEMO")
    print("  Research prototype — not intended for medical diagnosis.")
    print("=" * 80)

    client = TestClient(app)

    # --------------------------------------------------------------------------
    # 1. Health Check Endpoint (GET /api/v1/health)
    # --------------------------------------------------------------------------
    print("\n[1] Testing GET /api/v1/health:")
    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
    health_data = health_resp.json()
    print(f"    - Status:          {health_data['status']}")
    print(f"    - API Version:     {health_data['version']}")
    print(f"    - Models Loaded:   {health_data['models_loaded']}")
    print(f"    - Server Time:     {health_data['timestamp']}")

    # --------------------------------------------------------------------------
    # 2. ECG Processing & Rhythm Classification
    # --------------------------------------------------------------------------
    print("\n[2] Testing ECG Signal Processing & Rhythm Classification Endpoints:")
    loader = ECGDatasetLoader(data_dir=ROOT_DIR / "data" / "ecg")
    sig, fs, _ = loader.load_record("100", start_sec=0.0, duration_sec=5.0)
    sig_ecg = sig.tolist()

    # 2a. POST /api/v1/ecg/process
    ecg_proc_resp = client.post(
        "/api/v1/ecg/process",
        json={
            "sampling_rate_hz": 360.0,
            "signal": sig_ecg,
            "record_id": "100",
        },
    )
    assert ecg_proc_resp.status_code == 200
    ecg_proc_data = ecg_proc_resp.json()
    print(f"    - ECG Heart Rate:       {ecg_proc_data['heart_rate_bpm']} BPM")
    print(f"    - ECG Signal Quality:   {ecg_proc_data['signal_quality']:.3f} ({ecg_proc_data['quality_label']})")
    print(f"    - Detected R-Peaks:     {len(ecg_proc_data['r_peaks'])} peaks")

    # 2b. POST /api/v1/ecg/classify
    ecg_clf_resp = client.post(
        "/api/v1/ecg/classify",
        json={
            "sampling_rate_hz": 360.0,
            "signal": sig_ecg,
            "rr_intervals_ms": ecg_proc_data["rr_intervals_ms"],
        },
    )
    assert ecg_clf_resp.status_code == 200
    ecg_clf_data = ecg_clf_resp.json()
    print(f"    - Predicted Rhythm:     {ecg_clf_data['rhythm_class']}")
    print(f"    - Rhythm Confidence:    {ecg_clf_data['rhythm_confidence']:.2%}")
    print(f"    - Active Model:         {ecg_clf_data['model_version']}")

    # --------------------------------------------------------------------------
    # 3. Optical PPG Processing & Signal Quality Index
    # --------------------------------------------------------------------------
    print("\n[3] Testing Optical PPG Processing & Quality Assessment Endpoints:")
    _, clean_ppg = generate_synthetic_ppg(duration_sec=6.0, sampling_rate_hz=30.0, pulse_rate_bpm=75.0, noise_amplitude=0.02)
    ppg_signal_list = clean_ppg.tolist()

    # 3a. POST /api/v1/ppg/process
    ppg_proc_resp = client.post(
        "/api/v1/ppg/process",
        json={
            "sampling_rate_hz": 30.0,
            "signal": ppg_signal_list,
            "source": "smartphone_camera",
        },
    )
    assert ppg_proc_resp.status_code == 200
    ppg_proc_data = ppg_proc_resp.json()
    print(f"    - Estimated Pulse Rate: {ppg_proc_data['pulse_rate_bpm']} BPM")
    print(f"    - PPG Signal Quality:   {ppg_proc_data['signal_quality']:.3f} ({ppg_proc_data['quality_label']})")
    print(f"    - Detected Systolic:    {len(ppg_proc_data['peaks'])} peaks")

    # 3b. POST /api/v1/ppg/quality
    ppg_qual_resp = client.post(
        "/api/v1/ppg/quality",
        json={"sampling_rate_hz": 30.0, "signal": ppg_signal_list},
    )
    assert ppg_qual_resp.status_code == 200
    ppg_qual_data = ppg_qual_resp.json()
    print(f"    - Usable for Pulse:     {ppg_qual_data['is_usable']} (SNR: {ppg_qual_data['snr_db']:.1f} dB)")
    print(f"    - Quality Guidance:     {ppg_qual_data['message']}")

    # --------------------------------------------------------------------------
    # 4. Short-Term Rate Trend Prediction
    # --------------------------------------------------------------------------
    print("\n[4] Testing Rate Trend Prediction Endpoint (POST /api/v1/trend/predict):")
    base_t = datetime(2026, 9, 30, 12, 0, 0, tzinfo=timezone.utc)
    rising_times = [(base_t + timedelta(seconds=i * 15)).isoformat() for i in range(5)]
    rising_rates = [70.0, 74.0, 78.0, 82.0, 86.0]

    trend_resp = client.post(
        "/api/v1/trend/predict",
        json={"timestamps": rising_times, "rates": rising_rates},
    )
    assert trend_resp.status_code == 200
    trend_data = trend_resp.json()
    print(f"    - Evaluated Trend:      {trend_data['trend']}")
    print(f"    - Rate Slope:           {trend_data['slope_bpm_per_min']:+.2f} BPM/min")
    print(f"    - Trajectory Confidence:{trend_data['confidence']:.2%}")

    # --------------------------------------------------------------------------
    # 5. Session Management Lifecycle
    # --------------------------------------------------------------------------
    print("\n[5] Testing Session Management Lifecycle Endpoints:")
    # 5a. Create Session
    create_resp = client.post(
        "/api/v1/sessions",
        json={"mode": "smartphone_ppg", "notes": "Phase 8 demonstration monitoring run"},
    )
    assert create_resp.status_code == 201
    sess_id = create_resp.json()["session_id"]
    print(f"    - Created Session ID:   {sess_id}")

    # 5b. Record Canonical Measurement
    meas_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "smartphone_ppg",
        "heart_rate_bpm": None,
        "pulse_rate_bpm": 74.8,
        "signal_quality": 0.93,
        "quality_label": "GOOD",
        "rhythm_class": None,
        "rhythm_confidence": None,
        "trend": "STABLE",
        "model_version": "ppg-peak-v1.0.0",
    }
    meas_resp = client.post(f"/api/v1/sessions/{sess_id}/measurements", json=meas_payload)
    assert meas_resp.status_code == 200
    print(f"    - Recorded Measurement: Pulse Rate {meas_payload['pulse_rate_bpm']} BPM")

    # 5c. Inspect Session Detail
    detail_resp = client.get(f"/api/v1/sessions/{sess_id}")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    print(f"    - Session Total Count:  {len(detail_data['measurements'])} measurement(s)")
    print(f"    - Session Mode:         {detail_data['mode']}")

    # 5d. Export CSV
    export_resp = client.get(f"/api/v1/sessions/{sess_id}/export")
    assert export_resp.status_code == 200
    assert "pulse_rate_bpm" in export_resp.text
    print(f"    - CSV Export Length:    {len(export_resp.text)} bytes")

    # 5e. Delete Session
    del_resp = client.delete(f"/api/v1/sessions/{sess_id}")
    assert del_resp.status_code == 200
    print(f"    - Deleted Session:      {del_resp.json()['deleted']}")

    # --------------------------------------------------------------------------
    # 6. Live Optical PPG WebSocket Streaming (/ws/v1/ppg)
    # --------------------------------------------------------------------------
    print("\n[6] Testing Live Optical PPG WebSocket Stream (/ws/v1/ppg):")
    with client.websocket_connect("/ws/v1/ppg") as ws:
        # Create a session to bind the stream
        ws_sess = client.post("/api/v1/sessions", json={"mode": "smartphone_ppg"}).json()
        ws_sess_id = ws_sess["session_id"]

        # Stream 60 frames of synthetic optical pulse
        fps = 30.0
        t_now = datetime.now(timezone.utc)
        last_resp = None

        print(f"    - Streaming 60 optical frames into session {ws_sess_id}...")
        for i in range(60):
            val = float(clean_ppg[i % len(clean_ppg)])
            frame = {
                "type": "ppg_frame",
                "timestamp": (t_now + timedelta(seconds=i / fps)).isoformat(),
                "session_id": ws_sess_id,
                "red_channel_value": val,
                "green_channel_value": val * 0.6,
                "blue_channel_value": val * 0.4,
                "flash_enabled": True,
                "camera_fps": fps,
            }
            ws.send_text(json.dumps(frame))
            resp_txt = ws.receive_text()
            last_resp = json.loads(resp_txt)

        print(f"    - Received Live Response: Type={last_resp['type']}")
        print(f"    - Live Quality Label:    {last_resp['quality_label']} (SQI: {last_resp['signal_quality']:.3f})")
        print(f"    - Live Pulse Rate:       {last_resp['pulse_rate_bpm']} BPM")
        print(f"    - Live Warning:          {last_resp.get('warning')}")

        # Stream poor quality noise frame to test SQI degradation advisory
        print("\n    - Testing Degraded Optical Contact (Noise Injection)...")
        for i in range(50):
            noise_val = 120.0 + float((i % 7) * 0.05)  # almost flat, poor optical pulsatile variation
            noise_frame = {
                "type": "ppg_frame",
                "timestamp": (t_now + timedelta(seconds=(60 + i) / fps)).isoformat(),
                "session_id": ws_sess_id,
                "red_channel_value": noise_val,
                "camera_fps": fps,
            }
            ws.send_text(json.dumps(noise_frame))
            poor_resp = json.loads(ws.receive_text())

        print(f"    - Degraded Quality Label: {poor_resp['quality_label']}")
        print(f"    - Degraded Pulse Rate:    {poor_resp['pulse_rate_bpm']} (must be null for POOR)")
        print(f"    - Advisory Notice:        '{poor_resp.get('warning')}'")

    print("\n" + "=" * 80)
    print("  PHASE 8 BACKEND API & WEBSOCKET DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
