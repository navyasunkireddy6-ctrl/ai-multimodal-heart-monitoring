"""CLI Demonstration of Phase 9: Multimodal Synchronization, Transit Time & Rate Fusion.

Demonstrates:
1. Dual-rate simultaneous comparison: ECG Heart Rate vs Optical PPG Pulse Rate.
2. Beat-to-beat Pulse Arrival Time (PAT / PTT) vascular latency estimation.
3. Pulse Deficit detection (cardiac electrical events failing to generate peripheral pulses).
4. Fused Signal Quality Index (SQI) and sensor failover under degraded contact.
5. REST API evaluation over /api/v1/multimodal/analyze and /api/v1/multimodal/demo-frame.

Run with:
    python scripts/demo_multimodal_fusion.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from backend.api.app import app
from ml.ecg.dataset import ECGDatasetLoader
from ml.ecg.processor import ECGProcessor
from ml.ppg.processor import PPGProcessor
from ml.ppg.synthetic import generate_synthetic_ppg
from ml.fusion.synchronizer import MultimodalSynchronizer, MultimodalSyncResult


def run_demo():
    print("=" * 80)
    print("  PHASE 9: MULTIMODAL SYNCHRONIZATION, TRANSIT TIME & SIGNAL FUSION")
    print("  Research prototype — not intended for medical diagnosis.")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. Dual-Rate Synchronization & Pulse Arrival Time (PAT)
    # --------------------------------------------------------------------------
    print("\n[1] Synchronizing Simultaneous ECG and Optical PPG Signals:")
    loader = ECGDatasetLoader(data_dir=ROOT_DIR / "data" / "ecg")
    sig_ecg, fs_ecg, lead = loader.load_record("100", start_sec=0.0, duration_sec=5.0)

    ecg_proc = ECGProcessor(sampling_rate_hz=fs_ecg)
    ecg_res = ecg_proc.process(sig_ecg)

    # Synthesize corresponding optical PPG waveform matching the heart rate with physiological PAT delay
    target_bpm = ecg_res.heart_rate_bpm or 75.0
    r_peaks_sec = ecg_res.r_peaks / float(fs_ecg) if len(ecg_res.r_peaks) > 0 else None
    _, ppg_sig = generate_synthetic_ppg(
        duration_sec=5.0,
        sampling_rate_hz=30.0,
        pulse_rate_bpm=target_bpm,
        noise_amplitude=0.01,
        r_peaks_sec=r_peaks_sec,
        pat_delay_sec=0.22,
    )
    ppg_proc = PPGProcessor(sampling_rate_hz=30.0)
    ppg_res = ppg_proc.process(ppg_sig)

    synchronizer = MultimodalSynchronizer()
    sync_res: MultimodalSyncResult = synchronizer.synchronize(
        ecg_result=ecg_res,
        ppg_result=ppg_res,
        ecg_sampling_rate_hz=fs_ecg,
        ppg_sampling_rate_hz=30.0,
    )

    print(f"    - ECG Heart Rate:        {sync_res.ecg_heart_rate_bpm} BPM (SQI: {sync_res.ecg_signal_quality:.3f})")
    print(f"    - PPG Pulse Rate:        {sync_res.ppg_pulse_rate_bpm} BPM (SQI: {sync_res.ppg_signal_quality:.3f})")
    print(f"    - Rate Discrepancy (Δ):  {sync_res.rate_discrepancy_bpm} BPM (Rel: {sync_res.relative_discrepancy_pct}%)")
    print(f"    - Agreement Level:       {sync_res.agreement_level}")
    print(f"    - Pulse Arrival Time:    {sync_res.pulse_arrival_time_ms} ms (Variability: ±{sync_res.pat_variability_ms} ms)")
    print(f"    - Fused Signal Quality:  {sync_res.fused_signal_quality:.3f} ({sync_res.fused_quality_label})")
    print(f"    - Primary Modality:      {sync_res.recommended_primary_modality}")

    # --------------------------------------------------------------------------
    # 2. Pulse Deficit Anomaly Detection
    # --------------------------------------------------------------------------
    print("\n[2] Testing Pulse Deficit Detection (Missing Peripheral Pulses):")
    # Simulate condition where ventricular beats fail to produce optical pulsations
    _, short_ppg = generate_synthetic_ppg(
        duration_sec=5.0,
        sampling_rate_hz=30.0,
        pulse_rate_bpm=45.0,  # Far fewer peripheral pulses
    )
    ppg_deficit_res = ppg_proc.process(short_ppg)

    sync_deficit = synchronizer.synchronize(
        ecg_result=ecg_res,
        ppg_result=ppg_deficit_res,
        ecg_sampling_rate_hz=fs_ecg,
        ppg_sampling_rate_hz=30.0,
    )
    print(f"    - ECG Beats:             {sync_deficit.ecg_beat_count}")
    print(f"    - PPG Beats:             {sync_deficit.ppg_beat_count}")
    print(f"    - Deficit Detected:      {sync_deficit.pulse_deficit_detected}")
    print(f"    - Agreement Level:       {sync_deficit.agreement_level}")

    # --------------------------------------------------------------------------
    # 3. Degraded Optical Contact & Multimodal Failover
    # --------------------------------------------------------------------------
    print("\n[3] Testing Sensor Failover & Dynamic Reliability Weighting:")
    noisy_ppg = [128.0 + (i % 5) * 0.1 for i in range(150)]
    ppg_noisy_res = ppg_proc.process(noisy_ppg)

    sync_failover = synchronizer.synchronize(
        ecg_result=ecg_res,
        ppg_result=ppg_noisy_res,
        ecg_sampling_rate_hz=fs_ecg,
        ppg_sampling_rate_hz=30.0,
    )
    print(f"    - Degraded Optical SQI:  {sync_failover.ppg_signal_quality:.3f}")
    print(f"    - High-Confidence ECG:   {sync_failover.ecg_signal_quality:.3f}")
    print(f"    - Fused Reliability:     {sync_failover.fused_signal_quality:.3f} ({sync_failover.fused_quality_label})")
    print(f"    - Failover Selection:    {sync_failover.recommended_primary_modality} (relying on ECG)")

    # --------------------------------------------------------------------------
    # 4. REST API Endpoint: POST /api/v1/multimodal/analyze
    # --------------------------------------------------------------------------
    print("\n[4] Testing REST API /api/v1/multimodal/analyze:")
    client = TestClient(app)
    req_body = {
        "ecg_signal": sig_ecg.tolist(),
        "ecg_sampling_rate_hz": fs_ecg,
        "ppg_signal": ppg_sig.tolist(),
        "ppg_sampling_rate_hz": 30.0,
        "record_id": "100",
    }
    resp = client.post("/api/v1/multimodal/analyze", json=req_body)
    assert resp.status_code == 200, f"Error: {resp.text}"
    api_data = resp.json()

    print(f"    - Status:                200 OK")
    print(f"    - ECG Rate:              {api_data['ecg_heart_rate_bpm']} BPM")
    print(f"    - PPG Rate:              {api_data['ppg_pulse_rate_bpm']} BPM")
    print(f"    - Rhythm Class:          {api_data['rhythm_class']} (Confidence: {api_data['rhythm_confidence']:.2%})")
    print(f"    - Agreement Level:       {api_data['agreement_level']}")
    print(f"    - Pulse Arrival Time:    {api_data['pulse_arrival_time_ms']} ms")
    print(f"    - Canonical Schema:      Mode={api_data['canonical_measurement']['mode']}, Trend={api_data['canonical_measurement']['trend']}")

    # --------------------------------------------------------------------------
    # 5. REST API Demo Frame: GET /api/v1/multimodal/demo-frame
    # --------------------------------------------------------------------------
    print("\n[5] Testing Real-Time Dashboard Frame API (GET /api/v1/multimodal/demo-frame):")
    frame_resp = client.get("/api/v1/multimodal/demo-frame?start_sec=0.0&duration_sec=5.0&record_id=100")
    assert frame_resp.status_code == 200
    frame_json = frame_resp.json()

    print(f"    - Frame Mode:            {frame_json['mode']}")
    print(f"    - ECG Lead:              {frame_json['ecg']['lead_name']} ({len(frame_json['ecg']['filtered_waveform'])} samples)")
    print(f"    - PPG Waveform:          {len(frame_json['ppg']['filtered_waveform'])} optical samples")
    print(f"    - Fused Agreement:       {frame_json['fusion']['agreement_level']}")

    print("\n" + "=" * 80)
    print("  PHASE 9 MULTIMODAL SYNCHRONIZATION & FUSION DEMO COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
