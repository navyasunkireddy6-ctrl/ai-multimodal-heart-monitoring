"""CLI Demonstration of Phase 6: Optical PPG Signal Processing and SQI Assessment.

Demonstrates:
1. Optical filtering, systolic peak detection, and pulse rate estimation at resting rate (72 BPM).
2. Tachycardic pulse rate estimation at 115 BPM.
3. Motion artifact detection, confirming POOR quality and pulse rate suppression.
4. Canonical API contract roundtrip via PPGService.

Run with:
    python scripts/demo_ppg_processing.py
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ml.ppg.processor import PPGProcessor, PPGAnalysisResult
from ml.ppg.synthetic import generate_synthetic_ppg
from backend.services.ppg_service import PPGService
from backend.models.schemas import PPGProcessRequest, PPGProcessResponse, PPGQualityRequest, PPGQualityResponse


def run_demo():
    print("=" * 80)
    print("  PHASE 6: SMARTPHONE & OPTICAL PPG SIGNAL PROCESSING DEMONSTRATION")
    print("  Mode: smartphone_ppg / synthetic_ppg")
    print("  Research prototype — not intended for medical diagnosis.")
    print("=" * 80)

    fs = 30.0  # Standard smartphone camera frame rate (30 FPS)
    processor = PPGProcessor(sampling_rate_hz=fs)

    # --------------------------------------------------------------------------
    # 1. Normal Resting Pulse (72 BPM)
    # --------------------------------------------------------------------------
    print("\n[1] Evaluating Clean Optical PPG Waveform (Target: 72.0 BPM):")
    t1, raw1 = generate_synthetic_ppg(
        duration_sec=10.0,
        sampling_rate_hz=fs,
        pulse_rate_bpm=72.0,
        noise_amplitude=0.015,
        seed=42,
    )
    res1: PPGAnalysisResult = processor.process(raw1)
    print(f"    - Duration:              10.0s ({len(raw1)} frames at 30 FPS)")
    print(f"    - Detected Systolic Peaks: {len(res1.peaks)} peaks")
    print(f"    - Estimated Pulse Rate:  {res1.pulse_rate_bpm} BPM (Target: 72.0 BPM)")
    print(f"    - Signal Quality Index:  {res1.signal_quality * 100:.1f}% ({res1.quality_label})")
    print(f"    - Optical SNR:           {res1.snr_db:.1f} dB")
    print(f"    - Usable for Vitals:     {res1.is_usable}")
    print(f"    - System Guidance:       '{res1.message}'")

    # --------------------------------------------------------------------------
    # 2. Elevated Pulse Rate (115 BPM)
    # --------------------------------------------------------------------------
    print("\n[2] Evaluating Elevated Pulse Rate PPG (Target: 115.0 BPM):")
    t2, raw2 = generate_synthetic_ppg(
        duration_sec=8.0,
        sampling_rate_hz=fs,
        pulse_rate_bpm=115.0,
        noise_amplitude=0.02,
        seed=101,
    )
    res2: PPGAnalysisResult = processor.process(raw2)
    print(f"    - Detected Systolic Peaks: {len(res2.peaks)} peaks")
    print(f"    - Estimated Pulse Rate:  {res2.pulse_rate_bpm} BPM (Target: 115.0 BPM)")
    print(f"    - Signal Quality Index:  {res2.signal_quality * 100:.1f}% ({res2.quality_label})")
    print(f"    - Optical SNR:           {res2.snr_db:.1f} dB")
    print(f"    - Usable for Vitals:     {res2.is_usable}")

    # --------------------------------------------------------------------------
    # 3. Motion-Corrupted Optical Waveform (POOR Quality)
    # --------------------------------------------------------------------------
    print("\n[3] Evaluating Motion-Corrupted / Poor Optical Contact PPG:")
    t3, raw3 = generate_synthetic_ppg(
        duration_sec=8.0,
        sampling_rate_hz=fs,
        pulse_rate_bpm=75.0,
        noise_amplitude=0.35,
        motion_amplitude=3.5,
        seed=999,
    )
    res3: PPGAnalysisResult = processor.process(raw3)
    print(f"    - Detected Peaks:        {len(res3.peaks)} peaks")
    print(f"    - Estimated Pulse Rate:  {res3.pulse_rate_bpm} (Suppressed for safety when POOR)")
    print(f"    - Signal Quality Index:  {res3.signal_quality * 100:.1f}% ({res3.quality_label})")
    print(f"    - Optical SNR:           {res3.snr_db:.1f} dB")
    print(f"    - Usable for Vitals:     {res3.is_usable}")
    print(f"    - Advisory Message:      '{res3.message}'")

    # --------------------------------------------------------------------------
    # 4. Canonical API Contract Roundtrip via PPGService
    # --------------------------------------------------------------------------
    print("\n[4] Testing Canonical API Contract via PPGService:")
    service = PPGService(sampling_rate_hz=fs)

    # 4A: POST /api/v1/ppg/process
    process_req = PPGProcessRequest(
        sampling_rate_hz=fs,
        signal=raw1.tolist(),
        source="smartphone_camera",
    )
    process_resp: PPGProcessResponse = service.process_window(process_req)
    print(f"    [POST /api/v1/ppg/process]")
    print(f"      - Pulse Rate:          {process_resp.pulse_rate_bpm} BPM")
    print(f"      - Signal Quality:      {process_resp.signal_quality:.4f}")
    print(f"      - Quality Label:       '{process_resp.quality_label}'")
    print(f"      - Extracted Peaks:     {len(process_resp.peaks)} peaks")
    print(f"      - Filtered Samples:    {len(process_resp.filtered_signal)} samples")

    # 4B: POST /api/v1/ppg/quality
    quality_req = PPGQualityRequest(
        sampling_rate_hz=fs,
        signal=raw3.tolist(),
    )
    quality_resp: PPGQualityResponse = service.assess_quality(quality_req)
    print(f"\n    [POST /api/v1/ppg/quality (Corrupted Window)]")
    print(f"      - Signal Quality:      {quality_resp.signal_quality:.4f}")
    print(f"      - Quality Label:       '{quality_resp.quality_label}'")
    print(f"      - Optical SNR (dB):    {quality_resp.snr_db}")
    print(f"      - Is Usable:           {quality_resp.is_usable}")
    print(f"      - Guidance Message:    '{quality_resp.message}'")

    print("\n" + "=" * 80)
    print("  PHASE 6 PPG PROCESSING DEMO COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
