"""Master Orchestrator: Executes All 12 Phases of BioQuest End-to-End.

Runs all 12 system phases in sequence:
  Phase 1:  Environment & Canonical API Contract
  Phase 2:  ECG Signal Processing & Pan-Tompkins R-Peak Detection
  Phase 3:  MIT-BIH Arrhythmia Dataset & Streaming Playback Engine
  Phase 4:  Real-Time Web ECG Telemetry Viewer & Playback API
  Phase 5:  Machine Learning ECG Arrhythmia Classification (ResNet/CNN/PhysioNet)
  Phase 6:  Smartphone Optical PPG Processing & Systolic Peak Detection
  Phase 7:  PPG Signal Quality Index (SQI) & Heart Rate Trend Prediction
  Phase 8:  FastAPI Server, Session Store & Real-Time WebSocket Telemetry
  Phase 9:  Multimodal Synchronization, Pulse Arrival Time (PAT) & Sensor Fusion
  Phase 10: Android PPG App Architecture & Mobile Streaming Integration
  Phase 11: Real-Time Cardiac Anomaly Alerts & Emergency Notification Engine
  Phase 12: Automated Clinical Diagnostic Report Generation & Analytics Export

Usage:
    python scripts/run_all_phases.py
"""

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class PhaseTask:
    phase_num: int
    name: str
    command: List[str]
    description: str


PHASES: List[PhaseTask] = [
    PhaseTask(
        phase_num=1,
        name="Environment & API Schema Contract",
        command=[sys.executable, "-m", "pytest", "tests/test_environment.py", "tests/test_schemas.py", "-q"],
        description="Validates Python 3.14 runtime, scientific dependencies, and canonical Pydantic v2 schemas.",
    ),
    PhaseTask(
        phase_num=2,
        name="ECG Signal Processing & R-Peaks",
        command=[sys.executable, "scripts/demo_ecg_processing.py"],
        description="Bandpass filtering, derivative, squaring, moving integration, and adaptive Pan-Tompkins thresholding.",
    ),
    PhaseTask(
        phase_num=3,
        name="MIT-BIH Dataset & Playback Engine",
        command=[sys.executable, "scripts/demo_simulated_playback.py"],
        description="Synthetic & real PhysioNet MIT-BIH record loader, streaming buffer generator, and rate estimator.",
    ),
    PhaseTask(
        phase_num=4,
        name="Real-Time ECG Telemetry & Playback API",
        command=[sys.executable, "-m", "pytest", "tests/test_ecg_playback.py", "-q"],
        description="Validates chunked streaming playback, looping telemetry, and WebSocket frame delivery.",
    ),
    PhaseTask(
        phase_num=5,
        name="ML ECG Arrhythmia Classification",
        command=[sys.executable, "scripts/demo_ecg_classification.py"],
        description="QRS morphological beat segmentation and multi-class arrhythmia inference (Normal, PVC, PAC, etc.).",
    ),
    PhaseTask(
        phase_num=6,
        name="Smartphone Optical PPG Processing",
        command=[sys.executable, "scripts/demo_ppg_processing.py"],
        description="Green-channel photometric extraction, bandpass filtering, systolic peak detection, and pulse rate.",
    ),
    PhaseTask(
        phase_num=7,
        name="PPG SQI & Rate Trend Prediction",
        command=[sys.executable, "scripts/demo_trend_prediction.py"],
        description="Statistical/skewness/perfusion SQI scoring and linear-regression rate trajectory forecasting.",
    ),
    PhaseTask(
        phase_num=8,
        name="FastAPI Server & WebSocket Streaming",
        command=[sys.executable, "scripts/demo_backend_api.py"],
        description="RESTful session management, health endpoints, and real-time bidirectional WebSocket streaming.",
    ),
    PhaseTask(
        phase_num=9,
        name="Multimodal Synchronization & Fusion (PAT)",
        command=[sys.executable, "scripts/demo_multimodal_fusion.py"],
        description="ECG R-peak to PPG systolic peak transit latency (PAT), pulse deficit detection, and fused analytics.",
    ),
    PhaseTask(
        phase_num=10,
        name="Android PPG Mobile Client Verification",
        command=[sys.executable, "scripts/verify_android_app.py"],
        description="Gradle project tree, Kotlin CameraX/WebSocket codebase, Manifest permissions, and client telemetry.",
    ),
    PhaseTask(
        phase_num=11,
        name="Real-Time Cardiac Anomaly Alerts Engine",
        command=[sys.executable, "scripts/demo_cardiac_alerts.py"],
        description="Configurable thresholds, tachycardia/bradycardia/arrhythmia/deficit rules, debouncing, and triage.",
    ),
    PhaseTask(
        phase_num=12,
        name="Clinical Diagnostic Report Generation & Export",
        command=[sys.executable, "scripts/demo_clinical_reporting.py"],
        description="Vital statistics aggregation, arrhythmia distributions, automated impressions, and HTML export.",
    ),
]


def run_all() -> int:
    print("=" * 85)
    print("      BIOQUEST MULTIMODAL CARDIAC MONITORING SYSTEM - MASTER VERIFICATION")
    print("      Executing All 12 Phases Sequentially (Phases 1 through 12)")
    print("=" * 85)
    print(f" Working Directory : {ROOT_DIR}")
    print(f" Python Interpreter: {sys.executable}")
    print(f" Total Phases      : {len(PHASES)}")
    print("=" * 85)

    results = []
    total_start = time.perf_counter()

    for task in PHASES:
        print(f"\n>> [PHASE {task.phase_num:02d}/12] {task.name.upper()}...")
        print(f"   Action: {task.description}")
        print(f"   Command: {' '.join(task.command)}")
        t0 = time.perf_counter()

        proc = subprocess.run(
            task.command,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.perf_counter() - t0

        if proc.returncode == 0:
            print(f"   Result: [PASSED] in {elapsed:.2f}s")
            results.append((task.phase_num, task.name, "PASSED", elapsed, None))
        else:
            print(f"   Result: [FAILED] in {elapsed:.2f}s (Exit code: {proc.returncode})")
            error_preview = (proc.stderr or proc.stdout)[-800:]
            print(f"   Error snippet:\n{error_preview}")
            results.append((task.phase_num, task.name, "FAILED", elapsed, error_preview))

    total_elapsed = time.perf_counter() - total_start

    # Final Scorecard
    print("\n" + "=" * 85)
    print("                     ALL 12 PHASES EXECUTION SCORECARD")
    print("=" * 85)
    print(f" {'PHASE':<8} | {'PHASE NAME':<45} | {'STATUS':<8} | {'RUNTIME':<8}")
    print("-" * 85)

    failures = 0
    for num, name, status, el, err in results:
        status_str = f"PASS" if status == "PASSED" else f"FAIL"
        print(f" Phase {num:02d} | {name:<45} | {status_str:<8} | {el:>6.2f}s")
        if status != "PASSED":
            failures += 1

    print("-" * 85)
    print(f" Total Elapsed Time: {total_elapsed:.2f} seconds")
    print(f" Passed: {len(results) - failures}/{len(results)} phases | Failed: {failures}/{len(results)} phases")
    print("=" * 85)

    if failures == 0:
        print("\n [SUCCESS] ALL 12 PHASES EXECUTED CLEANLY WITH ZERO PERSISTENT ERRORS!")
        return 0
    else:
        print(f"\n [ERROR] {failures} phase(s) failed during execution.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all())
