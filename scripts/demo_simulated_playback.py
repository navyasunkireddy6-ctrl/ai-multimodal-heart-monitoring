"""CLI Demonstration of Simulated Real-Time ECG Playback.

Run with:
    python scripts/demo_simulated_playback.py
"""

import sys
from pathlib import Path

# Ensure root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.ecg.playback import ECGPlaybackEngine, PlaybackState


def run_playback_demo():
    print("=" * 75)
    print("  SIMULATED REAL-TIME ECG PLAYBACK DEMONSTRATION")
    print("  Mode: ECG Dataset / Simulated Real-Time Mode")
    print("  Research prototype — not intended for medical diagnosis.")
    print("=" * 75)

    record_id = "100"
    print(f"\n[1] Initializing Playback Engine:")
    print(f"    - Record:          MIT-BIH Record {record_id}")
    print(f"    - Window Length:   5.0 seconds (1800 samples at 360 Hz)")
    print(f"    - Step Size:       1.0 second (360 samples step)")
    print(f"    - Playback Speed:  2.0x (Accelerated demonstration)")

    engine = ECGPlaybackEngine(
        record_id=record_id,
        window_sec=5.0,
        step_sec=1.0,
        playback_speed=2.0
    )

    print(f"    - Initial State:   {engine.state.value}")

    print("\n[2] Starting Real-Time Playback Stream:")
    engine.start()
    print(f"    - State changed to: {engine.state.value}")

    print("\n" + "-" * 75)
    print(f" {'TIME':8s} | {'HR (BPM)':8s} | {'SQI':5s} | {'QUALITY':7s} | {'RHYTHM CLASS':25s} | {'PEAKS'}")
    print("-" * 75)

    # Stream 6 consecutive frames
    for i in range(6):
        frame = engine.next_frame()
        if not frame:
            break
        t_range = f"{frame.playback_time_sec:4.1f}s-{frame.playback_time_sec + 5.0:4.1f}s"
        hr_str = f"{frame.heart_rate_bpm:.1f}" if frame.heart_rate_bpm else "N/A"
        peaks_str = f"{len(frame.r_peaks)} peaks"
        print(f" {t_range:8s} | {hr_str:>8s} | {frame.signal_quality:5.3f} | {frame.quality_label:7s} | {frame.predicted_rhythm_class:25s} | {peaks_str}")

    print("-" * 75)
    print(f"  Frame Mode Label: '{frame.mode_label}'")

    print("\n[3] Testing Playback State Controls:")
    print(f"  --> Pausing playback...")
    engine.pause()
    print(f"      State: {engine.state.value} (next_frame returns None when paused: {engine.next_frame() is None})")

    print(f"  --> Resuming playback...")
    engine.resume()
    frame_resumed = engine.next_frame()
    print(f"      Resumed at time {frame_resumed.playback_time_sec}s with HR = {frame_resumed.heart_rate_bpm} BPM")

    print(f"  --> Resetting position...")
    engine.reset()
    frame_reset = engine.next_frame()
    print(f"      Position reset to time {frame_reset.playback_time_sec}s")

    print(f"  --> Stopping playback...")
    engine.stop()
    print(f"      Final State: {engine.state.value}")
    print("=" * 75)


if __name__ == "__main__":
    run_playback_demo()
