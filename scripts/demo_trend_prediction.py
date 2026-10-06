"""CLI Demonstration of Phase 7: Short-Term Rate Trend Analysis & Trajectory Prediction.

Demonstrates:
1. Rising heart/pulse rate evaluation (INCREASING trajectory).
2. Resting rate evaluation with natural physiological variability (STABLE trajectory).
3. Post-exercise recovery / rate deceleration (DECREASING trajectory).
4. Canonical API contract roundtrip via TrendService conforming to docs/API_CONTRACT.md.

Run with:
    python scripts/demo_trend_prediction.py
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ml.trend.analyzer import TrendAnalyzer, TrendAnalysisResult, MODEL_VERSION
from backend.services.trend_service import TrendService
from backend.models.schemas import TrendPredictRequest, TrendPredictResponse


def run_demo():
    print("=" * 80)
    print("  PHASE 7: SHORT-TERM RATE TREND ANALYSIS & TRAJECTORY PREDICTION")
    print(f"  Model Version: {MODEL_VERSION}")
    print("  Research prototype — not intended for medical diagnosis.")
    print("=" * 80)

    analyzer = TrendAnalyzer(slope_threshold_bpm_per_min=3.0)
    base_time = datetime(2026, 9, 30, 12, 0, 0, tzinfo=timezone.utc)

    # --------------------------------------------------------------------------
    # 1. Rising Heart Rate Trajectory (Exercise Onset / Acceleration)
    # --------------------------------------------------------------------------
    print("\n[1] Evaluating Accelerating Rate History (Physical Exertion):")
    # Rate climbing from 70.0 to 86.0 BPM over 60 seconds (~16 BPM/min slope)
    times_inc = [(base_time + timedelta(seconds=i * 15)).isoformat() for i in range(5)]
    rates_inc = [70.0, 74.0, 78.0, 82.0, 86.0]

    res_inc: TrendAnalysisResult = analyzer.analyze(times_inc, rates_inc)
    print(f"    - Window Duration:       {res_inc.window_duration_seconds:.1f} seconds")
    print(f"    - Rate Sequence:         {rates_inc} BPM")
    print(f"    - Calculated Slope:      {res_inc.slope_bpm_per_min:+.2f} BPM/min")
    print(f"    - Predicted Trajectory:  '{res_inc.trend}'")
    print(f"    - Statistical Confidence:{res_inc.confidence * 100:.1f}%")

    # --------------------------------------------------------------------------
    # 2. Resting Baseline Stability
    # --------------------------------------------------------------------------
    print("\n[2] Evaluating Stable Resting Rate History:")
    # Normal resting fluctuations around 72 BPM
    times_stab = [(base_time + timedelta(seconds=i * 10)).isoformat() for i in range(6)]
    rates_stab = [72.0, 72.8, 71.5, 72.2, 71.9, 72.1]

    res_stab: TrendAnalysisResult = analyzer.analyze(times_stab, rates_stab)
    print(f"    - Window Duration:       {res_stab.window_duration_seconds:.1f} seconds")
    print(f"    - Rate Sequence:         {rates_stab} BPM")
    print(f"    - Calculated Slope:      {res_stab.slope_bpm_per_min:+.2f} BPM/min")
    print(f"    - Predicted Trajectory:  '{res_stab.trend}'")
    print(f"    - Statistical Confidence:{res_stab.confidence * 100:.1f}%")

    # --------------------------------------------------------------------------
    # 3. Post-Exercise Recovery / Deceleration
    # --------------------------------------------------------------------------
    print("\n[3] Evaluating Decelerating Rate History (Post-Exercise Recovery):")
    # Rate dropping from 125.0 to 95.0 BPM over 90 seconds (~-20 BPM/min)
    times_dec = [(base_time + timedelta(seconds=i * 15)).isoformat() for i in range(7)]
    rates_dec = [125.0, 120.0, 114.0, 109.0, 104.0, 99.0, 95.0]

    res_dec: TrendAnalysisResult = analyzer.analyze(times_dec, rates_dec)
    print(f"    - Window Duration:       {res_dec.window_duration_seconds:.1f} seconds")
    print(f"    - Rate Sequence:         {rates_dec} BPM")
    print(f"    - Calculated Slope:      {res_dec.slope_bpm_per_min:+.2f} BPM/min")
    print(f"    - Predicted Trajectory:  '{res_dec.trend}'")
    print(f"    - Statistical Confidence:{res_dec.confidence * 100:.1f}%")

    # --------------------------------------------------------------------------
    # 4. Canonical API Contract Roundtrip via TrendService
    # --------------------------------------------------------------------------
    print("\n[4] Testing Canonical API Contract via TrendService (POST /api/v1/trend/predict):")
    service = TrendService(slope_threshold_bpm_per_min=3.0)
    request = TrendPredictRequest(
        timestamps=times_inc,
        rates=rates_inc,
    )
    api_response: TrendPredictResponse = service.predict_trend(request)

    print(f"    - Response Trend:             '{api_response.trend}'")
    print(f"    - Response Slope (BPM/min):   {api_response.slope_bpm_per_min:+.2f}")
    print(f"    - Response Confidence:        {api_response.confidence:.4f}")
    print(f"    - Window Duration (sec):      {api_response.window_duration_seconds:.1f}")
    print(f"    - Mandatory Disclaimer:       '{api_response.disclaimer}'")

    print("\n" + "=" * 80)
    print("  PHASE 7 TREND PREDICTION DEMO COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
