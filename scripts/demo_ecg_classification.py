"""CLI Demonstration of Phase 5: Machine Learning ECG Arrhythmia Classification.

Demonstrates:
1. Physiological HRV & morphological feature extraction (mean_rr, sdnn, rmssd, kurtosis, etc.)
2. Real-time rhythm classification using trained Random Forest model (rf-ecg-v1.0.0)
3. Canonical API Contract roundtrip via ECGService

Run with:
    python scripts/demo_ecg_classification.py
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.ecg.classifier import ECGClassifier, ECGFeatureExtractor, FEATURE_NAMES, MODEL_VERSION
from ml.ecg.dataset import ECGDatasetLoader
from backend.services.ecg_service import ECGService
from backend.models.schemas import ECGClassifyRequest, ECGClassifyResponse
from scripts.demo_ecg_processing import generate_synthetic_ecg

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_demo():
    print("=" * 80)
    print("  PHASE 5: ML ECG ARRHYTHMIA CLASSIFICATION DEMONSTRATION")
    print(f"  Model Version: {MODEL_VERSION}")
    print("  Research prototype — not intended for medical diagnosis.")
    print("=" * 80)

    fs = 360.0
    classifier = ECGClassifier(sampling_rate_hz=fs)
    print(f"\n[1] Initializing Classifier:")
    print(f"    - Model Path:    {classifier.model_path}")
    print(f"    - Model Loaded:  {classifier.model is not None}")
    print(f"    - Target Classes: {len(classifier.classes)} ({', '.join(classifier.classes)})")
    print(f"    - Features:      {len(FEATURE_NAMES)} features ({', '.join(FEATURE_NAMES[:5])}...)")

    print("\n[2] Evaluating Rhythm Classification Across Diverse Waveforms:")
    print("-" * 80)
    print(f"{'RHYTHM TYPE':<25} | {'HR (BPM)':<8} | {'SDNN':<7} | {'RMSSD':<7} | {'PREDICTED CLASS':<24} | {'CONF':<6}")
    print("-" * 80)

    # 1. Normal Sinus Rhythm (Synthetic)
    _, sig_nsr = generate_synthetic_ecg(duration_sec=6.0, sampling_rate_hz=fs, heart_rate_bpm=74.0, noise_amplitude=0.02, seed=42)
    res_nsr = classifier.predict(sig_nsr, sampling_rate_hz=fs)
    hr_nsr = res_nsr.features.get("heart_rate_bpm", 74.0)
    sdnn_nsr = res_nsr.features.get("sdnn_ms", 0.0)
    rmssd_nsr = res_nsr.features.get("rmssd_ms", 0.0)
    print(f"{'Synthetic NSR (74 BPM)':<25} | {hr_nsr:>8.1f} | {sdnn_nsr:>7.1f} | {rmssd_nsr:>7.1f} | {res_nsr.rhythm_class:<24} | {res_nsr.rhythm_confidence * 100:>5.1f}%")

    # 2. Real PhysioNet Record 100 (5-second segment)
    loader = ECGDatasetLoader()
    sig_real, fs_real, _ = loader.load_record("100", start_sec=10.0, duration_sec=5.0)
    res_real = classifier.predict(sig_real, sampling_rate_hz=fs_real)
    hr_real = res_real.features.get("heart_rate_bpm", 75.0)
    sdnn_real = res_real.features.get("sdnn_ms", 0.0)
    rmssd_real = res_real.features.get("rmssd_ms", 0.0)
    print(f"{'PhysioNet Record 100':<25} | {hr_real:>8.1f} | {sdnn_real:>7.1f} | {rmssd_real:>7.1f} | {res_real.rhythm_class:<24} | {res_real.rhythm_confidence * 100:>5.1f}%")

    # 3. Simulated Premature Ventricular Contraction (Ectopic beat with high voltage)
    _, sig_pvc = generate_synthetic_ecg(duration_sec=6.0, sampling_rate_hz=fs, heart_rate_bpm=72.0, noise_amplitude=0.02, seed=99)
    pvc_idx = int(fs * 2.5)
    sig_pvc[pvc_idx - 15 : pvc_idx + 15] += 2.0 * np.sin(np.linspace(0, np.pi, 30))
    res_pvc = classifier.predict(sig_pvc, sampling_rate_hz=fs)
    hr_pvc = res_pvc.features.get("heart_rate_bpm", 72.0)
    sdnn_pvc = res_pvc.features.get("sdnn_ms", 0.0)
    rmssd_pvc = res_pvc.features.get("rmssd_ms", 0.0)
    print(f"{'Simulated Ectopic PVC':<25} | {hr_pvc:>8.1f} | {sdnn_pvc:>7.1f} | {rmssd_pvc:>7.1f} | {res_pvc.rhythm_class:<24} | {res_pvc.rhythm_confidence * 100:>5.1f}%")

    # 4. Simulated Paced Beat
    _, sig_paced = generate_synthetic_ecg(duration_sec=6.0, sampling_rate_hz=fs, heart_rate_bpm=65.0, noise_amplitude=0.02, seed=101)
    for sp in np.arange(int(0.4 * fs), len(sig_paced) - int(0.2 * fs), int((60.0 / 65.0) * fs)):
        if sp - 15 >= 0:
            sig_paced[sp - 15] += 2.5
    res_paced = classifier.predict(sig_paced, sampling_rate_hz=fs)
    hr_paced = res_paced.features.get("heart_rate_bpm", 65.0)
    sdnn_paced = res_paced.features.get("sdnn_ms", 0.0)
    rmssd_paced = res_paced.features.get("rmssd_ms", 0.0)
    print(f"{'Simulated Paced Beat':<25} | {hr_paced:>8.1f} | {sdnn_paced:>7.1f} | {rmssd_paced:>7.1f} | {res_paced.rhythm_class:<24} | {res_paced.rhythm_confidence * 100:>5.1f}%")
    print("-" * 80)

    # 3. Canonical Service Roundtrip Verification
    print("\n[3] Testing Canonical API Contract via ECGService:")
    service = ECGService(sampling_rate_hz=fs)
    request = ECGClassifyRequest(
        sampling_rate_hz=fs,
        signal=sig_nsr.tolist(),
    )
    api_response: ECGClassifyResponse = service.classify_window(request)
    print(f"    - Response Rhythm Class:      '{api_response.rhythm_class}'")
    print(f"    - Response Confidence:        {api_response.rhythm_confidence:.4f}")
    print(f"    - Model Version:              '{api_response.model_version}'")
    print(f"    - Extracted Features Count:   {len(api_response.features)}")
    print(f"    - Mandatory Disclaimer:       '{api_response.disclaimer}'")
    print("=" * 80)
    print("  PHASE 5 CLASSIFIER DEMO COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
