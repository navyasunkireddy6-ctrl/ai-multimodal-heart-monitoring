"""Train and serialize the Random Forest ECG Arrhythmia Classifier.

Extracts physiological HRV and morphological features from PhysioNet MIT-BIH records
and synthetic physiological variations to train model rf-ecg-v1.0.0.

Usage:
    python scripts/train_ecg_classifier.py [--output models/rf_ecg_classifier_v1.0.0.joblib]
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from ml.ecg.dataset import ECGDatasetLoader, MITBIH_SYMBOL_MAP
from ml.ecg.classifier import (
    ECGClassifier,
    ECGFeatureExtractor,
    FEATURE_NAMES,
    DEFAULT_MODEL_PATH,
    MODEL_VERSION,
)
from scripts.demo_ecg_processing import generate_synthetic_ecg


def generate_augmented_training_data(
    extractor: ECGFeatureExtractor,
    samples_per_class: int = 50
) -> Tuple[np.ndarray, List[str]]:
    """Synthesize physiological rhythm patterns to guarantee balanced class training."""
    X_aug = []
    y_aug = []
    fs = extractor.sampling_rate_hz

    # 1. Normal Sinus Rhythm (60 - 95 BPM)
    for i in range(samples_per_class):
        hr = float(np.random.uniform(62.0, 92.0))
        _, sig = generate_synthetic_ecg(duration_sec=5.0, sampling_rate_hz=fs, heart_rate_bpm=hr, noise_amplitude=0.03, seed=1000 + i)
        _, vec = extractor.extract_features(sig)
        X_aug.append(vec)
        y_aug.append("Normal Sinus Rhythm")

    # 2. Premature Ventricular Contraction (sudden premature wide QRS beat)
    for i in range(samples_per_class):
        base_hr = float(np.random.uniform(65.0, 85.0))
        _, sig = generate_synthetic_ecg(duration_sec=5.0, sampling_rate_hz=fs, heart_rate_bpm=base_hr, noise_amplitude=0.03, seed=2000 + i)
        # Inject simulated ectopic PVC spike (broad, inverted/high voltage)
        pvc_idx = int(fs * np.random.uniform(2.0, 3.2))
        sig[pvc_idx - 15 : pvc_idx + 15] += 1.8 * np.sin(np.linspace(0, np.pi, 30))
        _, vec = extractor.extract_features(sig)
        X_aug.append(vec)
        y_aug.append("Premature Ventricular Contraction")

    # 3. Atrial Premature Beat (early P-wave and compensatory pause)
    for i in range(samples_per_class):
        hr = float(np.random.uniform(68.0, 88.0))
        _, sig = generate_synthetic_ecg(duration_sec=5.0, sampling_rate_hz=fs, heart_rate_bpm=hr, noise_amplitude=0.04, seed=3000 + i)
        # Inject an early beat
        early_idx = int(fs * 2.3)
        sig[early_idx : early_idx + 20] += 0.9 * np.sin(np.linspace(0, np.pi, 20))
        _, vec = extractor.extract_features(sig)
        X_aug.append(vec)
        y_aug.append("Atrial Premature Beat")

    # 4. Paced Beat (sharp pacing spike preceding QRS)
    for i in range(samples_per_class):
        hr = float(np.random.uniform(60.0, 75.0))
        _, sig = generate_synthetic_ecg(duration_sec=5.0, sampling_rate_hz=fs, heart_rate_bpm=hr, noise_amplitude=0.02, seed=4000 + i)
        # Add sharp spike
        spike_indices = np.arange(int(0.4 * fs), len(sig) - int(0.2 * fs), int((60.0 / hr) * fs))
        for sp in spike_indices:
            if sp - 20 >= 0 and sp + 2 < len(sig):
                sig[sp - 15] += 2.0  # Narrow pacing spike
        _, vec = extractor.extract_features(sig)
        X_aug.append(vec)
        y_aug.append("Paced Beat")

    return np.array(X_aug, dtype=np.float64), y_aug


def extract_dataset_windows(
    loader: ECGDatasetLoader,
    extractor: ECGFeatureExtractor,
    max_windows_per_record: int = 150
) -> Tuple[np.ndarray, List[str]]:
    """Extract features from real PhysioNet MIT-BIH recordings."""
    records = loader.list_available_records()
    if not records:
        print("  Notice: No local PhysioNet records found; using synthetic dataset.")
        return np.empty((0, len(FEATURE_NAMES))), []

    X_list = []
    y_list = []

    for rec_id in records:
        print(f"  -> Extracting features from PhysioNet Record {rec_id}...")
        windows = loader.get_windows(rec_id, window_sec=5.0, step_sec=2.0)
        count = 0
        for win in windows:
            label = win.dominant_rhythm_class
            if label and label != "Unclassifiable Beat":
                _, vec = extractor.extract_features(win.signal)
                X_list.append(vec)
                y_list.append(label)
                count += 1
                if count >= max_windows_per_record:
                    break

    if X_list:
        return np.array(X_list, dtype=np.float64), y_list
    return np.empty((0, len(FEATURE_NAMES))), []


def main():
    parser = argparse.ArgumentParser(description="Train ECG Arrhythmia Classifier")
    parser.add_argument("--output", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to save trained joblib model")
    args = parser.parse_args()

    print("=" * 70)
    print("  PHASE 5: ML ECG ARRHYTHMIA CLASSIFIER TRAINING")
    print(f"  Target Model:   {MODEL_VERSION}")
    print(f"  Output Path:    {args.output}")
    print("=" * 70)

    extractor = ECGFeatureExtractor(sampling_rate_hz=360.0)
    loader = ECGDatasetLoader()

    # 1. Real dataset extraction
    X_real, y_real = extract_dataset_windows(loader, extractor)
    print(f"  Extracted {len(y_real)} windows from real PhysioNet records.")

    # 2. Balanced physiological augmentation
    X_aug, y_aug = generate_augmented_training_data(extractor, samples_per_class=60)
    print(f"  Generated {len(y_aug)} augmented physiological rhythm samples.")

    # Combine datasets
    if len(y_real) > 0:
        X = np.vstack([X_real, X_aug])
        y = y_real + y_aug
    else:
        X = X_aug
        y = y_aug

    print(f"\n  Total dataset size: {len(y)} samples across {len(set(y))} classes.")
    for cls in sorted(set(y)):
        print(f"    - {cls}: {y.count(cls)} samples")

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. Train Random Forest Classifier
    print("\n[Training Random Forest Classifier...]")
    classifier = ECGClassifier(model_path=Path(args.output))
    train_metrics = classifier.train(X_train, y_train)

    # 5. Evaluate on Holdout Test Set
    test_preds = classifier.model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"  Training Accuracy: {train_metrics['train_accuracy'] * 100:.2f}%")
    print(f"  Test Accuracy:     {test_acc * 100:.2f}%\n")
    print("---------------- Classification Report ----------------")
    print(classification_report(y_test, test_preds, digits=3))

    # 6. Save Model Artifact
    saved_path = classifier.save(Path(args.output))
    print(f"  Successfully serialized model bundle to: {saved_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
