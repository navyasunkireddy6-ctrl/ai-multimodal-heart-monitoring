"""Automated unit and integration tests for Phase 5 ML ECG Arrhythmia Classifier."""

from pathlib import Path
import numpy as np
import pytest

from ml.ecg.classifier import (
    ECGClassifier,
    ECGFeatureExtractor,
    ECGClassificationResult,
    FEATURE_NAMES,
    DEFAULT_MODEL_PATH,
    MODEL_VERSION,
)
from ml.ecg.dataset import ECGDatasetLoader
from backend.services.ecg_service import ECGService
from backend.models.schemas import ECGClassifyRequest, ECGClassifyResponse
from scripts.demo_ecg_processing import generate_synthetic_ecg


def test_feature_extractor_dimensions_and_keys():
    """Verify that feature extractor returns all canonical features as finite numbers."""
    extractor = ECGFeatureExtractor(sampling_rate_hz=360.0)
    _, sig = generate_synthetic_ecg(duration_sec=5.0, sampling_rate_hz=360.0, heart_rate_bpm=72.0, seed=1)

    features_dict, feature_vec = extractor.extract_features(sig)

    assert len(feature_vec) == len(FEATURE_NAMES)
    assert set(features_dict.keys()) == set(FEATURE_NAMES)
    for k, v in features_dict.items():
        assert np.isfinite(v), f"Feature {k} returned non-finite value: {v}"
    assert features_dict["heart_rate_bpm"] > 65.0 and features_dict["heart_rate_bpm"] < 80.0


def test_feature_extractor_custom_rr_intervals():
    """Verify feature extractor handles pre-computed RR intervals accurately."""
    extractor = ECGFeatureExtractor(sampling_rate_hz=360.0)
    sig = np.zeros(360 * 5)
    custom_rr = [800.0, 810.0, 790.0, 805.0]

    features_dict, _ = extractor.extract_features(sig, rr_intervals_ms=custom_rr)

    assert abs(features_dict["mean_rr_ms"] - 801.25) < 0.1
    assert features_dict["sdnn_ms"] > 0.0
    assert features_dict["rmssd_ms"] > 0.0


def test_classifier_loads_serialized_model():
    """Verify that ECGClassifier loads the trained model bundle from disk."""
    assert DEFAULT_MODEL_PATH.exists(), f"Expected trained model at {DEFAULT_MODEL_PATH}"
    classifier = ECGClassifier(model_path=DEFAULT_MODEL_PATH)
    assert classifier.model is not None
    assert classifier.model_version == MODEL_VERSION
    assert len(classifier.classes) >= 4


def test_classifier_rule_based_fallback():
    """Verify that an uninitialized classifier (no file) falls back gracefully."""
    dummy_path = Path("models/non_existent_model.joblib")
    classifier = ECGClassifier(model_path=dummy_path)
    assert classifier.model is None

    _, sig = generate_synthetic_ecg(duration_sec=5.0, sampling_rate_hz=360.0, heart_rate_bpm=72.0, seed=2)
    result = classifier.predict(sig, sampling_rate_hz=360.0)

    assert isinstance(result, ECGClassificationResult)
    assert result.rhythm_class in ["Normal Sinus Rhythm", "Sinus Tachycardia", "Sinus Bradycardia"]
    assert 0.0 <= result.rhythm_confidence <= 1.0


def test_classify_normal_sinus_rhythm_physionet():
    """Verify that pure NSR segment in PhysioNet Record 100 is classified as Normal Sinus Rhythm."""
    loader = ECGDatasetLoader()
    # At start_sec=12.0s to 18.0s, Record 100 contains purely normal sinus beats (N)
    sig, fs, _ = loader.load_record("100", start_sec=12.0, duration_sec=6.0)

    classifier = ECGClassifier(sampling_rate_hz=fs)
    result = classifier.predict(sig, sampling_rate_hz=fs)

    assert result.rhythm_class == "Normal Sinus Rhythm"
    assert result.rhythm_confidence >= 0.50
    assert result.model_version == MODEL_VERSION


def test_classify_atrial_premature_beat_physionet():
    """Verify that clinical Atrial Premature Beat in Record 100 (5.0s-10.0s) is detected."""
    loader = ECGDatasetLoader()
    # Record 100 at 5.0s contains annotated APB at sample 244
    sig, fs, _ = loader.load_record("100", start_sec=5.0, duration_sec=5.0)

    classifier = ECGClassifier(sampling_rate_hz=fs)
    result = classifier.predict(sig, sampling_rate_hz=fs)

    assert result.rhythm_class == "Atrial Premature Beat"
    assert result.rhythm_confidence >= 0.50
    assert result.model_version == MODEL_VERSION


def test_classify_pvc_detection():
    """Verify that simulated premature ventricular contraction is identified."""
    fs = 360.0
    _, sig = generate_synthetic_ecg(duration_sec=6.0, sampling_rate_hz=fs, heart_rate_bpm=72.0, noise_amplitude=0.02, seed=99)
    pvc_idx = int(fs * 2.5)
    sig[pvc_idx - 15 : pvc_idx + 15] += 2.0 * np.sin(np.linspace(0, np.pi, 30))

    classifier = ECGClassifier(sampling_rate_hz=fs)
    result = classifier.predict(sig, sampling_rate_hz=fs)

    assert result.rhythm_class == "Premature Ventricular Contraction"
    assert result.rhythm_confidence >= 0.50


def test_classify_paced_beat():
    """Verify that sharp pacing spikes are identified as Paced Beat."""
    fs = 360.0
    hr = 65.0
    _, sig = generate_synthetic_ecg(duration_sec=6.0, sampling_rate_hz=fs, heart_rate_bpm=hr, noise_amplitude=0.02, seed=101)
    for sp in np.arange(int(0.4 * fs), len(sig) - int(0.2 * fs), int((60.0 / hr) * fs)):
        if sp - 15 >= 0:
            sig[sp - 15] += 2.5

    classifier = ECGClassifier(sampling_rate_hz=fs)
    result = classifier.predict(sig, sampling_rate_hz=fs)

    assert result.rhythm_class == "Paced Beat"
    assert result.rhythm_confidence >= 0.50


def test_ecg_service_classify_window_canonical_contract():
    """Verify ECGService.classify_window produces schema-compliant ECGClassifyResponse."""
    service = ECGService(sampling_rate_hz=360.0)
    _, sig = generate_synthetic_ecg(duration_sec=5.0, sampling_rate_hz=360.0, heart_rate_bpm=75.0, seed=5)

    request = ECGClassifyRequest(
        sampling_rate_hz=360.0,
        signal=sig.tolist(),
    )
    response = service.classify_window(request)

    assert isinstance(response, ECGClassifyResponse)
    assert isinstance(response.rhythm_class, str)
    assert 0.0 <= response.rhythm_confidence <= 1.0
    assert response.model_version == MODEL_VERSION
    assert len(response.features) == len(FEATURE_NAMES)
    assert "Research prototype" in response.disclaimer


def test_short_and_empty_signals_safety():
    """Verify classifier does not crash when passed empty or short signals."""
    classifier = ECGClassifier(sampling_rate_hz=360.0)

    # Empty array
    res_empty = classifier.predict([], sampling_rate_hz=360.0)
    assert res_empty.rhythm_class is not None
    assert res_empty.rhythm_confidence >= 0.0

    # Very short array (10 samples)
    res_short = classifier.predict([0.1] * 10, sampling_rate_hz=360.0)
    assert res_short.rhythm_class is not None
    assert res_short.rhythm_confidence >= 0.0


def test_nan_and_infinite_signals_safety():
    """Verify classifier safely handles NaN and infinite inputs without raising errors."""
    classifier = ECGClassifier(sampling_rate_hz=360.0)
    corrupted_signal = [float("nan")] * 20 + [float("inf")] * 5 + [0.5] * 50

    result = classifier.predict(corrupted_signal, sampling_rate_hz=360.0)
    assert isinstance(result, ECGClassificationResult)
    assert isinstance(result.rhythm_class, str)
    assert 0.0 <= result.rhythm_confidence <= 1.0

    # Ensure all features are finite numbers
    for k, v in result.features.items():
        assert np.isfinite(v), f"Feature {k} returned non-finite value: {v}"

    # Verify dictionary serialization
    d = result.to_dict()
    assert d["model_version"] == MODEL_VERSION
    assert isinstance(d["features"], dict)

