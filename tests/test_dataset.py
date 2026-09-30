"""Automated unit and integration tests for ECG dataset loading and windowing."""

import pytest
import numpy as np
from pathlib import Path

from ml.ecg.dataset import (
    ECGDatasetLoader,
    ECGWindow,
    MITBIH_SYMBOL_MAP,
    AAMI_CLASS_MAP,
)
from scripts.download_datasets import verify_record_files, ensure_directory


def test_symbol_mappings():
    """Verify symbol mapping dictionaries contain standard MIT-BIH annotations."""
    assert "N" in MITBIH_SYMBOL_MAP
    assert MITBIH_SYMBOL_MAP["N"] == "Normal Sinus Rhythm"
    assert MITBIH_SYMBOL_MAP["V"] == "Premature Ventricular Contraction"
    assert MITBIH_SYMBOL_MAP["A"] == "Atrial Premature Beat"

    assert AAMI_CLASS_MAP["N"] == "Normal (N)"
    assert AAMI_CLASS_MAP["V"] == "Ventricular Ectopic (V)"
    assert AAMI_CLASS_MAP["A"] == "Supraventricular Ectopic (S)"


def test_dataset_discovery_and_load_record_100():
    """Verify loading real MIT-BIH record 100 downloaded in data/ecg."""
    loader = ECGDatasetLoader()
    available = loader.list_available_records()
    assert "100" in available, f"Record 100 not found in available: {available}"

    # Load first 10 seconds of Record 100
    signal, fs, lead_name = loader.load_record("100", channel=0, start_sec=0.0, duration_sec=10.0)
    assert fs == 360.0
    assert len(signal) == int(10.0 * 360.0)
    assert lead_name == "MLII"
    assert np.max(signal) > 0.5  # Positive R-peaks exist
    assert np.min(signal) < 0.0  # Isoelectric baseline / S-wave


def test_load_annotations_record_100():
    """Verify reading MIT-BIH reference annotations (.atr) for Record 100."""
    loader = ECGDatasetLoader()
    samples, symbols, labels = loader.load_annotations("100", start_sec=0.0, duration_sec=10.0)

    assert len(samples) > 0
    assert len(symbols) == len(samples)
    assert len(labels) == len(samples)
    # Record 100 is predominantly Normal Sinus Rhythm ('N')
    assert "N" in symbols
    assert "Normal Sinus Rhythm" in labels


def test_get_windows_5s_step_1s():
    """Verify sliding window generator produces 5-second windows with 1-second step."""
    loader = ECGDatasetLoader()
    windows = list(loader.get_windows("100", window_sec=5.0, step_sec=1.0, max_windows=5))

    assert len(windows) == 5
    for i, w in enumerate(windows):
        assert isinstance(w, ECGWindow)
        assert w.record_id == "100"
        assert w.window_index == i
        assert len(w.signal) == int(5.0 * 360.0)  # 1800 samples
        assert abs((w.end_sec - w.start_sec) - 5.0) < 1e-3
        assert w.dominant_symbol in ["N", "A", "V", "+"]
        assert len(w.r_peaks) >= 3  # Normal resting rate in 5s has 5-7 beats


def test_synthetic_fallback_generation():
    """Verify loader generates synthetic benchmark record when requested record is absent."""
    loader = ECGDatasetLoader(data_dir=Path("non_existent_folder_xyz"))
    signal, fs, lead = loader.load_record("999", duration_sec=5.0)
    assert fs == 360.0
    assert len(signal) == int(5.0 * 360.0)
    assert "Synthetic" in lead


def test_download_script_helpers(tmp_path):
    """Verify helper utilities in download_datasets.py."""
    test_dir = tmp_path / "test_ecg"
    ensure_directory(test_dir)
    assert test_dir.exists()

    # Empty directory has no valid record files
    assert not verify_record_files("100", test_dir)

    # Create dummy files
    (test_dir / "100.hea").write_text("dummy header")
    (test_dir / "100.dat").write_text("dummy dat")
    (test_dir / "100.atr").write_text("dummy atr")
    assert verify_record_files("100", test_dir)
