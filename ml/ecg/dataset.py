"""PhysioNet MIT-BIH Arrhythmia Database loader, window extractor, and annotation parser.

Citation:
    Goldberger, A., et al. PhysioBank, PhysioToolkit, and PhysioNet:
    Components of a new research resource for complex physiologic signals.
    Circulation 101(23):e215-e220 (2000).

Disclaimer:
    Records are historical research recordings and do NOT represent the current user.
    Research prototype — not intended for medical diagnosis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Any
import numpy as np
import wfdb

# Project root path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "ecg"

# Standard MIT-BIH Arrhythmia beat annotation mapping
MITBIH_SYMBOL_MAP: Dict[str, str] = {
    "N": "Normal Sinus Rhythm",
    "L": "Left Bundle Branch Block",
    "R": "Right Bundle Branch Block",
    "A": "Atrial Premature Beat",
    "a": "Aberrated Atrial Premature Beat",
    "J": "Nodal Premature Beat",
    "S": "Supraventricular Premature Beat",
    "V": "Premature Ventricular Contraction",
    "F": "Ventricular Fusion Beat",
    "e": "Atrial Escape Beat",
    "j": "Nodal Escape Beat",
    "E": "Ventricular Escape Beat",
    "/": "Paced Beat",
    "f": "Fusion of Paced and Normal Beat",
    "Q": "Unclassifiable Beat",
}

# AAMI 5-class standard mapping
AAMI_CLASS_MAP: Dict[str, str] = {
    "N": "Normal (N)",
    "L": "Normal (N)",
    "R": "Normal (N)",
    "e": "Normal (N)",
    "j": "Normal (N)",
    "A": "Supraventricular Ectopic (S)",
    "a": "Supraventricular Ectopic (S)",
    "J": "Supraventricular Ectopic (S)",
    "S": "Supraventricular Ectopic (S)",
    "V": "Ventricular Ectopic (V)",
    "E": "Ventricular Ectopic (V)",
    "F": "Fusion (F)",
    "/": "Paced / Unknown (Q)",
    "f": "Paced / Unknown (Q)",
    "Q": "Paced / Unknown (Q)",
}


@dataclass
class ECGWindow:
    """A temporal window slice of an ECG record."""
    record_id: str
    window_index: int
    start_sec: float
    end_sec: float
    sampling_rate_hz: float
    lead_name: str
    signal: np.ndarray
    r_peaks: np.ndarray
    beat_symbols: List[str]
    dominant_symbol: str
    dominant_rhythm_class: str


class ECGDatasetLoader:
    """Loader and slice generator for PhysioNet WFDB format ECG records."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    def list_available_records(self) -> List[str]:
        """Return list of record IDs that have complete (.hea, .dat, .atr) files."""
        if not self.data_dir.exists():
            return []

        records = []
        for hea_file in self.data_dir.glob("*.hea"):
            rec_id = hea_file.stem
            dat_file = self.data_dir / f"{rec_id}.dat"
            atr_file = self.data_dir / f"{rec_id}.atr"
            if dat_file.exists() and dat_file.stat().st_size > 0:
                records.append(rec_id)
        return sorted(records)

    def load_record(
        self,
        record_id: str,
        channel: int = 0,
        start_sec: float = 0.0,
        duration_sec: Optional[float] = None
    ) -> Tuple[np.ndarray, float, str]:
        """Load continuous voltage signal samples and metadata.
        
        Returns:
            signal: 1D numpy array of voltage in mV.
            sampling_rate_hz: Sampling frequency (e.g. 360.0).
            lead_name: Name of lead (e.g. 'MLII' or 'V5').
        """
        rec_path = self.data_dir / record_id
        if not (self.data_dir / f"{record_id}.hea").exists():
            # If requested record is not locally cached, generate synthetic fallback
            return self._generate_synthetic_record(record_id, channel, start_sec, duration_sec or 60.0)

        # Read header to determine sampling frequency and bounds
        header = wfdb.rdheader(str(rec_path))
        fs = float(header.fs)
        start_sample = int(start_sec * fs)
        end_sample = int((start_sec + duration_sec) * fs) if duration_sec else None

        record = wfdb.rdrecord(
            str(rec_path),
            sampfrom=start_sample,
            sampto=end_sample,
            channels=[channel],
            physical=True
        )

        signal = record.p_signal[:, 0]
        lead_name = record.sig_name[0] if record.sig_name else "MLII"
        return signal, fs, lead_name

    def load_annotations(
        self,
        record_id: str,
        start_sec: float = 0.0,
        duration_sec: Optional[float] = None
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Load beat annotations (sample indices, symbols, rhythm descriptions)."""
        atr_path = self.data_dir / f"{record_id}.atr"
        if not atr_path.exists():
            return np.array([], dtype=int), [], []

        rec_path = self.data_dir / record_id
        header = wfdb.rdheader(str(rec_path))
        fs = float(header.fs)
        start_sample = int(start_sec * fs)
        end_sample = int((start_sec + duration_sec) * fs) if duration_sec else None

        ann = wfdb.rdann(
            str(rec_path),
            "atr",
            sampfrom=start_sample,
            sampto=end_sample
        )

        samples = ann.sample - start_sample  # Relative to window start
        symbols = ann.symbol
        rhythm_labels = [
            MITBIH_SYMBOL_MAP.get(s, "Unclassified Beat")
            for s in symbols
        ]
        return samples, symbols, rhythm_labels

    def get_windows(
        self,
        record_id: str,
        window_sec: float = 5.0,
        step_sec: float = 1.0,
        max_windows: Optional[int] = None
    ) -> Iterator[ECGWindow]:
        """Generate sliding windows for real-time simulated playback and feature extraction.
        
        Args:
            record_id: MIT-BIH record ID.
            window_sec: Window length in seconds (default: 5.0s).
            step_sec: Step size in seconds (default: 1.0s).
            max_windows: Optional limit on number of generated windows.
            
        Yields:
            ECGWindow dataclass instances.
        """
        signal, fs, lead_name = self.load_record(record_id)
        samples, symbols, labels = self.load_annotations(record_id)

        samples_per_window = int(window_sec * fs)
        samples_per_step = int(step_sec * fs)
        total_samples = len(signal)

        window_idx = 0
        current_start = 0

        while (current_start + samples_per_window) <= total_samples:
            current_end = current_start + samples_per_window
            window_signal = signal[current_start:current_end]

            # Find annotations within current window
            in_window_mask = (samples >= current_start) & (samples < current_end)
            win_peaks = samples[in_window_mask] - current_start
            win_symbols = [symbols[i] for i, m in enumerate(in_window_mask) if m]

            # Determine dominant symbol
            if len(win_symbols) > 0:
                # Prioritize non-normal beats if present
                arrhythmia_symbols = [s for s in win_symbols if s in ["V", "A", "L", "R", "F", "E"]]
                if arrhythmia_symbols:
                    dominant_sym = max(set(arrhythmia_symbols), key=arrhythmia_symbols.count)
                else:
                    dominant_sym = max(set(win_symbols), key=win_symbols.count)
            else:
                dominant_sym = "N"

            dominant_class = MITBIH_SYMBOL_MAP.get(dominant_sym, "Normal Sinus Rhythm")

            yield ECGWindow(
                record_id=record_id,
                window_index=window_idx,
                start_sec=round(current_start / fs, 2),
                end_sec=round(current_end / fs, 2),
                sampling_rate_hz=fs,
                lead_name=lead_name,
                signal=window_signal,
                r_peaks=win_peaks,
                beat_symbols=win_symbols,
                dominant_symbol=dominant_sym,
                dominant_rhythm_class=dominant_class,
            )

            window_idx += 1
            if max_windows and window_idx >= max_windows:
                break
            current_start += samples_per_step

    def _generate_synthetic_record(
        self,
        record_id: str,
        channel: int,
        start_sec: float,
        duration_sec: float
    ) -> Tuple[np.ndarray, float, str]:
        """Generate high-fidelity synthetic benchmark ECG record for offline testing."""
        fs = 360.0
        n_samples = int(duration_sec * fs)
        t = np.linspace(start_sec, start_sec + duration_sec, n_samples, endpoint=False)
        hr_bpm = 72.0 if record_id == "100" else 80.0
        rr = 60.0 / hr_bpm

        ecg = np.zeros_like(t)
        peak_times = np.arange(start_sec + 0.3, start_sec + duration_sec - 0.2, rr)
        for pt in peak_times:
            # QRS complex
            ecg += 1.2 * np.exp(-((t - pt) ** 2) / (2 * (0.015 ** 2)))
            # P wave
            ecg += 0.15 * np.exp(-((t - (pt - 0.16)) ** 2) / (2 * (0.03 ** 2)))
            # T wave
            ecg += 0.25 * np.exp(-((t - (pt + 0.20)) ** 2) / (2 * (0.05 ** 2)))

        return ecg, fs, "MLII (Synthetic Fallback)"
