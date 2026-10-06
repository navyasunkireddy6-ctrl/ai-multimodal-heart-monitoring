"""Simulated Real-Time ECG Playback Engine.

Replays historical ECG dataset records in a streaming windowed fashion (5s window, 1s step)
with state controls (start, pause, stop, reset, playback speed).

Label:
    "ECG Dataset / Simulated Real-Time Mode"
"""

import time
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, AsyncIterator

import numpy as np

from ml.ecg.processor import ECGProcessor, ECGAnalysisResult
from ml.ecg.dataset import ECGDatasetLoader, ECGWindow
from ml.ecg.classifier import ECGClassifier, ECGClassificationResult


class PlaybackState(str, Enum):
    """Playback engine state machine states."""
    STOPPED = "STOPPED"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"


@dataclass
class ECGPlaybackFrame:
    """A single real-time simulated update frame."""
    timestamp: str
    mode: str = "ecg_dataset"
    mode_label: str = "ECG Dataset / Simulated Real-Time Mode"
    record_id: str = "100"
    window_index: int = 0
    playback_time_sec: float = 0.0
    sampling_rate_hz: float = 360.0
    lead_name: str = "MLII"
    raw_waveform: List[float] = field(default_factory=list)
    filtered_waveform: List[float] = field(default_factory=list)
    r_peaks: List[int] = field(default_factory=list)
    rr_intervals_ms: List[float] = field(default_factory=list)
    heart_rate_bpm: Optional[float] = None
    signal_quality: float = 0.0
    quality_label: str = "POOR"
    predicted_rhythm_class: str = "Normal Sinus Rhythm"
    rhythm_confidence: float = 0.95
    state: str = "PLAYING"
    disclaimer: str = "Research prototype — not intended for medical diagnosis."

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame to serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "mode_label": self.mode_label,
            "record_id": self.record_id,
            "window_index": self.window_index,
            "playback_time_sec": self.playback_time_sec,
            "sampling_rate_hz": self.sampling_rate_hz,
            "lead_name": self.lead_name,
            "raw_waveform": self.raw_waveform,
            "filtered_waveform": self.filtered_waveform,
            "r_peaks": self.r_peaks,
            "rr_intervals_ms": self.rr_intervals_ms,
            "heart_rate_bpm": self.heart_rate_bpm,
            "signal_quality": self.signal_quality,
            "quality_label": self.quality_label,
            "predicted_rhythm_class": self.predicted_rhythm_class,
            "rhythm_confidence": self.rhythm_confidence,
            "state": self.state,
            "disclaimer": self.disclaimer,
        }


class ECGPlaybackEngine:
    """Engine simulating real-time playback from cached PhysioNet ECG records."""

    def __init__(
        self,
        record_id: str = "100",
        window_sec: float = 5.0,
        step_sec: float = 1.0,
        playback_speed: float = 1.0,
        loader: Optional[ECGDatasetLoader] = None,
        processor: Optional[ECGProcessor] = None,
    ):
        self.record_id = str(record_id)
        self.window_sec = float(window_sec)
        self.step_sec = float(step_sec)
        self.playback_speed = float(max(0.1, min(10.0, playback_speed)))

        self.loader = loader or ECGDatasetLoader()
        self.processor = processor or ECGProcessor(sampling_rate_hz=360.0)

        # State management
        self.state: PlaybackState = PlaybackState.STOPPED
        self.current_window_idx: int = 0
        self.current_sample_idx: int = 0

        # Cached record signals
        self._signal: Optional[np.ndarray] = None
        self._fs: float = 360.0
        self._lead_name: str = "MLII"
        self._ann_samples: np.ndarray = np.array([], dtype=int)
        self._ann_symbols: List[str] = []
        self._ann_labels: List[str] = []

        self._load_record_cache(self.record_id)

    def _load_record_cache(self, record_id: str) -> None:
        """Cache full record data in memory for low-latency window slicing."""
        self.record_id = record_id
        signal, fs, lead_name = self.loader.load_record(record_id)
        samples, symbols, labels = self.loader.load_annotations(record_id)

        self._signal = signal
        self._fs = fs
        self._lead_name = lead_name
        self._ann_samples = samples
        self._ann_symbols = symbols
        self._ann_labels = labels

        # Ensure processor and classifier have matching sampling rate
        if self.processor.sampling_rate_hz != fs:
            self.processor = ECGProcessor(sampling_rate_hz=fs)
        if not hasattr(self, "classifier") or self.classifier.sampling_rate_hz != fs:
            self.classifier = ECGClassifier(sampling_rate_hz=fs)

        self.current_window_idx = 0
        self.current_sample_idx = 0

    def start(self, record_id: Optional[str] = None) -> None:
        """Start or restart playback."""
        if record_id and record_id != self.record_id:
            self._load_record_cache(record_id)
        elif self.state == PlaybackState.STOPPED:
            self.current_window_idx = 0
            self.current_sample_idx = 0

        self.state = PlaybackState.PLAYING

    def pause(self) -> None:
        """Pause playback without resetting position."""
        if self.state == PlaybackState.PLAYING:
            self.state = PlaybackState.PAUSED

    def resume(self) -> None:
        """Resume playback from current position."""
        if self.state == PlaybackState.PAUSED:
            self.state = PlaybackState.PLAYING

    def stop(self) -> None:
        """Stop playback and reset cursor to the beginning."""
        self.state = PlaybackState.STOPPED
        self.current_window_idx = 0
        self.current_sample_idx = 0

    def reset(self) -> None:
        """Reset cursor to 0:00 without changing state."""
        self.current_window_idx = 0
        self.current_sample_idx = 0

    def set_speed(self, speed: float) -> None:
        """Adjust playback multiplier (e.g. 0.5x, 1.0x, 2.0x)."""
        self.playback_speed = float(max(0.1, min(10.0, speed)))

    def next_frame(self) -> Optional[ECGPlaybackFrame]:
        """Compute and return the next 5-second sliding window frame."""
        if self.state != PlaybackState.PLAYING:
            return None

        if self._signal is None or len(self._signal) == 0:
            return None

        samples_per_window = int(self.window_sec * self._fs)
        samples_per_step = int(self.step_sec * self._fs)

        # Handle end of signal -> loop back seamlessly
        if (self.current_sample_idx + samples_per_window) > len(self._signal):
            self.current_sample_idx = 0
            self.current_window_idx = 0

        start_idx = self.current_sample_idx
        end_idx = start_idx + samples_per_window
        raw_window = self._signal[start_idx:end_idx]

        # Process the window through ECG processor
        analysis: ECGAnalysisResult = self.processor.process(raw_window)

        # Real-time ML arrhythmia classification
        cls_res = self.classifier.predict(
            signal=raw_window,
            sampling_rate_hz=self._fs,
            rr_intervals_ms=analysis.rr_intervals_ms.tolist() if len(analysis.rr_intervals_ms) > 0 else None,
        )

        frame = ECGPlaybackFrame(
            timestamp=datetime.now(timezone.utc).isoformat(),
            mode="ecg_dataset",
            mode_label="ECG Dataset / Simulated Real-Time Mode",
            record_id=self.record_id,
            window_index=self.current_window_idx,
            playback_time_sec=round(start_idx / self._fs, 2),
            sampling_rate_hz=self._fs,
            lead_name=self._lead_name,
            raw_waveform=[round(float(v), 4) for v in raw_window],
            filtered_waveform=[round(float(v), 4) for v in analysis.filtered_signal],
            r_peaks=analysis.r_peaks.tolist(),
            rr_intervals_ms=[round(float(r), 1) for r in analysis.rr_intervals_ms],
            heart_rate_bpm=analysis.heart_rate_bpm,
            signal_quality=analysis.signal_quality,
            quality_label=analysis.quality_label,
            predicted_rhythm_class=cls_res.rhythm_class,
            rhythm_confidence=round(float(cls_res.rhythm_confidence), 4),
            state=self.state.value,
        )

        # Advance cursor by 1 step
        self.current_sample_idx += samples_per_step
        self.current_window_idx += 1

        return frame

    async def stream_frames(
        self,
        max_frames: Optional[int] = None,
        poll_interval: Optional[float] = None
    ) -> AsyncIterator[ECGPlaybackFrame]:
        """Asynchronously yield playback frames respecting the configured playback speed."""
        frames_sent = 0
        while True:
            if self.state == PlaybackState.PLAYING:
                frame = self.next_frame()
                if frame:
                    yield frame
                    frames_sent += 1
                    if max_frames and frames_sent >= max_frames:
                        break

                # Interval calculated from step_sec and playback_speed
                sleep_sec = poll_interval if poll_interval is not None else (self.step_sec / self.playback_speed)
                await asyncio.sleep(sleep_sec)
            elif self.state == PlaybackState.PAUSED:
                await asyncio.sleep(0.1)
            else:  # STOPPED
                break


def from_symbol_to_rhythm(symbol: str) -> str:
    """Helper mapping annotation symbol to human-readable rhythm name."""
    mapping = {
        "N": "Normal Sinus Rhythm",
        "L": "Left Bundle Branch Block",
        "R": "Right Bundle Branch Block",
        "A": "Atrial Premature Beat",
        "V": "Premature Ventricular Contraction",
        "F": "Ventricular Fusion Beat",
        "E": "Ventricular Escape Beat",
        "/": "Paced Beat",
    }
    return mapping.get(symbol, "Normal Sinus Rhythm")
