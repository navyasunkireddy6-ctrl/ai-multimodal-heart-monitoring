"""Automated unit and integration tests for simulated real-time ECG playback engine."""

import pytest
import asyncio
from ml.ecg.playback import (
    ECGPlaybackEngine,
    ECGPlaybackFrame,
    PlaybackState,
    from_symbol_to_rhythm,
)


def test_playback_engine_initialization():
    """Verify initial playback engine attributes and default STOPPED state."""
    engine = ECGPlaybackEngine(record_id="100", window_sec=5.0, step_sec=1.0)
    assert engine.state == PlaybackState.STOPPED
    assert engine.record_id == "100"
    assert engine.window_sec == 5.0
    assert engine.step_sec == 1.0
    assert engine.playback_speed == 1.0
    assert engine.current_window_idx == 0
    assert engine.current_sample_idx == 0


def test_playback_state_machine_transitions():
    """Verify state transitions: STOPPED -> PLAYING -> PAUSED -> PLAYING -> STOPPED."""
    engine = ECGPlaybackEngine(record_id="100")
    assert engine.state == PlaybackState.STOPPED

    # When STOPPED, next_frame() returns None
    assert engine.next_frame() is None

    # Start
    engine.start()
    assert engine.state == PlaybackState.PLAYING

    # Next frame generates data
    frame = engine.next_frame()
    assert frame is not None
    assert frame.window_index == 0

    # Pause
    engine.pause()
    assert engine.state == PlaybackState.PAUSED
    assert engine.next_frame() is None

    # Resume
    engine.resume()
    assert engine.state == PlaybackState.PLAYING
    frame2 = engine.next_frame()
    assert frame2 is not None
    assert frame2.window_index == 1

    # Stop resets position
    engine.stop()
    assert engine.state == PlaybackState.STOPPED
    assert engine.current_window_idx == 0
    assert engine.current_sample_idx == 0


def test_playback_frame_structure_and_canonical_label():
    """Verify emitted frame contents, waveform length, and exact mode label."""
    engine = ECGPlaybackEngine(record_id="100", window_sec=5.0, step_sec=1.0)
    engine.start()
    frame: ECGPlaybackFrame = engine.next_frame()

    assert frame is not None
    assert frame.mode == "ecg_dataset"
    assert frame.mode_label == "ECG Dataset / Simulated Real-Time Mode"
    assert frame.record_id == "100"
    assert frame.sampling_rate_hz == 360.0

    # 5 seconds at 360 Hz = 1800 samples
    assert len(frame.raw_waveform) == 1800
    assert len(frame.filtered_waveform) == 1800
    assert len(frame.r_peaks) >= 4

    # Physiological heart rate for Record 100 is ~70-80 BPM
    assert frame.heart_rate_bpm is not None
    assert 60.0 <= frame.heart_rate_bpm <= 90.0

    assert frame.signal_quality >= 0.70
    assert frame.quality_label == "GOOD"
    assert "not intended for medical diagnosis" in frame.disclaimer


def test_sliding_window_time_progression():
    """Verify consecutive frames advance by exact step size (1.0s)."""
    engine = ECGPlaybackEngine(record_id="100", window_sec=5.0, step_sec=1.0)
    engine.start()

    f0 = engine.next_frame()
    f1 = engine.next_frame()
    f2 = engine.next_frame()

    assert f0.playback_time_sec == 0.0
    assert f1.playback_time_sec == 1.0
    assert f2.playback_time_sec == 2.0
    assert f0.window_index == 0
    assert f1.window_index == 1
    assert f2.window_index == 2


def test_playback_reset():
    """Verify reset returns window cursor back to 0.0s."""
    engine = ECGPlaybackEngine(record_id="100")
    engine.start()
    _ = engine.next_frame()
    _ = engine.next_frame()
    _ = engine.next_frame()

    engine.reset()
    frame = engine.next_frame()
    assert frame.window_index == 0
    assert frame.playback_time_sec == 0.0


def test_playback_speed_adjustment():
    """Verify speed configuration and safety bounds."""
    engine = ECGPlaybackEngine(record_id="100")
    engine.set_speed(2.0)
    assert engine.playback_speed == 2.0

    # Clamping bounds
    engine.set_speed(0.01)
    assert engine.playback_speed == 0.1  # min clamped
    engine.set_speed(50.0)
    assert engine.playback_speed == 10.0  # max clamped


@pytest.mark.asyncio
async def test_async_stream_frames():
    """Verify asynchronous frame streaming generator."""
    engine = ECGPlaybackEngine(record_id="100", window_sec=5.0, step_sec=1.0)
    engine.start()

    received = []
    async for frame in engine.stream_frames(max_frames=3, poll_interval=0.01):
        received.append(frame)

    assert len(received) == 3
    assert received[0].window_index == 0
    assert received[1].window_index == 1
    assert received[2].window_index == 2


def test_frame_to_dict_serialization():
    """Verify frame converts to clean dictionary with all expected keys."""
    engine = ECGPlaybackEngine(record_id="100")
    engine.start()
    frame = engine.next_frame()
    d = frame.to_dict()

    expected_keys = {
        "timestamp",
        "mode",
        "mode_label",
        "record_id",
        "window_index",
        "playback_time_sec",
        "sampling_rate_hz",
        "lead_name",
        "raw_waveform",
        "filtered_waveform",
        "r_peaks",
        "rr_intervals_ms",
        "heart_rate_bpm",
        "signal_quality",
        "quality_label",
        "predicted_rhythm_class",
        "rhythm_confidence",
        "state",
        "disclaimer",
    }
    assert set(d.keys()) == expected_keys
    assert d["mode_label"] == "ECG Dataset / Simulated Real-Time Mode"
