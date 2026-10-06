"""Dependency providers and service singletons for FastAPI routes."""

from functools import lru_cache
from typing import Optional

from backend.services.ecg_service import ECGService
from backend.services.ppg_service import PPGService
from backend.services.trend_service import TrendService
from backend.services.session_service import SessionService
from ml.ecg.playback import ECGPlaybackEngine

_session_service = SessionService()
_ecg_service: Optional[ECGService] = None
_ppg_service: Optional[PPGService] = None
_trend_service: Optional[TrendService] = None
_playback_engine: Optional[ECGPlaybackEngine] = None


def get_session_service() -> SessionService:
    """Return the global SessionService instance."""
    return _session_service


def get_ecg_service() -> ECGService:
    """Return or initialize the global ECGService instance."""
    global _ecg_service
    if _ecg_service is None:
        _ecg_service = ECGService()
    return _ecg_service


def get_ppg_service() -> PPGService:
    """Return or initialize the global PPGService instance."""
    global _ppg_service
    if _ppg_service is None:
        _ppg_service = PPGService()
    return _ppg_service


def get_trend_service() -> TrendService:
    """Return or initialize the global TrendService instance."""
    global _trend_service
    if _trend_service is None:
        _trend_service = TrendService()
    return _trend_service


def get_playback_engine() -> ECGPlaybackEngine:
    """Return or initialize the simulated real-time ECG playback engine."""
    global _playback_engine
    if _playback_engine is None:
        _playback_engine = ECGPlaybackEngine(record_id="100", window_sec=5.0, step_sec=1.0)
        _playback_engine.start()
    return _playback_engine
