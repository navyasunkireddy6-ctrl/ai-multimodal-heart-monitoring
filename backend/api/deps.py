"""Dependency providers and service singletons for FastAPI routes."""

from functools import lru_cache
from typing import Optional

from backend.services.ecg_service import ECGService
from backend.services.ppg_service import PPGService
from backend.services.trend_service import TrendService
from backend.services.session_service import SessionService
from backend.services.multimodal_service import MultimodalService
from backend.services.alert_service import AlertService
from backend.services.report_service import ReportService
from backend.services.benchmark_service import BenchmarkService
from backend.services.audit_service import AuditService
from ml.ecg.playback import ECGPlaybackEngine

_session_service = SessionService()
_alert_service = AlertService()
_report_service: Optional[ReportService] = None
_benchmark_service: Optional[BenchmarkService] = None
_audit_service: Optional[AuditService] = None
_ecg_service: Optional[ECGService] = None
_ppg_service: Optional[PPGService] = None
_trend_service: Optional[TrendService] = None
_playback_engine: Optional[ECGPlaybackEngine] = None
_multimodal_service: Optional[MultimodalService] = None


def get_session_service() -> SessionService:
    """Return the global SessionService instance."""
    return _session_service


def get_alert_service() -> AlertService:
    """Return the global AlertService instance."""
    return _alert_service


def get_report_service() -> ReportService:
    """Return or initialize the global ReportService instance."""
    global _report_service
    if _report_service is None:
        _report_service = ReportService(
            session_service=_session_service,
            alert_service=_alert_service,
        )
    return _report_service


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


def get_multimodal_service() -> MultimodalService:
    """Return or initialize the global MultimodalService instance."""
    global _multimodal_service
    if _multimodal_service is None:
        _multimodal_service = MultimodalService(
            session_service=_session_service,
            alert_service=_alert_service,
        )
    return _multimodal_service


def get_benchmark_service() -> BenchmarkService:
    """Return or initialize the global BenchmarkService instance."""
    global _benchmark_service
    if _benchmark_service is None:
        _benchmark_service = BenchmarkService()
    return _benchmark_service


def get_audit_service() -> AuditService:
    """Return or initialize the global AuditService instance."""
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService(
            session_service=_session_service,
            alert_service=_alert_service,
            ecg_service=get_ecg_service(),
            ppg_service=get_ppg_service(),
            trend_service=get_trend_service(),
            multimodal_service=get_multimodal_service(),
            report_service=get_report_service(),
            benchmark_service=get_benchmark_service(),
        )
    return _audit_service
