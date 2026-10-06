"""Automated unit and integration tests for Phase 7 Rate Trend Analysis."""

from datetime import datetime, timezone, timedelta
import numpy as np
import pytest

from ml.trend.analyzer import TrendAnalyzer, TrendAnalysisResult, MODEL_VERSION
from backend.services.trend_service import TrendService
from backend.models.schemas import (
    TrendPredictRequest,
    TrendPredictResponse,
    TrendLabelEnum,
)


def test_trend_analyzer_initialization():
    """Verify default parameters and model version."""
    analyzer = TrendAnalyzer(slope_threshold_bpm_per_min=3.0)
    assert analyzer.slope_threshold == 3.0
    assert analyzer.model_version == MODEL_VERSION


def test_trend_increasing_linear_slope():
    """Verify that a rising rate sequence is classified as INCREASING with positive slope."""
    analyzer = TrendAnalyzer(slope_threshold_bpm_per_min=3.0)
    base_time = datetime(2026, 9, 30, 12, 0, 0, tzinfo=timezone.utc)

    # 70 to 86 BPM over 60 seconds -> +16.0 BPM/min slope
    timestamps = [(base_time + timedelta(seconds=i * 15)).isoformat() for i in range(5)]
    rates = [70.0, 74.0, 78.0, 82.0, 86.0]

    res = analyzer.analyze(timestamps, rates)
    assert isinstance(res, TrendAnalysisResult)
    assert res.trend == "INCREASING"
    assert abs(res.slope_bpm_per_min - 16.0) < 0.20
    assert res.confidence >= 0.85
    assert abs(res.window_duration_seconds - 60.0) < 0.10
    assert res.model_version == MODEL_VERSION


def test_trend_decreasing_linear_slope():
    """Verify that a decelerating rate sequence is classified as DECREASING."""
    analyzer = TrendAnalyzer(slope_threshold_bpm_per_min=3.0)
    base_time = datetime(2026, 9, 30, 12, 0, 0, tzinfo=timezone.utc)

    # 120 to 90 BPM over 60 seconds -> -30.0 BPM/min slope
    timestamps = [(base_time + timedelta(seconds=i * 15)).isoformat() for i in range(5)]
    rates = [120.0, 112.5, 105.0, 97.5, 90.0]

    res = analyzer.analyze(timestamps, rates)
    assert res.trend == "DECREASING"
    assert abs(res.slope_bpm_per_min - (-30.0)) < 0.20
    assert res.confidence >= 0.85
    assert abs(res.window_duration_seconds - 60.0) < 0.10


def test_trend_stable_resting_variability():
    """Verify that resting rates with small natural fluctuations are classified as STABLE."""
    analyzer = TrendAnalyzer(slope_threshold_bpm_per_min=3.0)
    base_time = datetime(2026, 9, 30, 12, 0, 0, tzinfo=timezone.utc)

    timestamps = [(base_time + timedelta(seconds=i * 10)).isoformat() for i in range(6)]
    rates = [72.0, 72.4, 71.8, 72.2, 71.9, 72.1]

    res = analyzer.analyze(timestamps, rates)
    assert res.trend == "STABLE"
    assert abs(res.slope_bpm_per_min) <= 3.0
    assert res.confidence >= 0.80
    assert abs(res.window_duration_seconds - 50.0) < 0.10


def test_trend_slope_bpm_per_min_scaling():
    """Verify slope conversion from seconds to minutes (30 sec interval, 5 BPM increase -> +10 BPM/min)."""
    analyzer = TrendAnalyzer()
    timestamps = ["2026-09-30T12:00:00Z", "2026-09-30T12:00:30Z"]
    rates = [70.0, 75.0]

    res = analyzer.analyze(timestamps, rates)
    assert res.trend == "INCREASING"
    assert abs(res.slope_bpm_per_min - 10.0) < 0.10
    assert abs(res.window_duration_seconds - 30.0) < 0.10


def test_trend_empty_and_single_point_safety():
    """Verify analyzer safely handles empty lists or single point."""
    analyzer = TrendAnalyzer()

    # Empty
    res_empty = analyzer.analyze([], [])
    assert res_empty.trend == "STABLE"
    assert res_empty.slope_bpm_per_min == 0.0
    assert res_empty.window_duration_seconds == 0.0

    # Single point
    res_single = analyzer.analyze(["2026-09-30T12:00:00Z"], [72.0])
    assert res_single.trend == "STABLE"
    assert res_single.slope_bpm_per_min == 0.0
    assert res_single.window_duration_seconds == 0.0


def test_trend_unordered_timestamps_sorting():
    """Verify analyzer automatically sorts timestamps chronologically."""
    analyzer = TrendAnalyzer()
    # Scrambled order
    timestamps = [
        "2026-09-30T12:00:30Z",
        "2026-09-30T12:00:00Z",
        "2026-09-30T12:00:15Z",
    ]
    rates = [80.0, 70.0, 75.0]

    res = analyzer.analyze(timestamps, rates)
    assert res.trend == "INCREASING"
    assert abs(res.slope_bpm_per_min - 20.0) < 0.20
    assert abs(res.window_duration_seconds - 30.0) < 0.10


def test_trend_numeric_and_iso_timestamps():
    """Verify numeric POSIX seconds and ISO-8601 strings yield equivalent results."""
    analyzer = TrendAnalyzer()
    epoch_base = 1790769600.0  # 2026-09-30 12:00:00 UTC
    numeric_ts = [epoch_base, epoch_base + 15.0, epoch_base + 30.0]
    rates = [70.0, 75.0, 80.0]

    res = analyzer.analyze(numeric_ts, rates)
    assert res.trend == "INCREASING"
    assert abs(res.slope_bpm_per_min - 20.0) < 0.20


def test_trend_nan_and_none_filtering():
    """Verify analyzer gracefully filters out None and NaN rates."""
    analyzer = TrendAnalyzer()
    timestamps = [
        "2026-09-30T12:00:00Z",
        "2026-09-30T12:00:10Z",
        "2026-09-30T12:00:20Z",
        "2026-09-30T12:00:30Z",
    ]
    rates = [70.0, None, float("nan"), 80.0]

    res = analyzer.analyze(timestamps, rates)
    assert res.trend == "INCREASING"
    assert abs(res.slope_bpm_per_min - 20.0) < 0.20


def test_trend_service_canonical_contract():
    """Verify TrendService.predict_trend produces canonical TrendPredictResponse."""
    service = TrendService(slope_threshold_bpm_per_min=3.0)
    request = TrendPredictRequest(
        timestamps=["2026-09-30T12:00:00Z", "2026-09-30T12:00:15Z"],
        rates=[70.0, 74.0],
    )
    response = service.predict_trend(request)

    assert isinstance(response, TrendPredictResponse)
    assert response.trend == TrendLabelEnum.INCREASING
    assert abs(response.slope_bpm_per_min - 16.0) < 0.20
    assert 0.0 <= response.confidence <= 1.0
    assert abs(response.window_duration_seconds - 15.0) < 0.10
    assert "Research prototype" in response.disclaimer


def test_trend_dict_serialization():
    """Verify TrendAnalysisResult converts to a clean dictionary."""
    result = TrendAnalysisResult(
        trend="STABLE",
        slope_bpm_per_min=0.5,
        confidence=0.92,
        window_duration_seconds=30.0,
    )
    d = result.to_dict()
    assert d["trend"] == "STABLE"
    assert d["slope_bpm_per_min"] == 0.5
    assert d["confidence"] == 0.92
    assert d["model_version"] == MODEL_VERSION
