"""Automated tests for Canonical API Contract and Pydantic Schemas."""

import json
import pytest
from pydantic import ValidationError

from backend.models.schemas import (
    ModeEnum,
    QualityLabelEnum,
    TrendLabelEnum,
    CanonicalMeasurement,
    HealthResponse,
    ECGProcessRequest,
    ECGProcessResponse,
    ECGClassifyRequest,
    ECGClassifyResponse,
    PPGProcessRequest,
    PPGProcessResponse,
    PPGQualityRequest,
    PPGQualityResponse,
    TrendPredictRequest,
    TrendPredictResponse,
    SessionCreateRequest,
    SessionResponse,
    SessionDetailResponse,
    SessionListResponse,
    PPGWebSocketFrame,
    PPGWebSocketResponse,
)


def test_canonical_measurement_exact_fields():
    """Verify CanonicalMeasurement supports the exact canonical fields."""
    expected_fields = {
        "timestamp",
        "mode",
        "heart_rate_bpm",
        "pulse_rate_bpm",
        "signal_quality",
        "quality_label",
        "rhythm_class",
        "rhythm_confidence",
        "trend",
        "model_version",
    }
    actual_fields = set(CanonicalMeasurement.model_fields.keys())
    assert actual_fields == expected_fields, f"Fields mismatch: {actual_fields} vs {expected_fields}"


def test_canonical_enums():
    """Verify all canonical modes, quality labels, and trend labels."""
    assert {e.value for e in ModeEnum} == {
        "ecg_dataset",
        "smartphone_ppg",
        "synthetic_ppg",
        "offline_demo",
    }

    assert {e.value for e in QualityLabelEnum} == {"GOOD", "FAIR", "POOR"}

    assert {e.value for e in TrendLabelEnum} == {
        "INCREASING",
        "STABLE",
        "DECREASING",
    }


def test_canonical_measurement_valid_ecg():
    """Test valid ECG measurement creation and serialization."""
    measurement = CanonicalMeasurement(
        timestamp="2026-09-30T12:00:00Z",
        mode=ModeEnum.ECG_DATASET,
        heart_rate_bpm=74.5,
        pulse_rate_bpm=None,
        signal_quality=0.92,
        quality_label=QualityLabelEnum.GOOD,
        rhythm_class="Normal Sinus Rhythm",
        rhythm_confidence=0.96,
        trend=TrendLabelEnum.STABLE,
        model_version="rf-ecg-v1.0.0",
    )
    data = measurement.model_dump()
    assert data["mode"] == "ecg_dataset"
    assert data["quality_label"] == "GOOD"
    assert data["trend"] == "STABLE"
    assert data["heart_rate_bpm"] == 74.5
    assert data["pulse_rate_bpm"] is None

    # Test JSON serialization round-trip
    json_str = measurement.model_dump_json()
    parsed = CanonicalMeasurement.model_validate_json(json_str)
    assert parsed.heart_rate_bpm == 74.5
    assert parsed.signal_quality == 0.92


def test_canonical_measurement_valid_ppg():
    """Test valid PPG measurement without ECG fields."""
    measurement = CanonicalMeasurement(
        mode=ModeEnum.SMARTPHONE_PPG,
        heart_rate_bpm=None,
        pulse_rate_bpm=71.2,
        signal_quality=0.85,
        quality_label=QualityLabelEnum.GOOD,
        rhythm_class=None,
        rhythm_confidence=None,
        trend=TrendLabelEnum.INCREASING,
        model_version="ppg-peak-v1.0.0",
    )
    assert measurement.pulse_rate_bpm == 71.2
    assert measurement.heart_rate_bpm is None
    assert measurement.mode == "smartphone_ppg"


def test_canonical_measurement_validation_boundaries():
    """Test value boundary validations (signal_quality, confidence, rates)."""
    # Invalid signal_quality > 1.0
    with pytest.raises(ValidationError):
        CanonicalMeasurement(
            mode=ModeEnum.ECG_DATASET,
            signal_quality=1.5,
            quality_label=QualityLabelEnum.GOOD,
            model_version="v1",
        )

    # Invalid signal_quality < 0.0
    with pytest.raises(ValidationError):
        CanonicalMeasurement(
            mode=ModeEnum.ECG_DATASET,
            signal_quality=-0.1,
            quality_label=QualityLabelEnum.GOOD,
            model_version="v1",
        )

    # Invalid mode enum
    with pytest.raises(ValidationError):
        CanonicalMeasurement(
            mode="invalid_mode",  # type: ignore
            signal_quality=0.5,
            quality_label=QualityLabelEnum.GOOD,
            model_version="v1",
        )

    # Invalid quality label enum
    with pytest.raises(ValidationError):
        CanonicalMeasurement(
            mode=ModeEnum.ECG_DATASET,
            signal_quality=0.5,
            quality_label="EXCELLENT",  # type: ignore
            model_version="v1",
        )


def test_health_response_schema():
    """Test HealthResponse schema."""
    resp = HealthResponse(
        models_loaded={"ecg_classifier": "rf-ecg-v1.0.0"}
    )
    assert resp.status == "healthy"
    assert "not intended for medical diagnosis" in resp.disclaimer


def test_ecg_process_schemas():
    """Test ECG process request and response schemas."""
    req = ECGProcessRequest(
        sampling_rate_hz=360.0,
        signal=[0.1, 0.2, 1.2, 0.3],
        record_id="100",
    )
    assert req.sampling_rate_hz == 360.0

    resp = ECGProcessResponse(
        heart_rate_bpm=75.0,
        signal_quality=0.95,
        quality_label=QualityLabelEnum.GOOD,
        r_peaks=[2],
        rr_intervals_ms=[800.0],
        filtered_signal=[0.05, 0.15, 1.1, 0.25],
    )
    assert resp.quality_label == "GOOD"
    assert len(resp.r_peaks) == 1


def test_ecg_classify_schemas():
    """Test ECG classify request and response schemas."""
    req = ECGClassifyRequest(
        sampling_rate_hz=360.0,
        signal=[0.0] * 500,
        rr_intervals_ms=[800.0, 810.0],
    )
    resp = ECGClassifyResponse(
        rhythm_class="Normal Sinus Rhythm",
        rhythm_confidence=0.98,
        model_version="rf-ecg-v1.0.0",
        features={"mean_rr": 805.0},
    )
    assert resp.rhythm_class == "Normal Sinus Rhythm"
    assert "not intended for medical diagnosis" in resp.disclaimer


def test_ppg_process_schemas():
    """Test PPG process request and response schemas."""
    req = PPGProcessRequest(
        sampling_rate_hz=30.0,
        signal=[120.0, 130.0, 140.0],
    )
    assert req.source == "smartphone_camera"

    resp = PPGProcessResponse(
        pulse_rate_bpm=72.0,
        signal_quality=0.88,
        quality_label=QualityLabelEnum.GOOD,
        peaks=[10, 40],
        filtered_signal=[0.1, 0.5, 0.8],
    )
    assert resp.pulse_rate_bpm == 72.0


def test_ppg_quality_schemas():
    """Test PPG quality request and response schemas."""
    req = PPGQualityRequest(
        sampling_rate_hz=30.0,
        signal=[10.0, 12.0, 11.0],
    )
    resp = PPGQualityResponse(
        signal_quality=0.45,
        quality_label=QualityLabelEnum.FAIR,
        snr_db=8.5,
        is_usable=True,
        message="Moderate noise detected.",
    )
    assert resp.quality_label == "FAIR"
    assert resp.is_usable is True


def test_trend_predict_schemas():
    """Test rate trend predict request and response schemas."""
    req = TrendPredictRequest(
        timestamps=["2026-09-30T12:00:00Z", "2026-09-30T12:00:10Z"],
        rates=[70.0, 78.0],
    )
    resp = TrendPredictResponse(
        trend=TrendLabelEnum.INCREASING,
        slope_bpm_per_min=48.0,
        confidence=0.92,
        window_duration_seconds=10.0,
    )
    assert resp.trend == "INCREASING"
    assert "not intended for medical diagnosis" in resp.disclaimer


def test_session_schemas():
    """Test session creation and detail schemas."""
    req = SessionCreateRequest(
        mode=ModeEnum.SMARTPHONE_PPG,
        notes="Subject at rest",
    )
    assert req.mode == "smartphone_ppg"

    measurement = CanonicalMeasurement(
        mode=ModeEnum.SMARTPHONE_PPG,
        pulse_rate_bpm=72.0,
        signal_quality=0.9,
        quality_label=QualityLabelEnum.GOOD,
        model_version="v1.0.0",
    )

    detail = SessionDetailResponse(
        session_id="session-1234",
        created_at="2026-09-30T12:00:00Z",
        mode=ModeEnum.SMARTPHONE_PPG,
        notes="Session notes",
        measurements=[measurement],
    )
    assert detail.session_id == "session-1234"
    assert len(detail.measurements) == 1


def test_websocket_frame_schemas():
    """Test WebSocket inbound frame and outbound response schemas."""
    inbound = PPGWebSocketFrame(
        red_channel_value=145.2,
        green_channel_value=80.1,
        blue_channel_value=75.3,
        flash_enabled=True,
        camera_fps=30.0,
    )
    assert inbound.type == "ppg_frame"
    assert inbound.red_channel_value == 145.2

    outbound = PPGWebSocketResponse(
        pulse_rate_bpm=74.0,
        signal_quality=0.92,
        quality_label=QualityLabelEnum.GOOD,
        trend=TrendLabelEnum.STABLE,
        model_version="ppg-peak-v1.0.0",
        raw_red=145.2,
        filtered_value=0.35,
    )
    assert outbound.type == "measurement_update"
    assert outbound.pulse_rate_bpm == 74.0
    assert outbound.quality_label == "GOOD"
