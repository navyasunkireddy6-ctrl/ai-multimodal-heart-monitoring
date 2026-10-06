"""Canonical Pydantic Schemas for Multimodal Heart & Pulse Rate Monitoring System.

This module represents the canonical data schema and interface types.
Field names and enum values are non-negotiable and strictly adhere to docs/API_CONTRACT.md.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator


# ============================================================================
# Canonical Enums
# ============================================================================

class ModeEnum(str, Enum):
    """Canonical operating modes."""
    ECG_DATASET = "ecg_dataset"
    SMARTPHONE_PPG = "smartphone_ppg"
    SYNTHETIC_PPG = "synthetic_ppg"
    OFFLINE_DEMO = "offline_demo"


class QualityLabelEnum(str, Enum):
    """Canonical signal quality labels."""
    GOOD = "GOOD"
    FAIR = "FAIR"
    POOR = "POOR"


class TrendLabelEnum(str, Enum):
    """Canonical heart/pulse rate trend labels."""
    INCREASING = "INCREASING"
    STABLE = "STABLE"
    DECREASING = "DECREASING"


# ============================================================================
# Canonical Measurement Schema
# ============================================================================

class CanonicalMeasurement(BaseModel):
    """Single Source of Truth for cardiovascular measurement data."""
    model_config = ConfigDict(use_enum_values=True, populate_by_name=True)

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 formatted timestamp of the measurement window"
    )
    mode: ModeEnum = Field(
        ...,
        description="Operating mode: ecg_dataset, smartphone_ppg, synthetic_ppg, offline_demo"
    )
    heart_rate_bpm: Optional[float] = Field(
        None,
        description="Estimated heart rate in beats per minute derived from ECG R-R intervals",
        ge=0.0,
        le=300.0
    )
    pulse_rate_bpm: Optional[float] = Field(
        None,
        description="Estimated pulse rate in beats per minute derived from PPG inter-beat intervals",
        ge=0.0,
        le=300.0
    )
    signal_quality: float = Field(
        ...,
        description="Normalized Signal Quality Index (SQI) between 0.0 and 1.0",
        ge=0.0,
        le=1.0
    )
    quality_label: QualityLabelEnum = Field(
        ...,
        description="Categorical signal quality: GOOD, FAIR, POOR"
    )
    rhythm_class: Optional[str] = Field(
        None,
        description="Predicted ECG rhythm class from actual dataset labels (e.g., 'Normal Sinus Rhythm')"
    )
    rhythm_confidence: Optional[float] = Field(
        None,
        description="Classifier probability/confidence score in range [0.0, 1.0]",
        ge=0.0,
        le=1.0
    )
    trend: Optional[TrendLabelEnum] = Field(
        None,
        description="Rate trajectory: INCREASING, STABLE, DECREASING"
    )
    model_version: str = Field(
        ...,
        description="Identifier string of the model or algorithm pipeline utilized"
    )


# ============================================================================
# REST Endpoint Request & Response Schemas
# ============================================================================

class HealthResponse(BaseModel):
    """Backend service health and system information."""
    model_config = ConfigDict(use_enum_values=True)

    status: str = Field("healthy", description="Current health status of backend service")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Current server time (ISO-8601)"
    )
    version: str = Field("1.0.0", description="API contract version")
    models_loaded: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary mapping component name to active model version"
    )
    disclaimer: str = Field(
        "Research prototype — not intended for medical diagnosis.",
        description="Mandatory prototype research notice"
    )


class ECGProcessRequest(BaseModel):
    """Request payload for ECG signal filtering and peak detection."""
    sampling_rate_hz: float = Field(360.0, description="Sampling rate in Hz (e.g. 360 for MIT-BIH)", gt=0)
    signal: List[float] = Field(..., description="Raw digitized ECG signal voltage samples")
    record_id: Optional[str] = Field(None, description="Optional record reference identifier")


class ECGProcessResponse(BaseModel):
    """Response containing filtered waveform and extracted peaks."""
    model_config = ConfigDict(use_enum_values=True)

    heart_rate_bpm: Optional[float] = Field(None, description="Calculated heart rate from valid R-R intervals")
    signal_quality: float = Field(..., description="ECG signal quality index [0.0, 1.0]")
    quality_label: QualityLabelEnum = Field(..., description="Quality assessment: GOOD, FAIR, POOR")
    r_peaks: List[int] = Field(default_factory=list, description="Sample indices corresponding to detected R-peaks")
    rr_intervals_ms: List[float] = Field(default_factory=list, description="Extracted consecutive R-R intervals in ms")
    filtered_signal: List[float] = Field(default_factory=list, description="Zero-phase bandpass filtered signal samples")


class ECGClassifyRequest(BaseModel):
    """Request payload for ECG rhythm classification."""
    sampling_rate_hz: float = Field(360.0, gt=0)
    signal: List[float] = Field(..., description="ECG signal segment")
    rr_intervals_ms: Optional[List[float]] = Field(default=None, description="Optional pre-extracted R-R intervals")


class ECGClassifyResponse(BaseModel):
    """Response containing predicted ECG rhythm class."""
    model_config = ConfigDict(use_enum_values=True)

    rhythm_class: str = Field(..., description="Predicted ECG rhythm class based on dataset labels")
    rhythm_confidence: float = Field(..., description="Classification confidence [0.0, 1.0]")
    model_version: str = Field(..., description="Active model release identifier")
    features: Dict[str, float] = Field(default_factory=dict, description="Extracted HRV and morphological feature values")
    disclaimer: str = Field(
        "Research prototype — not intended for medical diagnosis.",
        description="Mandatory medical disclaimer"
    )


class PPGProcessRequest(BaseModel):
    """Request payload for processing optical fingertip PPG signal."""
    sampling_rate_hz: float = Field(30.0, description="Video/camera frame rate in Hz (typically 30 FPS)", gt=0)
    signal: List[float] = Field(..., description="Extracted mean optical intensity series (e.g. red channel)")
    source: Optional[str] = Field("smartphone_camera", description="Acquisition source: smartphone_camera or synthetic")


class PPGProcessResponse(BaseModel):
    """Response containing pulse rate estimation and quality."""
    model_config = ConfigDict(use_enum_values=True)

    pulse_rate_bpm: Optional[float] = Field(None, description="Pulse rate in BPM (null if quality is POOR)")
    signal_quality: float = Field(..., description="Normalized PPG signal quality [0.0, 1.0]")
    quality_label: QualityLabelEnum = Field(..., description="Quality label: GOOD, FAIR, POOR")
    peaks: List[int] = Field(default_factory=list, description="Indices of detected systolic peaks")
    filtered_signal: List[float] = Field(default_factory=list, description="Bandpass filtered & normalized signal")


class PPGQualityRequest(BaseModel):
    """Request payload for standalone PPG signal quality assessment."""
    sampling_rate_hz: float = Field(30.0, gt=0)
    signal: List[float] = Field(..., description="Optical intensity values")


class PPGQualityResponse(BaseModel):
    """Detailed quality assessment metrics."""
    model_config = ConfigDict(use_enum_values=True)

    signal_quality: float = Field(..., description="Overall SQI [0.0, 1.0]")
    quality_label: QualityLabelEnum = Field(..., description="GOOD, FAIR, POOR")
    snr_db: Optional[float] = Field(None, description="Estimated optical SNR in decibels")
    is_usable: bool = Field(..., description="True if pulse rate calculation is deemed reliable")
    message: str = Field(..., description="Human-readable feedback or guidance")


class TrendPredictRequest(BaseModel):
    """Request payload for rate trend calculation."""
    timestamps: List[str] = Field(..., description="List of ISO-8601 measurement timestamps")
    rates: List[float] = Field(..., description="Corresponding heart rate or pulse rate values in BPM")


class TrendPredictResponse(BaseModel):
    """Rate trend classification response."""
    model_config = ConfigDict(use_enum_values=True)

    trend: TrendLabelEnum = Field(..., description="INCREASING, STABLE, DECREASING")
    slope_bpm_per_min: float = Field(..., description="Rate of change in BPM per minute")
    confidence: float = Field(..., description="Statistical confidence in trend label [0.0, 1.0]")
    window_duration_seconds: float = Field(..., description="Duration of evaluated time window")
    disclaimer: str = Field(
        "Research prototype — not intended for medical diagnosis.",
        description="Mandatory medical disclaimer"
    )


# ============================================================================
# Session Storage Schemas
# ============================================================================

class SessionCreateRequest(BaseModel):
    """Payload to initiate a new cardiac monitoring session."""
    mode: ModeEnum = Field(..., description="Operating mode of session")
    notes: Optional[str] = Field(None, description="Optional session notes or metadata")


class SessionResponse(BaseModel):
    """Summary representation of a recorded session."""
    model_config = ConfigDict(use_enum_values=True)

    session_id: str = Field(..., description="Unique UUID for this session")
    created_at: str = Field(..., description="ISO-8601 creation timestamp")
    mode: ModeEnum = Field(..., description="Primary mode of session")
    measurement_count: int = Field(0, description="Total recorded measurement entries")
    avg_heart_rate_bpm: Optional[float] = Field(None, description="Average ECG heart rate across session")
    avg_pulse_rate_bpm: Optional[float] = Field(None, description="Average PPG pulse rate across session")
    notes: Optional[str] = Field(None)


class SessionDetailResponse(BaseModel):
    """Detailed session report including complete historical measurements."""
    model_config = ConfigDict(use_enum_values=True)

    session_id: str = Field(..., description="Unique UUID")
    created_at: str = Field(..., description="Creation timestamp")
    mode: ModeEnum = Field(...)
    notes: Optional[str] = Field(None)
    measurements: List[CanonicalMeasurement] = Field(default_factory=list)


class SessionListResponse(BaseModel):
    """List container of recorded sessions."""
    model_config = ConfigDict(use_enum_values=True)

    total: int = Field(..., description="Total sessions stored")
    sessions: List[SessionResponse] = Field(default_factory=list)


# ============================================================================
# Live PPG WebSocket Frame Schemas
# ============================================================================

class PPGWebSocketFrame(BaseModel):
    """Inbound optical frame data sent over /ws/v1/ppg."""
    type: str = Field("ppg_frame", description="Frame message identifier")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Frame capture timestamp"
    )
    session_id: Optional[str] = Field(None, description="Associated session ID if recorded")
    red_channel_value: float = Field(..., description="Mean intensity of red channel in fingertip ROI")
    green_channel_value: Optional[float] = Field(None, description="Mean intensity of green channel")
    blue_channel_value: Optional[float] = Field(None, description="Mean intensity of blue channel")
    flash_enabled: bool = Field(True, description="State of smartphone torch/flash")
    camera_fps: float = Field(30.0, description="Camera acquisition frame rate in Hz")


class PPGWebSocketResponse(BaseModel):
    """Outbound real-time update broadcast from /ws/v1/ppg."""
    model_config = ConfigDict(use_enum_values=True)

    type: str = Field("measurement_update", description="WebSocket packet type")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    mode: ModeEnum = Field(ModeEnum.SMARTPHONE_PPG)
    heart_rate_bpm: Optional[float] = Field(None)
    pulse_rate_bpm: Optional[float] = Field(None)
    signal_quality: float = Field(..., ge=0.0, le=1.0)
    quality_label: QualityLabelEnum = Field(...)
    rhythm_class: Optional[str] = Field(None)
    rhythm_confidence: Optional[float] = Field(None)
    trend: Optional[TrendLabelEnum] = Field(None)
    model_version: str = Field(...)
    raw_red: Optional[float] = Field(None)
    filtered_value: Optional[float] = Field(None)
    warning: Optional[str] = Field(None, description="Advisory message when signal quality is degraded")


# ============================================================================
# Multimodal Synchronization & Fusion Schemas
# ============================================================================

class MultimodalAnalyzeRequest(BaseModel):
    """Request payload for simultaneous multimodal ECG and PPG evaluation."""
    ecg_signal: List[float] = Field(..., description="ECG signal voltage samples")
    ecg_sampling_rate_hz: float = Field(360.0, description="ECG sampling rate in Hz", gt=0)
    ppg_signal: List[float] = Field(..., description="Optical PPG intensity samples")
    ppg_sampling_rate_hz: float = Field(30.0, description="PPG camera/sensor sampling rate in Hz", gt=0)
    record_id: Optional[str] = Field(None, description="Optional dataset record identifier")
    session_id: Optional[str] = Field(None, description="Optional active session ID to append record to")


class MultimodalAnalyzeResponse(BaseModel):
    """Response containing synchronized dual-rate comparison, transit time, and fused metrics."""
    model_config = ConfigDict(use_enum_values=True)

    ecg_heart_rate_bpm: Optional[float] = Field(None, description="ECG-derived heart rate")
    ppg_pulse_rate_bpm: Optional[float] = Field(None, description="PPG-derived pulse rate")
    rate_discrepancy_bpm: Optional[float] = Field(None, description="Absolute difference |HR - PR| in BPM")
    relative_discrepancy_pct: Optional[float] = Field(None, description="Percentage discrepancy relative to HR")
    agreement_level: str = Field(..., description="EXCELLENT, ACCEPTABLE, DISCREPANT, or UNAVAILABLE")
    pulse_arrival_time_ms: Optional[float] = Field(None, description="Mean transit latency between R-peak and systolic peak in ms")
    pat_variability_ms: Optional[float] = Field(None, description="Standard deviation of transit latency across window")
    ecg_beat_count: int = Field(..., description="Detected ECG R-peaks in window")
    ppg_beat_count: int = Field(..., description="Detected PPG systolic peaks in window")
    pulse_deficit_detected: bool = Field(False, description="Flag indicating pulse deficit anomaly")
    ecg_signal_quality: float = Field(..., description="ECG SQI [0.0, 1.0]")
    ppg_signal_quality: float = Field(..., description="PPG SQI [0.0, 1.0]")
    fused_signal_quality: float = Field(..., description="Weighted multimodal fusion SQI [0.0, 1.0]")
    fused_quality_label: QualityLabelEnum = Field(..., description="GOOD, FAIR, POOR")
    rhythm_class: Optional[str] = Field(None, description="ECG predicted rhythm class")
    rhythm_confidence: Optional[float] = Field(None, description="Arrhythmia classifier confidence")
    trend: Optional[TrendLabelEnum] = Field(None, description="Rate trajectory trend")
    recommended_primary_modality: str = Field("BOTH", description="Recommended primary sensor: BOTH, ECG, or PPG")
    canonical_measurement: CanonicalMeasurement = Field(..., description="Standardized canonical measurement record")
    model_version: str = Field(..., description="Fusion pipeline version")
    disclaimer: str = Field(
        "Research prototype — not intended for medical diagnosis.",
        description="Mandatory medical disclaimer"
    )

