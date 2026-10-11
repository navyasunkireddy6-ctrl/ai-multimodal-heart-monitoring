"""Real-time Cardiac Anomaly Alerting & Emergency Notification Service.

Evaluates continuous physiological telemetry (heart rate, pulse rate, arrhythmia
classification, transit latency, and rate trend) against clinical thresholds to trigger
critical alarms, warning notifications, and debounced alert events.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from backend.models.schemas import (
    AlertAcknowledgeResponse,
    AlertConfig,
    AlertListResponse,
    AlertSeverityEnum,
    AlertTypeEnum,
    CanonicalMeasurement,
    CardiacAlert,
    MultimodalAnalyzeResponse,
)


class AlertService:
    """Manages threshold evaluation, cooldown debouncing, and subscriber dispatch for cardiac alerts."""

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self._alerts: List[CardiacAlert] = []
        self._subscribers: List[Callable[[CardiacAlert], Any]] = []
        # Tracks timestamp and severity of last triggered alert per type: {alert_type: (epoch_sec, severity)}
        self._last_alert_state: Dict[str, tuple[float, AlertSeverityEnum]] = {}

    def get_config(self) -> AlertConfig:
        """Retrieve current alert threshold configuration."""
        return self.config

    def update_config(self, new_config: AlertConfig) -> AlertConfig:
        """Update active alert configuration thresholds."""
        self.config = new_config
        return self.config

    def register_subscriber(self, callback: Callable[[CardiacAlert], Any]) -> None:
        """Register a subscriber callback invoked whenever a new alert is triggered."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unregister_subscriber(self, callback: Callable[[CardiacAlert], Any]) -> None:
        """Unregister an alert notification callback."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def clear(self) -> None:
        """Clear all stored alerts and reset cooldown state."""
        self._alerts.clear()
        self._last_alert_state.clear()

    def evaluate_measurement(
        self,
        measurement: CanonicalMeasurement,
        session_id: Optional[str] = None,
    ) -> List[CardiacAlert]:
        """Evaluate a single canonical measurement and trigger appropriate alerts."""
        now_epoch = time.time()
        triggered_alerts: List[CardiacAlert] = []

        # Determine effective rate: prefer heart_rate_bpm, fallback to pulse_rate_bpm
        rate = measurement.heart_rate_bpm if measurement.heart_rate_bpm is not None else measurement.pulse_rate_bpm

        # 1. Rate Thresholds: Tachycardia & Bradycardia
        if rate is not None and rate > 0.0:
            # Tachycardia
            if rate >= self.config.critical_tachycardia_bpm:
                alert = self._create_alert_if_ready(
                    alert_type=AlertTypeEnum.TACHYCARDIA,
                    severity=AlertSeverityEnum.CRITICAL,
                    message=f"CRITICAL: Acute tachycardia detected at {rate:.1f} BPM (Threshold: {self.config.critical_tachycardia_bpm:.1f} BPM).",
                    metrics={"rate_bpm": rate, "threshold": self.config.critical_tachycardia_bpm},
                    session_id=session_id,
                    now_epoch=now_epoch,
                )
                if alert:
                    triggered_alerts.append(alert)
            elif rate >= self.config.tachycardia_threshold_bpm:
                alert = self._create_alert_if_ready(
                    alert_type=AlertTypeEnum.TACHYCARDIA,
                    severity=AlertSeverityEnum.WARNING,
                    message=f"WARNING: Elevated heart rate (Tachycardia) at {rate:.1f} BPM (Threshold: {self.config.tachycardia_threshold_bpm:.1f} BPM).",
                    metrics={"rate_bpm": rate, "threshold": self.config.tachycardia_threshold_bpm},
                    session_id=session_id,
                    now_epoch=now_epoch,
                )
                if alert:
                    triggered_alerts.append(alert)

            # Bradycardia
            if rate <= self.config.critical_bradycardia_bpm:
                alert = self._create_alert_if_ready(
                    alert_type=AlertTypeEnum.BRADYCARDIA,
                    severity=AlertSeverityEnum.CRITICAL,
                    message=f"CRITICAL: Severe bradycardia detected at {rate:.1f} BPM (Threshold: {self.config.critical_bradycardia_bpm:.1f} BPM).",
                    metrics={"rate_bpm": rate, "threshold": self.config.critical_bradycardia_bpm},
                    session_id=session_id,
                    now_epoch=now_epoch,
                )
                if alert:
                    triggered_alerts.append(alert)
            elif rate <= self.config.bradycardia_threshold_bpm:
                alert = self._create_alert_if_ready(
                    alert_type=AlertTypeEnum.BRADYCARDIA,
                    severity=AlertSeverityEnum.WARNING,
                    message=f"WARNING: Low heart rate (Bradycardia) at {rate:.1f} BPM (Threshold: {self.config.bradycardia_threshold_bpm:.1f} BPM).",
                    metrics={"rate_bpm": rate, "threshold": self.config.bradycardia_threshold_bpm},
                    session_id=session_id,
                    now_epoch=now_epoch,
                )
                if alert:
                    triggered_alerts.append(alert)

        # 2. Arrhythmia Detection
        if (
            measurement.rhythm_class
            and measurement.rhythm_class != "Normal Sinus Rhythm"
            and (measurement.rhythm_confidence or 0.0) >= self.config.min_arrhythmia_confidence
        ):
            is_ventricular = "Ventricular" in measurement.rhythm_class
            severity = AlertSeverityEnum.CRITICAL if is_ventricular else AlertSeverityEnum.WARNING
            alert = self._create_alert_if_ready(
                alert_type=AlertTypeEnum.ARRHYTHMIA_DETECTED,
                severity=severity,
                message=f"ARRHYTHMIA ALERT: {measurement.rhythm_class} classified with {measurement.rhythm_confidence * 100:.1f}% confidence.",
                metrics={
                    "rhythm_class": measurement.rhythm_class,
                    "confidence": measurement.rhythm_confidence,
                    "model_version": measurement.model_version,
                },
                session_id=session_id,
                now_epoch=now_epoch,
            )
            if alert:
                triggered_alerts.append(alert)

        # 3. Signal Quality Advisory
        if measurement.signal_quality < self.config.poor_sqi_threshold:
            alert = self._create_alert_if_ready(
                alert_type=AlertTypeEnum.POOR_SIGNAL_QUALITY,
                severity=AlertSeverityEnum.WARNING,
                message=f"POOR SIGNAL QUALITY: SQI degraded to {measurement.signal_quality:.2f} ({measurement.quality_label}). Check sensor contact.",
                metrics={"sqi": measurement.signal_quality, "quality_label": measurement.quality_label},
                session_id=session_id,
                now_epoch=now_epoch,
            )
            if alert:
                triggered_alerts.append(alert)

        # Dispatch and store
        for a in triggered_alerts:
            self._store_and_dispatch(a)

        return triggered_alerts

    def evaluate_multimodal_response(
        self,
        response: MultimodalAnalyzeResponse,
        session_id: Optional[str] = None,
    ) -> List[CardiacAlert]:
        """Evaluate multimodal fusion response including pulse deficit detection."""
        now_epoch = time.time()
        triggered_alerts = self.evaluate_measurement(response.canonical_measurement, session_id=session_id)

        # Check Arrhythmia from multimodal response if not already triggered
        if (
            response.rhythm_class
            and response.rhythm_class != "Normal Sinus Rhythm"
            and (response.rhythm_confidence or 0.0) >= self.config.min_arrhythmia_confidence
            and not any(a.alert_type == AlertTypeEnum.ARRHYTHMIA_DETECTED for a in triggered_alerts)
        ):
            is_ventricular = "Ventricular" in response.rhythm_class
            severity = AlertSeverityEnum.CRITICAL if is_ventricular else AlertSeverityEnum.WARNING
            arrhythmia_alert = self._create_alert_if_ready(
                alert_type=AlertTypeEnum.ARRHYTHMIA_DETECTED,
                severity=severity,
                message=f"ARRHYTHMIA ALERT: {response.rhythm_class} classified with {response.rhythm_confidence * 100:.1f}% confidence.",
                metrics={
                    "rhythm_class": response.rhythm_class,
                    "confidence": response.rhythm_confidence,
                    "model_version": response.model_version,
                },
                session_id=session_id,
                now_epoch=now_epoch,
            )
            if arrhythmia_alert:
                triggered_alerts.append(arrhythmia_alert)
                self._store_and_dispatch(arrhythmia_alert)

        # Check Pulse Deficit
        if self.config.pulse_deficit_alert_enabled and response.pulse_deficit_detected:
            deficit_severity = (
                AlertSeverityEnum.CRITICAL
                if (response.rate_discrepancy_bpm or 0.0) > 15.0
                else AlertSeverityEnum.WARNING
            )
            alert = self._create_alert_if_ready(
                alert_type=AlertTypeEnum.PULSE_DEFICIT,
                severity=deficit_severity,
                message=(
                    f"PULSE DEFICIT ALERT: ECG rate ({response.ecg_heart_rate_bpm:.1f} BPM) and PPG rate "
                    f"({response.ppg_pulse_rate_bpm:.1f} BPM) differ by {response.rate_discrepancy_bpm:.1f} BPM. "
                    "Possible peripheral vascular deficit or premature non-perfusing contraction."
                ),
                metrics={
                    "ecg_hr": response.ecg_heart_rate_bpm,
                    "ppg_pr": response.ppg_pulse_rate_bpm,
                    "discrepancy_bpm": response.rate_discrepancy_bpm,
                    "pat_ms": response.pulse_arrival_time_ms,
                },
                session_id=session_id,
                now_epoch=now_epoch,
            )
            if alert:
                triggered_alerts.append(alert)
                self._store_and_dispatch(alert)

        return triggered_alerts

    def _create_alert_if_ready(
        self,
        alert_type: AlertTypeEnum,
        severity: AlertSeverityEnum,
        message: str,
        metrics: Dict[str, Any],
        session_id: Optional[str],
        now_epoch: float,
    ) -> Optional[CardiacAlert]:
        """Debounce alerts against cooldown period unless severity has escalated."""
        type_key = alert_type.value
        last_state = self._last_alert_state.get(type_key)

        if last_state is not None:
            last_epoch, last_sev = last_state
            elapsed = now_epoch - last_epoch
            # Suppress if within cooldown period, unless escalating from WARNING to CRITICAL
            escalating = (last_sev == AlertSeverityEnum.WARNING and severity == AlertSeverityEnum.CRITICAL)
            if elapsed < self.config.cooldown_seconds and not escalating:
                return None

        # Update last alert state
        self._last_alert_state[type_key] = (now_epoch, severity)

        return CardiacAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            severity=severity,
            alert_type=alert_type,
            message=message,
            metrics=metrics,
            session_id=session_id,
            acknowledged=False,
            disclaimer="Research prototype — not intended for medical diagnosis.",
        )

    def _store_and_dispatch(self, alert: CardiacAlert) -> None:
        """Store alert in history and dispatch to subscribers."""
        self._alerts.insert(0, alert)  # newest first
        if len(self._alerts) > 500:
            self._alerts.pop()

        for sub in list(self._subscribers):
            try:
                sub(alert)
            except Exception as e:
                print(f"[AlertService dispatch error: {e}]")

    def list_alerts(
        self,
        severity: Optional[AlertSeverityEnum] = None,
        alert_type: Optional[AlertTypeEnum] = None,
        unacknowledged_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> AlertListResponse:
        """Query alert history with optional filtering."""
        filtered = self._alerts

        if severity is not None:
            filtered = [a for a in filtered if a.severity == severity]
        if alert_type is not None:
            filtered = [a for a in filtered if a.alert_type == alert_type]
        if unacknowledged_only:
            filtered = [a for a in filtered if not a.acknowledged]

        total = len(filtered)
        unack_count = sum(1 for a in self._alerts if not a.acknowledged)
        paginated = filtered[offset : offset + limit]

        return AlertListResponse(
            total=total,
            unacknowledged_count=unack_count,
            alerts=paginated,
        )

    def get_active_alerts(self) -> List[CardiacAlert]:
        """Retrieve all currently active unacknowledged alerts."""
        return [a for a in self._alerts if not a.acknowledged]

    def acknowledge_alert(self, alert_id: str) -> Optional[AlertAcknowledgeResponse]:
        """Acknowledge a specific alert incident."""
        for a in self._alerts:
            if a.alert_id == alert_id:
                a.acknowledged = True
                a.acknowledged_at = datetime.now(timezone.utc).isoformat()
                return AlertAcknowledgeResponse(
                    alert_id=alert_id,
                    acknowledged=True,
                    acknowledged_at=a.acknowledged_at,
                    message="Alert acknowledged successfully.",
                )
        return None
