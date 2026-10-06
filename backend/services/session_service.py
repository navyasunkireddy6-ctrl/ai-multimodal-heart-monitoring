"""Service managing cardiac monitoring sessions and historical measurements.

Conforms strictly to docs/API_CONTRACT.md and canonical Pydantic schemas.
Provides thread-safe in-memory session persistence, measurement aggregation,
and CSV export functionality.
"""

import csv
import io
import threading
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.models.schemas import (
    CanonicalMeasurement,
    ModeEnum,
    SessionCreateRequest,
    SessionDetailResponse,
    SessionListResponse,
    SessionResponse,
)


class SessionRecord:
    """Internal container for a recorded session."""

    def __init__(self, session_id: str, mode: ModeEnum, notes: Optional[str] = None):
        self.session_id = session_id
        self.mode = mode
        self.notes = notes
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.measurements: List[CanonicalMeasurement] = []

    @property
    def avg_heart_rate_bpm(self) -> Optional[float]:
        hr_vals = [m.heart_rate_bpm for m in self.measurements if m.heart_rate_bpm is not None]
        return round(float(sum(hr_vals) / len(hr_vals)), 1) if hr_vals else None

    @property
    def avg_pulse_rate_bpm(self) -> Optional[float]:
        pr_vals = [m.pulse_rate_bpm for m in self.measurements if m.pulse_rate_bpm is not None]
        return round(float(sum(pr_vals) / len(pr_vals)), 1) if pr_vals else None

    def to_summary(self) -> SessionResponse:
        return SessionResponse(
            session_id=self.session_id,
            created_at=self.created_at,
            mode=self.mode,
            measurement_count=len(self.measurements),
            avg_heart_rate_bpm=self.avg_heart_rate_bpm,
            avg_pulse_rate_bpm=self.avg_pulse_rate_bpm,
            notes=self.notes,
        )

    def to_detail(self) -> SessionDetailResponse:
        return SessionDetailResponse(
            session_id=self.session_id,
            created_at=self.created_at,
            mode=self.mode,
            notes=self.notes,
            measurements=list(self.measurements),
        )


class SessionService:
    """Thread-safe session repository and measurement manager."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionRecord] = {}

    def create_session(self, request: SessionCreateRequest) -> SessionResponse:
        """Create and initialize a new monitoring session."""
        session_id = str(uuid.uuid4())
        record = SessionRecord(
            session_id=session_id,
            mode=request.mode,
            notes=request.notes,
        )
        with self._lock:
            self._sessions[session_id] = record

        return record.to_summary()

    def list_sessions(self, limit: int = 50, offset: int = 0) -> SessionListResponse:
        """Retrieve paginated list of sessions ordered by creation date descending."""
        with self._lock:
            records = list(self._sessions.values())

        # Sort reverse chronological
        records.sort(key=lambda r: r.created_at, reverse=True)
        total = len(records)
        paged = records[offset : offset + limit]

        return SessionListResponse(
            total=total,
            sessions=[r.to_summary() for r in paged],
        )

    def get_session(self, session_id: str) -> Optional[SessionDetailResponse]:
        """Fetch complete session details including measurement history."""
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return None
            return record.to_detail()

    def delete_session(self, session_id: str) -> bool:
        """Delete session if found."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def add_measurement(self, session_id: str, measurement: CanonicalMeasurement) -> bool:
        """Append a canonical measurement to an existing session."""
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return False
            record.measurements.append(measurement)
            return True

    def export_csv(self, session_id: str) -> Optional[str]:
        """Export session measurements as a CSV string conforming to the canonical schema."""
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return None
            measurements = list(record.measurements)

        output = io.StringIO()
        fieldnames = [
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
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for m in measurements:
            writer.writerow({
                "timestamp": m.timestamp,
                "mode": m.mode if isinstance(m.mode, str) else m.mode.value,
                "heart_rate_bpm": m.heart_rate_bpm if m.heart_rate_bpm is not None else "",
                "pulse_rate_bpm": m.pulse_rate_bpm if m.pulse_rate_bpm is not None else "",
                "signal_quality": m.signal_quality,
                "quality_label": m.quality_label if isinstance(m.quality_label, str) else m.quality_label.value,
                "rhythm_class": m.rhythm_class or "",
                "rhythm_confidence": m.rhythm_confidence if m.rhythm_confidence is not None else "",
                "trend": (m.trend if isinstance(m.trend, str) else (m.trend.value if m.trend else "")) or "",
                "model_version": m.model_version,
            })

        return output.getvalue()

    def clear(self) -> None:
        """Clear all sessions (used for test teardown)."""
        with self._lock:
            self._sessions.clear()
