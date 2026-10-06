"""Short-Term Heart Rate and Pulse Rate Trend Analysis.

Evaluates rate history sequences over moving time windows to determine rate trajectory
(INCREASING, STABLE, DECREASING) with linear slope and statistical confidence.

Aligned with docs/API_CONTRACT.md (model_version: trend-rule-v1.0.0).

Disclaimer:
    Research prototype — not intended for medical diagnosis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from backend.models.schemas import TrendLabelEnum

MODEL_VERSION = "trend-rule-v1.0.0"


@dataclass
class TrendAnalysisResult:
    """Structured output from rate trend trajectory analysis."""
    trend: str
    slope_bpm_per_min: float
    confidence: float
    window_duration_seconds: float
    model_version: str = MODEL_VERSION
    disclaimer: str = "Research prototype — not intended for medical diagnosis."

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "trend": self.trend,
            "slope_bpm_per_min": round(float(self.slope_bpm_per_min), 2),
            "confidence": round(float(self.confidence), 4),
            "window_duration_seconds": round(float(self.window_duration_seconds), 1),
            "model_version": self.model_version,
            "disclaimer": self.disclaimer,
        }


class TrendAnalyzer:
    """Evaluates short-term rate history to determine trajectory and rate of change."""

    def __init__(
        self,
        slope_threshold_bpm_per_min: float = 3.0,
        model_version: str = MODEL_VERSION,
    ):
        self.slope_threshold = float(slope_threshold_bpm_per_min)
        self.model_version = model_version

    @staticmethod
    def _parse_timestamp(ts: str | float | int | datetime) -> float:
        """Parse timestamp input into POSIX seconds float."""
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, datetime):
            return ts.timestamp()
        if isinstance(ts, str):
            # Normalize trailing Z to +00:00 for ISO-8601 parsing
            clean = ts.strip().replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(clean)
                return dt.timestamp()
            except ValueError:
                # Fallback to float conversion if numeric string
                try:
                    return float(ts)
                except ValueError:
                    raise ValueError(f"Unparseable ISO-8601 timestamp string: '{ts}'")
        raise TypeError(f"Unsupported timestamp type: {type(ts)}")

    def analyze(
        self,
        timestamps: List[str | float | datetime],
        rates: List[float],
    ) -> TrendAnalysisResult:
        """Analyze rate sequence over time window.
        
        Args:
            timestamps: List of measurement timestamps (ISO-8601 strings or numeric epoch seconds).
            rates: List of heart rate / pulse rate measurements in BPM.
            
        Returns:
            TrendAnalysisResult containing trend label, slope (BPM/min), confidence, and window duration.
        """
        if not timestamps or not rates:
            return TrendAnalysisResult(
                trend=TrendLabelEnum.STABLE.value,
                slope_bpm_per_min=0.0,
                confidence=0.50,
                window_duration_seconds=0.0,
                model_version=self.model_version,
            )

        # 1. Clean, pair, and sort data
        pairs = []
        for ts, r in zip(timestamps, rates):
            if r is None:
                continue
            try:
                t_sec = self._parse_timestamp(ts)
                r_val = float(r)
                if np.isfinite(t_sec) and np.isfinite(r_val) and 30.0 <= r_val <= 250.0:
                    pairs.append((t_sec, r_val))
            except Exception:
                continue

        if len(pairs) == 0:
            return TrendAnalysisResult(
                trend=TrendLabelEnum.STABLE.value,
                slope_bpm_per_min=0.0,
                confidence=0.50,
                window_duration_seconds=0.0,
                model_version=self.model_version,
            )

        # Sort chronologically
        pairs.sort(key=lambda p: p[0])
        t_arr = np.array([p[0] for p in pairs], dtype=np.float64)
        y_arr = np.array([p[1] for p in pairs], dtype=np.float64)

        t_rel = t_arr - t_arr[0]
        duration_sec = float(t_arr[-1] - t_arr[0])
        n = len(pairs)

        # 2. Single observation or zero time span
        if n < 2 or duration_sec < 1e-3:
            return TrendAnalysisResult(
                trend=TrendLabelEnum.STABLE.value,
                slope_bpm_per_min=0.0,
                confidence=0.75,
                window_duration_seconds=duration_sec,
                model_version=self.model_version,
            )

        # 3. Linear Regression Fit
        # y = slope_sec * t_rel + intercept
        t_mean = float(np.mean(t_rel))
        y_mean = float(np.mean(y_arr))

        s_tt = float(np.sum((t_rel - t_mean) ** 2))
        s_ty = float(np.sum((t_rel - t_mean) * (y_arr - y_mean)))
        s_yy = float(np.sum((y_arr - y_mean) ** 2))

        if s_tt > 1e-6:
            slope_sec = s_ty / s_tt
            slope_min = slope_sec * 60.0
        else:
            slope_sec = 0.0
            slope_min = 0.0

        # R-squared (coefficient of determination)
        r_sq = (s_ty ** 2) / (s_tt * s_yy + 1e-12) if s_yy > 1e-6 else 1.0
        r_sq = float(np.clip(r_sq, 0.0, 1.0))

        # Standard deviation of rates
        std_y = float(np.std(y_arr))

        # 4. Trajectory Classification
        if slope_min > self.slope_threshold:
            trend_label = TrendLabelEnum.INCREASING.value
            # Confidence combines goodness of linear fit and sample size
            conf = 0.50 + 0.45 * r_sq * min(1.0, n / 4.0)
        elif slope_min < -self.slope_threshold:
            trend_label = TrendLabelEnum.DECREASING.value
            conf = 0.50 + 0.45 * r_sq * min(1.0, n / 4.0)
        else:
            trend_label = TrendLabelEnum.STABLE.value
            # For stable trajectory, confidence reflects low rate variance
            variance_penalty = min(0.40, std_y / 15.0)
            conf = 0.95 - variance_penalty

        conf = float(np.clip(conf, 0.50, 0.98))

        return TrendAnalysisResult(
            trend=trend_label,
            slope_bpm_per_min=round(slope_min, 2),
            confidence=round(conf, 4),
            window_duration_seconds=round(duration_sec, 2),
            model_version=self.model_version,
        )
