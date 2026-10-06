"""Trend Analysis Package.

Exports rate trajectory prediction and linear trend analyzer.
"""

from ml.trend.analyzer import (
    TrendAnalyzer,
    TrendAnalysisResult,
    MODEL_VERSION,
)

__all__ = [
    "TrendAnalyzer",
    "TrendAnalysisResult",
    "MODEL_VERSION",
]
