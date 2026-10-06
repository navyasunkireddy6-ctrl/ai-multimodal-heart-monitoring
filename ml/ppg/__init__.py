"""PPG Signal Processing Package.

Exports core processing and synthetic waveform utilities for optical PPG.
"""

from ml.ppg.processor import (
    PPGProcessor,
    PPGAnalysisResult,
)
from ml.ppg.synthetic import (
    generate_synthetic_ppg,
)

__all__ = [
    "PPGProcessor",
    "PPGAnalysisResult",
    "generate_synthetic_ppg",
]
