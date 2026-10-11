"""Pytest configuration and environment fixtures for BioQuest.

Provides compatibility workarounds for Windows Smart App Control blocking
unsigned C-extension libraries in scikit-learn.
"""

import sys
from unittest.mock import MagicMock

# Windows Smart App Control may block unsigned C-extension DLLs in scikit-learn
# which are only used by radius neighbors pairwise reductions (not used in our RF model).
try:
    import sklearn.metrics._pairwise_distances_reduction._radius_neighbors_classmode
except Exception:
    sys.modules.setdefault(
        "sklearn.metrics._pairwise_distances_reduction._radius_neighbors_classmode",
        MagicMock(),
    )
