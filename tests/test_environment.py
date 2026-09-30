"""Test environment, dependencies, and project directory structure."""

import sys
import sqlite3
from pathlib import Path
import pytest


def test_python_version():
    """Verify Python version is 3.11 or greater."""
    assert sys.version_info >= (3, 11), f"Python version {sys.version} is lower than 3.11"


def test_required_dependencies():
    """Verify all critical framework and scientific packages can be imported."""
    import fastapi
    import uvicorn
    import pydantic
    import websockets
    import wfdb
    import numpy as np
    import scipy
    import pandas as pd
    import sklearn
    import joblib
    import httpx

    assert fastapi.__version__ is not None
    assert pydantic.__version__ is not None
    assert np.__version__ is not None
    assert scipy.__version__ is not None
    assert pd.__version__ is not None
    assert sklearn.__version__ is not None
    assert wfdb.__version__ is not None


def test_sqlite_availability():
    """Verify built-in SQLite database support is functioning."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT);")
    cursor.execute("INSERT INTO test (name) VALUES ('cardiac');")
    cursor.execute("SELECT name FROM test WHERE id = 1;")
    result = cursor.fetchone()
    conn.close()
    assert result == ("cardiac",)


def test_project_structure():
    """Verify that required directory structure exists."""
    expected_dirs = [
        "backend/api",
        "backend/models",
        "backend/services",
        "ml/ecg",
        "ml/ppg",
        "ml/trend",
        "data/ecg",
        "data/ppg",
        "models",
        "dashboard",
        "android",
        "tests",
        "scripts",
        "docs",
    ]
    for d in expected_dirs:
        dir_path = Path(d)
        assert dir_path.exists(), f"Missing required directory: {d}"
        assert dir_path.is_dir(), f"Expected directory but found file: {d}"
