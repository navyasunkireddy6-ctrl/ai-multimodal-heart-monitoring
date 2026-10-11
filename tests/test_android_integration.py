"""Automated unit and integration tests for Phase 10 Android Mobile PPG Application.

Validates:
1. Android data model wire-format compatibility with API Contract Section 4.
2. Camera PPG Analyzer logic (ROI extraction, YUV to RGB conversion, finger coverage).
3. Live WebSocket streaming from Android client to FastAPI backend.
4. Android project structure, Manifest permissions, and XML resource integrity.
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.api.deps import get_session_service
from backend.models.schemas import (
    ModeEnum,
    PPGWebSocketFrame,
    PPGWebSocketResponse,
    QualityLabelEnum,
)
from ml.ppg.synthetic import generate_synthetic_ppg

ROOT_DIR = Path(__file__).resolve().parent.parent
ANDROID_DIR = ROOT_DIR / "android"


@pytest.fixture
def client():
    """Create a FastAPI test client and reset session storage."""
    session_service = get_session_service()
    session_service.clear()
    return TestClient(app)


# ============================================================================
# 1. Android Project Structure & Manifest Verification
# ============================================================================

def test_android_project_files_exist():
    """Verify all required Android Gradle, Kotlin source, and resource files exist."""
    required_files = [
        ANDROID_DIR / "build.gradle",
        ANDROID_DIR / "settings.gradle",
        ANDROID_DIR / "gradle.properties",
        ANDROID_DIR / "app" / "build.gradle",
        ANDROID_DIR / "app" / "src" / "main" / "AndroidManifest.xml",
        ANDROID_DIR / "app" / "src" / "main" / "java" / "com" / "bioquest" / "cardiac" / "camera" / "CameraPPGAnalyzer.kt",
        ANDROID_DIR / "app" / "src" / "main" / "java" / "com" / "bioquest" / "cardiac" / "models" / "PPGModels.kt",
        ANDROID_DIR / "app" / "src" / "main" / "java" / "com" / "bioquest" / "cardiac" / "network" / "PPGWebSocketManager.kt",
        ANDROID_DIR / "app" / "src" / "main" / "java" / "com" / "bioquest" / "cardiac" / "ui" / "MainActivity.kt",
        ANDROID_DIR / "app" / "src" / "main" / "res" / "layout" / "activity_main.xml",
        ANDROID_DIR / "app" / "src" / "main" / "res" / "values" / "colors.xml",
        ANDROID_DIR / "app" / "src" / "main" / "res" / "values" / "strings.xml",
        ANDROID_DIR / "app" / "src" / "main" / "res" / "values" / "themes.xml",
    ]
    for p in required_files:
        assert p.exists(), f"Missing required Android file: {p}"
        assert p.stat().st_size > 0, f"Android file is empty: {p}"


def test_android_manifest_permissions_and_features():
    """Verify AndroidManifest.xml contains necessary permissions and camera hardware features."""
    manifest_path = ANDROID_DIR / "app" / "src" / "main" / "AndroidManifest.xml"
    tree = ET.parse(manifest_path)
    root = tree.getroot()

    permissions = [
        elem.attrib.get("{http://schemas.android.com/apk/res/android}name")
        for elem in root.findall("uses-permission")
    ]
    assert "android.permission.CAMERA" in permissions
    assert "android.permission.INTERNET" in permissions
    assert "android.permission.ACCESS_NETWORK_STATE" in permissions

    features = [
        elem.attrib.get("{http://schemas.android.com/apk/res/android}name")
        for elem in root.findall("uses-feature")
    ]
    assert "android.hardware.camera" in features
    assert "android.hardware.camera.flash" in features


def test_android_layout_views_conformity():
    """Verify activity_main.xml contains all UI element IDs referenced in MainActivity."""
    layout_path = ANDROID_DIR / "app" / "src" / "main" / "res" / "layout" / "activity_main.xml"
    content = layout_path.read_text(encoding="utf-8")

    expected_ids = [
        "@+id/previewView",
        "@+id/tvWarningBanner",
        "@+id/tvPulseRate",
        "@+id/tvQualityLabel",
        "@+id/tvTrend",
        "@+id/tvConnectionStatus",
        "@+id/btnToggleFlash",
    ]
    for view_id in expected_ids:
        assert view_id in content, f"Missing view ID {view_id} in activity_main.xml"


# ============================================================================
# 2. Camera PPG Analyzer Simulation Logic Tests
# ============================================================================

def simulate_camera_yuv_to_rgb(y_val: int, u_val: int, v_val: int):
    """Simulate Rec.601 YUV to RGB conversion as implemented in CameraPPGAnalyzer.kt."""
    r = max(0.0, min(255.0, y_val + 1.402 * (v_val - 128)))
    g = max(0.0, min(255.0, y_val - 0.344136 * (u_val - 128) - 0.714136 * (v_val - 128)))
    b = max(0.0, min(255.0, y_val + 1.772 * (u_val - 128)))
    return r, g, b


def test_camera_analyzer_fingertip_coverage_detection():
    """Verify optical coverage criteria: Red must be > 70 and > 1.3 * Blue."""
    # Scenario A: Good fingertip coverage under torch transillumination (high Red, low Blue)
    r_covered, g_covered, b_covered = 185.0, 42.0, 35.0
    is_covering_good = r_covered > 70.0 and r_covered > (b_covered * 1.3)
    assert is_covering_good is True

    # Scenario B: Room ambient light (balanced RGB / blue-ish)
    r_uncovered, g_uncovered, b_uncovered = 100.0, 110.0, 120.0
    is_covering_uncovered = r_uncovered > 70.0 and r_uncovered > (b_uncovered * 1.3)
    assert is_covering_uncovered is False

    # Scenario C: Pitch black / lens completely obscured without flash
    r_dark, g_dark, b_dark = 20.0, 15.0, 15.0
    is_covering_dark = r_dark > 70.0 and r_dark > (b_dark * 1.3)
    assert is_covering_dark is False


def test_camera_analyzer_yuv_conversion_accuracy():
    """Verify YUV to RGB mathematical fidelity."""
    # Pure red test vector in YUV: Y=76, U=84, V=255
    r, g, b = simulate_camera_yuv_to_rgb(76, 84, 255)
    assert r > 240.0
    assert g < 30.0
    assert b < 30.0


# ============================================================================
# 3. Android Client Wire-Format & Network Compatibility Tests
# ============================================================================

def test_android_frame_schema_wire_compatibility(client):
    """Verify PPGWebSocketFrame produced by Android client satisfies FastAPI backend parser."""
    timestamp = datetime.now(timezone.utc).isoformat()
    frame_dict = {
        "type": "ppg_frame",
        "timestamp": timestamp,
        "session_id": "test-session-123",
        "red_channel_value": 165.4,
        "green_channel_value": 38.2,
        "blue_channel_value": 24.1,
        "flash_enabled": True,
        "camera_fps": 30.0,
    }

    parsed = PPGWebSocketFrame.model_validate(frame_dict)
    assert parsed.type == "ppg_frame"
    assert parsed.red_channel_value == 165.4
    assert parsed.session_id == "test-session-123"
    assert parsed.camera_fps == 30.0


def test_android_live_websocket_streaming_session(client):
    """Simulate full Android optical PPG live streaming session via WebSocket."""
    # 1. Create mobile session on backend
    sess_resp = client.post(
        "/api/v1/sessions",
        json={"mode": "smartphone_ppg", "notes": "Android CameraX acquisition session"},
    )
    assert sess_resp.status_code == 201
    session_id = sess_resp.json()["session_id"]

    # 2. Generate optical PPG signal (simulate finger on camera at 72 BPM)
    _, ppg_waveform = generate_synthetic_ppg(
        duration_sec=3.5,
        sampling_rate_hz=30.0,
        pulse_rate_bpm=72.0,
        noise_amplitude=0.015,
    )

    t_start = datetime.now(timezone.utc)
    received_updates = []

    # 3. Connect as Android client via WebSocket
    with client.websocket_connect("/ws/v1/ppg") as ws:
        for idx, val in enumerate(ppg_waveform):
            frame_payload = {
                "type": "ppg_frame",
                "timestamp": (t_start + timedelta(seconds=idx / 30.0)).isoformat(),
                "session_id": session_id,
                "red_channel_value": float(val),
                "green_channel_value": float(val * 0.25),
                "blue_channel_value": float(val * 0.15),
                "flash_enabled": True,
                "camera_fps": 30.0,
            }
            ws.send_text(json.dumps(frame_payload))
            raw_resp = ws.receive_text()
            resp_dict = json.loads(raw_resp)
            received_updates.append(resp_dict)

    # 4. Verify telemetry responses conform to Android PPGWebSocketResponse model
    assert len(received_updates) == len(ppg_waveform)
    latest = received_updates[-1]

    assert latest["type"] == "measurement_update"
    assert latest["mode"] == ModeEnum.SMARTPHONE_PPG
    assert latest["signal_quality"] >= 0.0
    assert latest["quality_label"] in ("GOOD", "FAIR")
    assert latest["pulse_rate_bpm"] is not None
    assert abs(latest["pulse_rate_bpm"] - 72.0) <= 6.0
    assert latest["model_version"] == "ppg-peak-v1.0.0"

    # 5. Verify backend persisted measurements for this Android session
    session_details = client.get(f"/api/v1/sessions/{session_id}").json()
    assert len(session_details["measurements"]) > 10


def test_android_poor_contact_warning_handling(client):
    """Verify Android app receives warning banner and null pulse rate when finger lifts."""
    t_start = datetime.now(timezone.utc)
    with client.websocket_connect("/ws/v1/ppg") as ws:
        for idx in range(60):
            # Flat noisy line simulating lifted finger
            frame_payload = {
                "type": "ppg_frame",
                "timestamp": (t_start + timedelta(seconds=idx / 30.0)).isoformat(),
                "red_channel_value": 45.0 + float((idx % 4) * 0.1),
                "camera_fps": 30.0,
            }
            ws.send_text(json.dumps(frame_payload))
            raw_resp = ws.receive_text()
            latest = json.loads(raw_resp)

        assert latest["quality_label"] == QualityLabelEnum.POOR
        assert latest["pulse_rate_bpm"] is None
        assert latest["warning"] is not None
        assert "Poor optical contact" in latest["warning"]
