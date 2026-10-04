"""Lightweight Web Viewer Server for Phase 4 Simulated Real-Time ECG.

Run with:
    python scripts/serve_phase4_viewer.py
Then open:
    http://127.0.0.1:8000
"""

import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, Query, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from ml.ecg.playback import ECGPlaybackEngine, PlaybackState

app = FastAPI(title="ECG Simulated Real-Time Viewer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global playback engine initialized with Record 100
engine = ECGPlaybackEngine(record_id="100", window_sec=5.0, step_sec=1.0, playback_speed=1.0)
engine.start()

HTML_FILE = ROOT_DIR / "dashboard" / "phase4_viewer.html"


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the interactive web viewer."""
    if HTML_FILE.exists():
        return HTML_FILE.read_text(encoding="utf-8")
    return HTMLResponse("<h3>Error: dashboard/phase4_viewer.html not found</h3>", status_code=404)


@app.get("/api/v1/playback/frame")
def get_frame():
    """Return next simulated real-time ECG frame."""
    frame = engine.next_frame()
    if frame is None and engine.state == PlaybackState.PAUSED:
        # If paused, temporarily advance or return last known
        engine.state = PlaybackState.PLAYING
        frame = engine.next_frame()
        engine.state = PlaybackState.PAUSED

    if frame:
        return frame.to_dict()
    return JSONResponse({"status": "no_data", "state": engine.state.value})


@app.post("/api/v1/playback/control")
def control_playback(action: str = Query(...), payload: Optional[dict] = Body(default=None)):
    """Handle UI actions: pause, resume, reset, speed, step."""
    action = action.lower()
    if action == "pause":
        engine.pause()
    elif action == "resume" or action == "play":
        engine.resume()
    elif action == "reset":
        engine.reset()
    elif action == "step":
        pass  # next frame fetch will naturally step
    elif action == "speed":
        speed = float(payload.get("speed", 1.0)) if payload else 1.0
        engine.set_speed(speed)

    return {"status": "ok", "state": engine.state.value, "speed": engine.playback_speed}


def main():
    print("=" * 70)
    print("  PHASE 4 — ECG SIMULATED REAL-TIME BROWSER VIEWER")
    print("  Opening URL: http://127.0.0.1:8000")
    print("  Press Ctrl+C to terminate.")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
