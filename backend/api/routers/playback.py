"""Simulated real-time ECG playback endpoints.

Enables browser viewers and dashboards to stream playback frames and manage playback states.
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, Body, Depends, Query
from fastapi.responses import JSONResponse

from backend.api.deps import get_playback_engine
from ml.ecg.playback import ECGPlaybackEngine, PlaybackState

router = APIRouter(prefix="/playback", tags=["Playback"])


@router.get("/frame")
def get_playback_frame(engine: ECGPlaybackEngine = Depends(get_playback_engine)) -> Dict[str, Any]:
    """Retrieve the next simulated real-time ECG playback frame."""
    frame = engine.next_frame()
    if frame is None and engine.state == PlaybackState.PAUSED:
        # Step one frame forward if paused
        engine.state = PlaybackState.PLAYING
        frame = engine.next_frame()
        engine.state = PlaybackState.PAUSED

    if frame:
        return frame.to_dict()

    return {"status": "no_data", "state": engine.state.value}


@router.post("/control")
def control_playback(
    action: str = Query(..., description="Action: play, resume, pause, reset, speed, step"),
    payload: Optional[Dict[str, Any]] = Body(default=None),
    engine: ECGPlaybackEngine = Depends(get_playback_engine),
) -> Dict[str, Any]:
    """Control playback engine: pause, resume, reset, speed, step."""
    action = action.lower()
    if action in ("pause",):
        engine.pause()
    elif action in ("resume", "play"):
        engine.resume()
    elif action in ("reset",):
        engine.reset()
    elif action in ("step",):
        pass
    elif action in ("speed",):
        speed = float(payload.get("speed", 1.0)) if payload else 1.0
        engine.set_speed(speed)

    return {"status": "ok", "state": engine.state.value, "speed": engine.playback_speed}
