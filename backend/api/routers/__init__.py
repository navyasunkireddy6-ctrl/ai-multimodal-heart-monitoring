"""API routers package."""

from backend.api.routers.health import router as health_router
from backend.api.routers.ecg import router as ecg_router
from backend.api.routers.ppg import router as ppg_router
from backend.api.routers.trend import router as trend_router
from backend.api.routers.sessions import router as sessions_router
from backend.api.routers.playback import router as playback_router
from backend.api.routers.websocket import router as websocket_router

__all__ = [
    "health_router",
    "ecg_router",
    "ppg_router",
    "trend_router",
    "sessions_router",
    "playback_router",
    "websocket_router",
]
