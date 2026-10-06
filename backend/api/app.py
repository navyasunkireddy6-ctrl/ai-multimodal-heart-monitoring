"""FastAPI application factory and configuration.

Mounts all canonical REST routers under /api/v1 and WebSocket streaming endpoints
under /ws/v1. Configures CORS middleware and server-level metadata.
"""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from backend.api.routers import (
    health_router,
    ecg_router,
    ppg_router,
    trend_router,
    sessions_router,
    playback_router,
    websocket_router,
)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DASHBOARD_HTML = ROOT_DIR / "dashboard" / "phase4_viewer.html"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Multimodal Real-Time Cardiac Monitoring API",
        description="Research prototype backend combining ECG and smartphone optical PPG with machine learning.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Enable full Cross-Origin Resource Sharing (CORS) for web dashboard & mobile clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount canonical REST routers under base path /api/v1
    api_v1_prefix = "/api/v1"
    app.include_router(health_router, prefix=api_v1_prefix)
    app.include_router(ecg_router, prefix=api_v1_prefix)
    app.include_router(ppg_router, prefix=api_v1_prefix)
    app.include_router(trend_router, prefix=api_v1_prefix)
    app.include_router(sessions_router, prefix=api_v1_prefix)
    app.include_router(playback_router, prefix=api_v1_prefix)

    # Mount live streaming WebSocket router
    app.include_router(websocket_router)

    # Informational root and viewer routes
    @app.get("/", response_class=HTMLResponse)
    def index():
        """Serve the ECG/PPG dashboard or API metadata."""
        if DASHBOARD_HTML.exists():
            return DASHBOARD_HTML.read_text(encoding="utf-8")
        return HTMLResponse(
            "<h3>Multimodal Real-Time Cardiac Monitoring API</h3><p>Navigate to <a href='/docs'>/docs</a> for Swagger UI.</p>"
        )

    @app.get("/api")
    def api_meta():
        return JSONResponse({
            "service": "Multimodal Real-Time Cardiac Monitoring API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/api/v1/health",
                "ecg_process": "/api/v1/ecg/process",
                "ecg_classify": "/api/v1/ecg/classify",
                "ppg_process": "/api/v1/ppg/process",
                "ppg_quality": "/api/v1/ppg/quality",
                "trend_predict": "/api/v1/trend/predict",
                "sessions": "/api/v1/sessions",
                "playback": "/api/v1/playback/frame",
                "websocket_ppg": "/ws/v1/ppg",
            },
            "disclaimer": "Research prototype — not intended for medical diagnosis.",
        })

    return app


app = create_app()
