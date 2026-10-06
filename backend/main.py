"""Main entrypoint for running the Multimodal Cardiac Monitoring FastAPI backend."""

import uvicorn
from backend.api.app import app


def main():
    print("=" * 80)
    print("  MULTIMODAL REAL-TIME CARDIAC MONITORING SYSTEM — BACKEND API")
    print("  Version: 1.0.0")
    print("  Documentation: http://127.0.0.1:8000/docs")
    print("  Live WebSocket: ws://127.0.0.1:8000/ws/v1/ppg")
    print("  Research prototype — not intended for medical diagnosis.")
    print("=" * 80)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
