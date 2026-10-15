"""Main entrypoint for running the Multimodal Cardiac Monitoring FastAPI backend."""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn
from backend.api.app import app



def main():
    print("=" * 80)
    print("  MULTIMODAL REAL-TIME CARDIAC MONITORING SYSTEM — BACKEND API")
    print("  Version: 1.0.0")
    print("  Host: 0.0.0.0 (Localhost & Mobile Emulator / LAN)")
    print("  Documentation: http://127.0.0.1:8000/docs")
    print("  Live WebSocket: ws://127.0.0.1:8000/ws/v1/ppg (or ws://10.0.2.2:8000/ws/v1/ppg for Android Emulator)")
    print("  Research prototype — not intended for medical diagnosis.")
    print("=" * 80)
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
