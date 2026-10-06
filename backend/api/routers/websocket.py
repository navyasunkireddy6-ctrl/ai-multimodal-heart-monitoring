"""Live PPG streaming WebSocket endpoint conforming to Section 4 of docs/API_CONTRACT.md."""

import json
from collections import deque
from datetime import datetime, timezone
from typing import Deque, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.deps import get_session_service
from backend.models.schemas import (
    CanonicalMeasurement,
    ModeEnum,
    PPGWebSocketFrame,
    PPGWebSocketResponse,
    QualityLabelEnum,
    TrendLabelEnum,
)
from ml.ppg.processor import PPGProcessor
from ml.trend.analyzer import TrendAnalyzer

router = APIRouter(tags=["WebSocket"])

PPG_MODEL_VERSION = "ppg-peak-v1.0.0"


class PPGStreamHandler:
    """Manages sliding window signal buffering, quality estimation, and trend analysis for a WebSocket stream."""

    def __init__(self, window_size: int = 90, min_samples: int = 45):
        self.window_size = window_size
        self.min_samples = min_samples
        self.raw_buffer: Deque[float] = deque(maxlen=window_size)
        self.time_buffer: Deque[str] = deque(maxlen=window_size)
        self.recent_rates: Deque[float] = deque(maxlen=20)
        self.recent_rate_times: Deque[str] = deque(maxlen=20)
        self.processor = PPGProcessor(sampling_rate_hz=30.0)
        self.trend_analyzer = TrendAnalyzer(slope_threshold_bpm_per_min=3.0)

    def process_frame(
        self,
        frame: PPGWebSocketFrame,
        session_service,
    ) -> PPGWebSocketResponse:
        """Process an inbound optical PPG frame and compute real-time cardiac metrics."""
        self.raw_buffer.append(float(frame.red_channel_value))
        self.time_buffer.append(frame.timestamp)

        # Update processor sampling rate if camera FPS specified
        if frame.camera_fps and abs(frame.camera_fps - self.processor.sampling_rate_hz) > 1.0:
            self.processor = PPGProcessor(sampling_rate_hz=float(frame.camera_fps))

        # Check if enough samples for reliable physiological analysis
        if len(self.raw_buffer) < self.min_samples:
            # Buffer filling phase: emit baseline update
            return PPGWebSocketResponse(
                type="measurement_update",
                timestamp=frame.timestamp,
                mode=ModeEnum.SMARTPHONE_PPG,
                heart_rate_bpm=None,
                pulse_rate_bpm=None,
                signal_quality=0.50,
                quality_label=QualityLabelEnum.FAIR,
                rhythm_class=None,
                rhythm_confidence=None,
                trend=None,
                model_version=PPG_MODEL_VERSION,
                raw_red=frame.red_channel_value,
                filtered_value=0.0,
                warning="Acquiring optical pulse baseline...",
            )

        signal_array = list(self.raw_buffer)
        result = self.processor.process(signal_array)

        filtered_val = (
            float(result.filtered_signal[-1])
            if len(result.filtered_signal) > 0
            else None
        )

        quality_enum = QualityLabelEnum(result.quality_label)
        pulse_rate: Optional[float] = None
        warning: Optional[str] = None
        trend_label: Optional[TrendLabelEnum] = None

        if quality_enum == QualityLabelEnum.POOR:
            pulse_rate = None
            warning = "Poor optical contact. Cover camera and flash steadily."
        else:
            pulse_rate = (
                round(float(result.pulse_rate_bpm), 1)
                if result.pulse_rate_bpm is not None
                else None
            )

            # If valid pulse rate, update rate trajectory history
            if pulse_rate is not None:
                self.recent_rates.append(pulse_rate)
                self.recent_rate_times.append(frame.timestamp)

                if len(self.recent_rates) >= 3:
                    try:
                        t_res = self.trend_analyzer.analyze(
                            timestamps=list(self.recent_rate_times),
                            rates=list(self.recent_rates),
                        )
                        trend_label = TrendLabelEnum(t_res.trend)
                    except Exception:
                        trend_label = TrendLabelEnum.STABLE

        response = PPGWebSocketResponse(
            type="measurement_update",
            timestamp=frame.timestamp,
            mode=ModeEnum.SMARTPHONE_PPG,
            heart_rate_bpm=None,
            pulse_rate_bpm=pulse_rate,
            signal_quality=round(float(result.signal_quality), 4),
            quality_label=quality_enum,
            rhythm_class=None,
            rhythm_confidence=None,
            trend=trend_label,
            model_version=PPG_MODEL_VERSION,
            raw_red=round(float(frame.red_channel_value), 2),
            filtered_value=round(filtered_val, 4) if filtered_val is not None else None,
            warning=warning,
        )

        # Record to active session if session_id provided and exists
        if frame.session_id:
            meas = CanonicalMeasurement(
                timestamp=frame.timestamp,
                mode=ModeEnum.SMARTPHONE_PPG,
                heart_rate_bpm=None,
                pulse_rate_bpm=pulse_rate,
                signal_quality=round(float(result.signal_quality), 4),
                quality_label=quality_enum,
                rhythm_class=None,
                rhythm_confidence=None,
                trend=trend_label,
                model_version=PPG_MODEL_VERSION,
            )
            session_service.add_measurement(frame.session_id, meas)

        return response


@router.websocket("/ws/v1/ppg")
async def websocket_ppg_stream(websocket: WebSocket):
    """Bidirectional streaming WebSocket endpoint for optical smartphone PPG frames."""
    await websocket.accept()
    session_service = get_session_service()
    handler = PPGStreamHandler()

    try:
        while True:
            text_data = await websocket.receive_text()
            try:
                frame_dict = json.loads(text_data)
                frame = PPGWebSocketFrame.model_validate(frame_dict)
            except Exception as e:
                err_resp = {
                    "type": "error",
                    "message": f"Invalid PPG WebSocket frame format: {str(e)}",
                }
                await websocket.send_text(json.dumps(err_resp))
                continue

            resp_frame = handler.process_frame(frame, session_service)
            await websocket.send_text(resp_frame.model_dump_json())

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
