# Canonical API Contract

**System**: Multimodal Real-Time Cardiac Monitoring Using ECG-Based Heart Rate and Smartphone PPG-Based Pulse Rate with Machine Learning  
**Version**: 1.0.0  
**Disclaimer**: *Research prototype — not intended for medical diagnosis.*

This document serves as the **Single Source of Truth (SSOT)** for all backend interfaces, data types, and network contracts between clients (Web Dashboard, Android App, CLI scripts) and the FastAPI backend.

---

## 1. Canonical Enums

### 1.1 Operating Modes (`mode`)
- `ecg_dataset`: Playback or processing of historical ECG records (e.g., MIT-BIH Arrhythmia Database).
- `smartphone_ppg`: Live optical fingertip photoplethysmogram acquired from the smartphone camera.
- `synthetic_ppg`: Algorithmically generated PPG waveforms for testing and calibration.
- `offline_demo`: Self-contained demonstration combining cached ECG and synthetic/recorded PPG.

### 1.2 Quality Labels (`quality_label`)
- `GOOD`: High signal-to-noise ratio; morphological features and peaks are unambiguous.
- `FAIR`: Moderate noise or slight baseline wander; rate estimation is reliable within tolerance.
- `POOR`: High motion artifact or poor sensor contact; rates must not be reported as confident.

### 1.3 Rate Trend Labels (`trend`)
- `INCREASING`: Statistically significant upward slope in recent rate history.
- `STABLE`: Heart/pulse rate fluctuating within normal resting variability window.
- `DECREASING`: Statistically significant downward slope in recent rate history.

---

## 2. Canonical Measurement Schema

Every processed cardiovascular measurement or session record conforms to this exact field schema:

```json
{
  "timestamp": "2026-09-30T12:00:00.000Z",
  "mode": "ecg_dataset",
  "heart_rate_bpm": 74.5,
  "pulse_rate_bpm": null,
  "signal_quality": 0.92,
  "quality_label": "GOOD",
  "rhythm_class": "Normal Sinus Rhythm",
  "rhythm_confidence": 0.96,
  "trend": "STABLE",
  "model_version": "rf-ecg-v1.0.0"
}
```

### Exact Field Definitions:
| Field Name | Type | Nullable | Description |
|---|---|---|---|
| `timestamp` | string (ISO-8601) | No | Timestamp of the measurement window |
| `mode` | string (Enum) | No | One of: `ecg_dataset`, `smartphone_ppg`, `synthetic_ppg`, `offline_demo` |
| `heart_rate_bpm` | float | Yes | Calculated from ECG R-R intervals (null if ECG unavailable) |
| `pulse_rate_bpm` | float | Yes | Calculated from PPG inter-beat intervals (null if PPG unavailable or POOR quality) |
| `signal_quality` | float | No | Normalized signal quality index in range `[0.0, 1.0]` |
| `quality_label` | string (Enum) | No | One of: `GOOD`, `FAIR`, `POOR` |
| `rhythm_class` | string | Yes | Predicted ECG rhythm class based on dataset labels (null for PPG-only) |
| `rhythm_confidence` | float | Yes | Classifier confidence in range `[0.0, 1.0]` |
| `trend` | string (Enum) | Yes | One of: `INCREASING`, `STABLE`, `DECREASING` |
| `model_version` | string | No | Identifier string of active ML model or algorithm pipeline |

---

## 3. Canonical REST Endpoints

Base path: `/api/v1`

### 3.1 Health Check
- **Endpoint**: `GET /api/v1/health`
- **Description**: Returns backend service status, runtime environment, and loaded ML model versions.
- **Response `200 OK`**:
```json
{
  "status": "healthy",
  "timestamp": "2026-09-30T12:00:00.000Z",
  "version": "1.0.0",
  "models_loaded": {
    "ecg_classifier": "rf-ecg-v1.0.0",
    "trend_analyzer": "trend-rule-v1.0.0"
  },
  "disclaimer": "Research prototype — not intended for medical diagnosis."
}
```

---

### 3.2 ECG Signal Processing
- **Endpoint**: `POST /api/v1/ecg/process`
- **Description**: Filters raw ECG window, identifies R-peaks, calculates R-R intervals, heart rate, and signal quality index (SQI).
- **Request `POST /api/v1/ecg/process`**:
```json
{
  "sampling_rate_hz": 360.0,
  "signal": [0.12, 0.15, -0.05, 1.25, 0.32, -0.10],
  "record_id": "100"
}
```
- **Response `200 OK`**:
```json
{
  "heart_rate_bpm": 75.2,
  "signal_quality": 0.94,
  "quality_label": "GOOD",
  "r_peaks": [36, 324, 612],
  "rr_intervals_ms": [800.0, 800.0],
  "filtered_signal": [0.05, 0.08, -0.01, 1.15, 0.28, -0.08]
}
```

---

### 3.3 ECG Rhythm Classification
- **Endpoint**: `POST /api/v1/ecg/classify`
- **Description**: Classifies an ECG segment using extracted physiological features into trained dataset rhythm classes.
- **Request `POST /api/v1/ecg/classify`**:
```json
{
  "sampling_rate_hz": 360.0,
  "signal": [0.12, 0.15, -0.05, 1.25, 0.32, -0.10],
  "rr_intervals_ms": [800.0, 810.0, 795.0]
}
```
- **Response `200 OK`**:
```json
{
  "rhythm_class": "Normal Sinus Rhythm",
  "rhythm_confidence": 0.954,
  "model_version": "rf-ecg-v1.0.0",
  "features": {
    "mean_rr_ms": 801.67,
    "sdnn_ms": 7.64,
    "rmssd_ms": 12.25
  },
  "disclaimer": "Research prototype — not intended for medical diagnosis."
}
```

---

### 3.4 PPG Signal Processing
- **Endpoint**: `POST /api/v1/ppg/process`
- **Description**: Filters raw optical PPG waveform, detects systolic peaks, calculates pulse rate and quality.
- **Request `POST /api/v1/ppg/process`**:
```json
{
  "sampling_rate_hz": 30.0,
  "signal": [128.4, 132.1, 140.5, 145.0, 138.2, 125.1],
  "source": "smartphone_camera"
}
```
- **Response `200 OK`**:
```json
{
  "pulse_rate_bpm": 72.0,
  "signal_quality": 0.88,
  "quality_label": "GOOD",
  "peaks": [15, 40, 65],
  "filtered_signal": [-0.5, 0.2, 1.1, 1.4, 0.6, -0.8]
}
```

---

### 3.5 PPG Quality Assessment
- **Endpoint**: `POST /api/v1/ppg/quality`
- **Description**: Evaluates optical PPG signal quality index based on periodicity, amplitude consistency, and spectral power.
- **Request `POST /api/v1/ppg/quality`**:
```json
{
  "sampling_rate_hz": 30.0,
  "signal": [128.4, 132.1, 140.5, 145.0, 138.2, 125.1]
}
```
- **Response `200 OK`**:
```json
{
  "signal_quality": 0.88,
  "quality_label": "GOOD",
  "snr_db": 14.5,
  "is_usable": true,
  "message": "Signal quality sufficient for reliable pulse rate estimation."
}
```

---

### 3.6 Short-Term Trend Prediction
- **Endpoint**: `POST /api/v1/trend/predict`
- **Description**: Evaluates rate sequence history over a moving time window to determine rate trajectory.
- **Request `POST /api/v1/trend/predict`**:
```json
{
  "timestamps": [
    "2026-09-30T12:00:00Z",
    "2026-09-30T12:00:05Z",
    "2026-09-30T12:00:10Z",
    "2026-09-30T12:00:15Z"
  ],
  "rates": [70.0, 72.0, 75.0, 78.0]
}
```
- **Response `200 OK`**:
```json
{
  "trend": "INCREASING",
  "slope_bpm_per_min": 16.0,
  "confidence": 0.91,
  "window_duration_seconds": 15.0,
  "disclaimer": "Research prototype — not intended for medical diagnosis."
}
```

---

### 3.7 Session Management Endpoints
- **Create Session**: `POST /api/v1/sessions`
  - Body: `SessionCreateRequest` (accepts `mode`, optional initial `notes`)
  - Response `201 Created`: `SessionResponse`
- **List Sessions**: `GET /api/v1/sessions?limit=50&offset=0`
  - Response `200 OK`: `SessionListResponse`
- **Get Session Details**: `GET /api/v1/sessions/{session_id}`
  - Response `200 OK`: `SessionDetailResponse` (includes full measurement history)
- **Delete Session**: `DELETE /api/v1/sessions/{session_id}`
  - Response `200 OK`: `{"deleted": true, "session_id": "uuid"}`
- **Export Session CSV**: `GET /api/v1/sessions/{session_id}/export`
  - Response `200 OK`: `text/csv` file download

---

## 4. Canonical Live PPG WebSocket Endpoint

- **URI**: `/ws/v1/ppg`
- **Protocol**: WebSocket (JSON text frames)
- **Direction**: Bidirectional streaming between Android client / Web client and FastAPI backend.

### 4.1 Client Inbound Frame (Android Camera / Synthetic Stream)
```json
{
  "type": "ppg_frame",
  "timestamp": "2026-09-30T12:00:01.032Z",
  "session_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "red_channel_value": 142.6,
  "green_channel_value": 85.3,
  "blue_channel_value": 78.1,
  "flash_enabled": true,
  "camera_fps": 30.0
}
```

### 4.2 Server Outbound Response Frame (Broadcast to App & Dashboard)
```json
{
  "type": "measurement_update",
  "timestamp": "2026-09-30T12:00:01.032Z",
  "mode": "smartphone_ppg",
  "heart_rate_bpm": null,
  "pulse_rate_bpm": 73.4,
  "signal_quality": 0.91,
  "quality_label": "GOOD",
  "rhythm_class": null,
  "rhythm_confidence": null,
  "trend": "STABLE",
  "model_version": "ppg-peak-v1.0.0",
  "raw_red": 142.6,
  "filtered_value": 0.45
}
```

If signal quality drops to `POOR`, server responds with `pulse_rate_bpm: null` and explicit advisory:
```json
{
  "type": "measurement_update",
  "timestamp": "2026-09-30T12:00:02.100Z",
  "mode": "smartphone_ppg",
  "heart_rate_bpm": null,
  "pulse_rate_bpm": null,
  "signal_quality": 0.32,
  "quality_label": "POOR",
  "rhythm_class": null,
  "rhythm_confidence": null,
  "trend": null,
  "model_version": "ppg-peak-v1.0.0",
  "warning": "Poor optical contact. Cover camera and flash steadily."
}
```
