package com.bioquest.cardiac.models

import com.google.gson.annotations.SerializedName

/**
 * Inbound optical frame streamed to FastAPI backend at /ws/v1/ppg.
 * Conforms strictly to Section 4.1 of docs/API_CONTRACT.md.
 */
data class PPGWebSocketFrame(
    @SerializedName("type")
    val type: String = "ppg_frame",

    @SerializedName("timestamp")
    val timestamp: String,

    @SerializedName("session_id")
    val sessionId: String? = null,

    @SerializedName("red_channel_value")
    val redChannelValue: Double,

    @SerializedName("green_channel_value")
    val greenChannelValue: Double? = null,

    @SerializedName("blue_channel_value")
    val blueChannelValue: Double? = null,

    @SerializedName("flash_enabled")
    val flashEnabled: Boolean = true,

    @SerializedName("camera_fps")
    val cameraFps: Double = 30.0
)

/**
 * Outbound real-time telemetry broadcast received from /ws/v1/ppg.
 * Conforms strictly to Section 4.2 of docs/API_CONTRACT.md.
 */
data class PPGWebSocketResponse(
    @SerializedName("type")
    val type: String = "measurement_update",

    @SerializedName("timestamp")
    val timestamp: String,

    @SerializedName("mode")
    val mode: String = "smartphone_ppg",

    @SerializedName("heart_rate_bpm")
    val heartRateBpm: Double? = null,

    @SerializedName("pulse_rate_bpm")
    val pulseRateBpm: Double? = null,

    @SerializedName("signal_quality")
    val signalQuality: Double,

    @SerializedName("quality_label")
    val qualityLabel: String, // GOOD, FAIR, POOR

    @SerializedName("rhythm_class")
    val rhythmClass: String? = null,

    @SerializedName("rhythm_confidence")
    val rhythmConfidence: Double? = null,

    @SerializedName("trend")
    val trend: String? = null, // INCREASING, STABLE, DECREASING

    @SerializedName("model_version")
    val modelVersion: String = "ppg-peak-v1.0.0",

    @SerializedName("raw_red")
    val rawRed: Double? = null,

    @SerializedName("filtered_value")
    val filteredValue: Double? = null,

    @SerializedName("warning")
    val warning: String? = null
)
