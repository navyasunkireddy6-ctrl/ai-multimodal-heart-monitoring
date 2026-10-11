package com.bioquest.cardiac.camera

import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.bioquest.cardiac.models.PPGWebSocketFrame
import java.nio.ByteBuffer
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/**
 * CameraX ImageAnalysis analyzer for optical fingertip photoplethysmography.
 *
 * Samples the central region of interest (ROI) where the patient's fingertip covers
 * the camera lens and LED torch, extracting spatial mean Red, Green, and Blue intensities.
 */
class CameraPPGAnalyzer(
    private val onFrameAnalyzed: (PPGWebSocketFrame) -> Unit,
    private val onCoverageAlert: (isFingerCovering: Boolean) -> Unit
) : ImageAnalysis.Analyzer {

    private var targetFps = 30.0
    private var lastFrameTimeMs = 0L
    private val minFrameIntervalMs = (1000.0 / targetFps).toLong()

    private val isoDateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).apply {
        timeZone = TimeZone.getTimeZone("UTC")
    }

    override fun analyze(image: ImageProxy) {
        val currentTimeMs = System.currentTimeMillis()

        // Regulate capture rate to ~30 FPS
        if (currentTimeMs - lastFrameTimeMs < minFrameIntervalMs) {
            image.close()
            return
        }
        lastFrameTimeMs = currentTimeMs

        try {
            val planes = image.planes
            if (planes.isEmpty()) {
                return
            }

            // Central 50% Region of Interest (ROI)
            val width = image.width
            val height = image.height
            val roiStartX = width / 4
            val roiEndX = (width * 3) / 4
            val roiStartY = height / 4
            val roiEndY = (height * 3) / 4

            // Extract YUV_420_888 channel intensity
            val yBuffer = planes[0].buffer
            val uBuffer = if (planes.size > 1) planes[1].buffer else null
            val vBuffer = if (planes.size > 2) planes[2].buffer else null

            val yRowStride = planes[0].rowStride
            val yPixelStride = planes[0].pixelStride

            val uRowStride = if (planes.size > 1) planes[1].rowStride else 0
            val uPixelStride = if (planes.size > 1) planes[1].pixelStride else 0
            val vRowStride = if (planes.size > 2) planes[2].rowStride else 0
            val vPixelStride = if (planes.size > 2) planes[2].pixelStride else 0

            var totalRed = 0.0
            var totalGreen = 0.0
            var totalBlue = 0.0
            var pixelCount = 0

            // Sample rows and columns with step 2 for low CPU footprint and high throughput
            val step = 2
            for (y in roiStartY until roiEndY step step) {
                for (x in roiStartX until roiEndX step step) {
                    val yIndex = y * yRowStride + x * yPixelStride
                    val yVal = if (yIndex < yBuffer.limit()) (yBuffer.get(yIndex).toInt() and 0xFF) else 0

                    var uVal = 128
                    var vVal = 128

                    if (uBuffer != null && vBuffer != null) {
                        val uIndex = (y / 2) * uRowStride + (x / 2) * uPixelStride
                        val vIndex = (y / 2) * vRowStride + (x / 2) * vPixelStride
                        if (uIndex < uBuffer.limit()) uVal = (uBuffer.get(uIndex).toInt() and 0xFF)
                        if (vIndex < vBuffer.limit()) vVal = (vBuffer.get(vIndex).toInt() and 0xFF)
                    }

                    // Standard Rec. 601 YUV to RGB Conversion
                    val r = (yVal + 1.402 * (vVal - 128)).coerceIn(0.0, 255.0)
                    val g = (yVal - 0.344136 * (uVal - 128) - 0.714136 * (vVal - 128)).coerceIn(0.0, 255.0)
                    val b = (yVal + 1.772 * (uVal - 128)).coerceIn(0.0, 255.0)

                    totalRed += r
                    totalGreen += g
                    totalBlue += b
                    pixelCount++
                }
            }

            if (pixelCount > 0) {
                val meanRed = totalRed / pixelCount
                val meanGreen = totalGreen / pixelCount
                val meanBlue = totalBlue / pixelCount

                // Check fingertip coverage: Red channel should dominate and be sufficiently bright
                val isCovering = meanRed > 70.0 && meanRed > (meanBlue * 1.3)
                onCoverageAlert(isCovering)

                val frame = PPGWebSocketFrame(
                    timestamp = isoDateFormat.format(Date(currentTimeMs)),
                    redChannelValue = meanRed,
                    greenChannelValue = meanGreen,
                    blueChannelValue = meanBlue,
                    flashEnabled = true,
                    cameraFps = targetFps
                )
                onFrameAnalyzed(frame)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            image.close()
        }
    }
}
