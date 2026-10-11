package com.bioquest.cardiac.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.bioquest.cardiac.R
import com.bioquest.cardiac.camera.CameraPPGAnalyzer
import com.bioquest.cardiac.models.PPGWebSocketResponse
import com.bioquest.cardiac.network.PPGWebSocketManager
import java.util.concurrent.Executors

/**
 * Main Activity providing live optical camera preview, torch control,
 * and real-time pulse rate telemetry visualization.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var tvPulseRate: TextView
    private lateinit var tvQualityLabel: TextView
    private lateinit var tvTrend: TextView
    private lateinit var tvWarningBanner: TextView
    private lateinit var tvConnectionStatus: TextView
    private lateinit var btnToggleFlash: Button

    private var camera: Camera? = null
    private var isTorchOn = false
    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private lateinit var webSocketManager: PPGWebSocketManager

    companion object {
        private const val CAMERA_PERMISSION_CODE = 1001
        // Default emulator loopback host; change to host LAN IP for physical device testing
        private const val DEFAULT_WS_URL = "ws://10.0.2.2:8000/ws/v1/ppg"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initViews()

        webSocketManager = PPGWebSocketManager(
            serverUrl = DEFAULT_WS_URL,
            onUpdateReceived = { update -> runOnUiThread { handleMeasurementUpdate(update) } },
            onConnectionStateChanged = { isConnected, status ->
                runOnUiThread {
                    tvConnectionStatus.text = status
                    val statusColor = if (isConnected) {
                        ContextCompat.getColor(this, R.color.status_good)
                    } else {
                        ContextCompat.getColor(this, R.color.status_poor)
                    }
                    tvConnectionStatus.setTextColor(statusColor)
                }
            }
        )

        if (checkCameraPermission()) {
            startCamera()
        } else {
            requestCameraPermission()
        }
    }

    private fun initViews() {
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        tvPulseRate = findViewById(R.id.tvPulseRate)
        tvQualityLabel = findViewById(R.id.tvQualityLabel)
        tvTrend = findViewById(R.id.tvTrend)
        tvWarningBanner = findViewById(R.id.tvWarningBanner)
        tvConnectionStatus = findViewById(R.id.tvConnectionStatus)
        btnToggleFlash = findViewById(R.id.btnToggleFlash)

        btnToggleFlash.setOnClickListener {
            toggleTorch()
        }
    }

    private fun toggleTorch() {
        camera?.let { cam ->
            isTorchOn = !isTorchOn
            cam.cameraControl.enableTorch(isTorchOn)
            btnToggleFlash.text = if (isTorchOn) {
                getString(R.string.torch_off_label)
            } else {
                getString(R.string.torch_on_label)
            }
        }
    }

    private fun checkCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_CODE
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            Toast.makeText(
                this,
                "Camera permission is required for optical PPG acquisition.",
                Toast.LENGTH_LONG
            ).show()
            tvWarningBanner.visibility = View.VISIBLE
            tvWarningBanner.text = "Camera permission denied. Please allow camera access in Settings."
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, CameraPPGAnalyzer(
                        onFrameAnalyzed = { frame -> webSocketManager.sendFrame(frame) },
                        onCoverageAlert = { isCovering ->
                            runOnUiThread {
                                if (!isCovering) {
                                    tvWarningBanner.visibility = View.VISIBLE
                                    tvWarningBanner.text = getString(R.string.instruction_text)
                                } else {
                                    tvWarningBanner.visibility = View.GONE
                                }
                            }
                        }
                    ))
                }

            try {
                cameraProvider.unbindAll()
                camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)

                // Automatically activate torch/flash for fingertip transillumination
                camera?.cameraControl?.enableTorch(true)
                isTorchOn = true
                btnToggleFlash.text = getString(R.string.torch_off_label)

                // Initiate backend WebSocket streaming
                webSocketManager.connect()
            } catch (exc: Exception) {
                exc.printStackTrace()
                tvConnectionStatus.text = "Camera initialization error: ${exc.localizedMessage}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun handleMeasurementUpdate(update: PPGWebSocketResponse) {
        if (update.pulseRateBpm != null) {
            tvPulseRate.text = "%.1f BPM".format(update.pulseRateBpm)
        } else {
            tvPulseRate.text = "-- BPM"
        }

        val sqiText = "SQI: %.2f (%s)".format(update.signalQuality, update.qualityLabel)
        tvQualityLabel.text = sqiText

        val qualityColor = when (update.qualityLabel) {
            "GOOD" -> ContextCompat.getColor(this, R.color.status_good)
            "FAIR" -> ContextCompat.getColor(this, R.color.status_fair)
            else -> ContextCompat.getColor(this, R.color.status_poor)
        }
        tvQualityLabel.setTextColor(qualityColor)

        tvTrend.text = "Trend: %s".format(update.trend ?: "STABLE")

        if (update.warning != null) {
            tvWarningBanner.visibility = View.VISIBLE
            tvWarningBanner.text = update.warning
        } else {
            tvWarningBanner.visibility = View.GONE
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        webSocketManager.disconnect()
    }
}
