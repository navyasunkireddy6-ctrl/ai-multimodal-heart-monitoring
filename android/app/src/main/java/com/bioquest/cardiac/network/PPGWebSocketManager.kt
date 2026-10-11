package com.bioquest.cardiac.network

import com.bioquest.cardiac.models.PPGWebSocketFrame
import com.bioquest.cardiac.models.PPGWebSocketResponse
import com.google.gson.Gson
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.util.concurrent.TimeUnit

/**
 * Manages bidirectional WebSocket streaming between the Android client and the FastAPI backend.
 * Conforms to Section 4 of docs/API_CONTRACT.md.
 */
class PPGWebSocketManager(
    private var serverUrl: String = "ws://10.0.2.2:8000/ws/v1/ppg",
    private val onUpdateReceived: (PPGWebSocketResponse) -> Unit,
    private val onConnectionStateChanged: (isConnected: Boolean, status: String) -> Unit
) {

    private val gson = Gson()
    private val client = OkHttpClient.Builder()
        .readTimeout(10, TimeUnit.SECONDS)
        .writeTimeout(10, TimeUnit.SECONDS)
        .pingInterval(15, TimeUnit.SECONDS)
        .retryOnConnectionFailure(true)
        .build()

    private var webSocket: WebSocket? = null
    private var isConnected = false

    fun setServerUrl(newUrl: String) {
        if (serverUrl != newUrl) {
            serverUrl = newUrl
            if (isConnected) {
                disconnect()
                connect()
            }
        }
    }

    fun getServerUrl(): String = serverUrl

    fun isConnected(): Boolean = isConnected

    fun connect() {
        if (isConnected && webSocket != null) {
            return
        }

        onConnectionStateChanged(false, "Connecting to $serverUrl...")
        val request = Request.Builder().url(serverUrl).build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: Response) {
                isConnected = true
                onConnectionStateChanged(true, "Connected to backend telemetry")
            }

            override fun onMessage(ws: WebSocket, text: String) {
                try {
                    val update = gson.fromJson(text, PPGWebSocketResponse::class.java)
                    if (update != null && update.type == "measurement_update") {
                        onUpdateReceived(update)
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: Response?) {
                isConnected = false
                val errorMsg = t.localizedMessage ?: "Unknown network failure"
                onConnectionStateChanged(false, "Connection error: $errorMsg")
            }

            override fun onClosed(ws: WebSocket, code: Int, reason: String) {
                isConnected = false
                onConnectionStateChanged(false, "Disconnected: $reason")
            }
        })
    }

    fun sendFrame(frame: PPGWebSocketFrame) {
        if (isConnected && webSocket != null) {
            val json = gson.toJson(frame)
            webSocket?.send(json)
        }
    }

    fun disconnect() {
        try {
            webSocket?.close(1000, "Client initiated disconnect")
        } catch (e: Exception) {
            e.printStackTrace()
        } finally {
            webSocket = null
            isConnected = false
            onConnectionStateChanged(false, "Disconnected")
        }
    }
}
