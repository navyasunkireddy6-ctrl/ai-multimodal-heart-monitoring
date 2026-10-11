"""Unit and integration tests for Phase 13: High-Concurrency Multi-Patient Stress Benchmarking."""

import pytest
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.models.schemas import (
    ConcurrencyBenchmarkRequest,
    ConcurrencyBenchmarkResult,
    ModeEnum,
)
from backend.services.benchmark_service import BenchmarkService


def test_benchmark_service_frame_generation():
    """Verify synthetic photometric frame generation produces physiologically valid bounds."""
    service = BenchmarkService()
    frame = service.generate_patient_frame(patient_index=0, frame_index=15, base_hr=72.0)

    assert frame["type"] == "ppg_frame"
    assert frame["patient_id"] == "PT-SIM-001"
    assert frame["frame_index"] == 15
    assert 100.0 <= frame["red_channel_value"] <= 200.0
    assert 60.0 <= frame["green_channel_value"] <= 120.0
    assert 50.0 <= frame["blue_channel_value"] <= 90.0
    assert frame["flash_enabled"] is True
    assert frame["camera_fps"] == 30.0


@pytest.mark.anyio
async def test_benchmark_single_stream_metrics():
    """Verify a single simulated telemetry stream executes and records latency and throughput."""
    service = BenchmarkService()
    metrics = await service.simulate_single_patient_stream(
        patient_index=1,
        frames_count=15,
        target_fps=30.0,
    )

    assert metrics.patient_id == "PT-SIM-002"
    assert metrics.frames_sent == 15
    assert metrics.frames_received == 15
    assert metrics.packet_loss_rate == 0.0
    assert metrics.mean_latency_ms >= 0.0
    assert metrics.p95_latency_ms >= 0.0
    assert metrics.fps > 0.0


@pytest.mark.anyio
async def test_benchmark_multi_patient_concurrency():
    """Verify 10 concurrent simulated patient streams execute simultaneously."""
    service = BenchmarkService(p95_slo_threshold_ms=100.0)
    req = ConcurrencyBenchmarkRequest(
        concurrent_patients=10,
        frames_per_patient=20,
        target_fps=30.0,
        mode=ModeEnum.SMARTPHONE_PPG,
    )

    result = await service.run_benchmark(req)

    assert isinstance(result, ConcurrencyBenchmarkResult)
    assert result.concurrent_patients == 10
    assert result.total_frames_processed == 200
    assert result.total_duration_seconds > 0.0
    assert result.overall_throughput_fps > 0.0
    assert result.p50_latency_ms >= 0.0
    assert result.p95_latency_ms >= result.p50_latency_ms
    assert result.p99_latency_ms >= result.p95_latency_ms
    assert result.overall_packet_loss_pct == 0.0
    assert result.slo_passed is True
    assert len(result.patient_metrics) == 10


def test_benchmark_html_rendering():
    """Verify HTML performance report rendering contains required metrics and elements."""
    service = BenchmarkService()
    req = ConcurrencyBenchmarkRequest(concurrent_patients=4, frames_per_patient=10)

    import asyncio
    result = asyncio.run(service.run_benchmark(req))
    html = service.render_benchmark_html(result)

    assert "<!DOCTYPE html>" in html
    assert "BioQuest Multimodal Telemetry Concurrency Benchmark" in html
    assert "PT-SIM-001" in html
    assert "Throughput" in html
    assert "PASSED" in html or "FAILED" in html
    assert result.disclaimer in html


def test_benchmark_rest_api_lifecycle():
    """Verify REST API endpoints: POST /run, GET /last, and GET /html."""
    client = TestClient(app)

    # 1. Trigger benchmark run
    payload = {
        "concurrent_patients": 5,
        "frames_per_patient": 10,
        "target_fps": 30.0,
        "mode": "smartphone_ppg",
    }
    run_resp = client.post("/api/v1/benchmark/run", json=payload)
    assert run_resp.status_code == 200
    data = run_resp.json()
    assert data["concurrent_patients"] == 5
    assert data["total_frames_processed"] == 50
    assert data["slo_passed"] is True

    # 2. Query last benchmark scorecard
    last_resp = client.get("/api/v1/benchmark/last")
    assert last_resp.status_code == 200
    last_data = last_resp.json()
    assert last_data["benchmark_id"] == data["benchmark_id"]

    # 3. Retrieve standalone HTML dashboard
    html_resp = client.get("/api/v1/benchmark/html")
    assert html_resp.status_code == 200
    assert "text/html" in html_resp.headers.get("content-type", "")
    assert "<!DOCTYPE html>" in html_resp.text
