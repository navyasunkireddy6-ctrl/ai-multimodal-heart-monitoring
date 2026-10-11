"""CLI Demonstration of Phase 13: High-Concurrency Multi-Patient Stress & Performance Benchmarking Engine.

Demonstrates:
1. Multi-stream concurrency orchestration (10 simultaneous patient streams).
2. Asynchronous high-frequency frame transmission and arrival latency profiling.
3. Quantile latency breakdown (p50, p95, p99, Max).
4. Inter-frame network & processing jitter measurement.
5. Packet loss and stream reliability validation.
6. Clinical Service Level Objective (SLO) compliance verification.
7. HTML & JSON benchmark scorecard artifact export to reports/.

Run with:
    python scripts/demo_concurrency_benchmark.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.models.schemas import ConcurrencyBenchmarkRequest, ModeEnum
from backend.services.benchmark_service import BenchmarkService


def run_demo():
    print("=" * 85)
    print("  PHASE 13: HIGH-CONCURRENCY MULTI-PATIENT STRESS & PERFORMANCE BENCHMARK")
    print("  BioQuest Multimodal Cardiac Monitoring System (Research Prototype)")
    print("=" * 85)

    num_patients = 10
    frames_per_pt = 30
    target_fps = 30.0

    print(f"\n[1] Benchmark Configuration:")
    print(f"    - Concurrent Patients:     {num_patients} active telemetry streams")
    print(f"    - Frames Per Patient:       {frames_per_pt} frames")
    print(f"    - Total Target Frames:      {num_patients * frames_per_pt} frames")
    print(f"    - Target Stream Pace:       {target_fps:.1f} FPS")
    print(f"    - Latency SLO Target (p95): <= 50.0 ms")
    print(f"    - Packet Loss SLO Target:   0.00%")

    service = BenchmarkService(p95_slo_threshold_ms=50.0, max_packet_loss_slo=0.0)

    # --------------------------------------------------------------------------
    # 2. Execute Multi-Patient Benchmark
    # --------------------------------------------------------------------------
    print(f"\n[2] Executing Asynchronous Multi-Stream Telemetry Benchmark...")
    t_start = time.perf_counter()

    req = ConcurrencyBenchmarkRequest(
        concurrent_patients=num_patients,
        frames_per_patient=frames_per_pt,
        target_fps=target_fps,
        mode=ModeEnum.SMARTPHONE_PPG,
    )

    result = asyncio.run(service.run_benchmark(req))
    elapsed = time.perf_counter() - t_start
    print(f"    -> Stress test execution completed in {elapsed:.2f} seconds.")

    # --------------------------------------------------------------------------
    # 3. Display Performance Scorecard
    # --------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("                  CONCURRENCY & LATENCY BENCHMARK SCORECARD")
    print("=" * 85)
    print(f" Benchmark UUID:        {result.benchmark_id}")
    print(f" Execution Timestamp:   {result.timestamp}")
    print(f" Total Active Streams:  {result.concurrent_patients} concurrent patients")
    print(f" Total Frames Ingested: {result.total_frames_processed} frames")
    print(f" Aggregate Throughput:  {result.overall_throughput_fps:.1f} FPS")
    print(f" Overall Packet Loss:   {result.overall_packet_loss_pct:.2f}%")
    print("-" * 85)
    print(" LATENCY & JITTER DISTRIBUTION (Round-Trip Processing):")
    print(f"   • Median Latency (p50):  {result.p50_latency_ms:>6.2f} ms")
    print(f"   • 95th Percentile (p95): {result.p95_latency_ms:>6.2f} ms")
    print(f"   • 99th Percentile (p99): {result.p99_latency_ms:>6.2f} ms")
    print(f"   • Peak Latency (Max):    {result.max_latency_ms:>6.2f} ms")
    print(f"   • Mean Stream Jitter:    {result.mean_jitter_ms:>6.2f} ms")
    print("-" * 85)
    print(" CLINICAL SERVICE LEVEL OBJECTIVES (SLO) AUDIT:")
    for slo_name, passed in result.slo_details.items():
        status_tag = "[PASS]" if passed else "[FAIL]"
        print(f"   • {slo_name:<40} : {status_tag}")
    slo_overall = "PASSED (100% SLOs Compliant)" if result.slo_passed else "FAILED"
    print(f"\n   -> OVERALL SLO STATUS: {slo_overall}")
    print("-" * 85)
    print(" PER-PATIENT STREAM SUMMARY:")
    print(f" {'PATIENT ID':<12} | {'SENT':<6} | {'RECV':<6} | {'MEAN LAT':<10} | {'p95 LAT':<10} | {'JITTER':<10} | {'RATE':<10}")
    print("-" * 85)
    for m in result.patient_metrics:
        print(
            f" {m.patient_id:<12} | {m.frames_sent:>5}  | {m.frames_received:>5}  | "
            f"{m.mean_latency_ms:>7.2f} ms | {m.p95_latency_ms:>7.2f} ms | {m.jitter_ms:>7.2f} ms | {m.fps:>6.1f} FPS"
        )
    print("=" * 85)

    # --------------------------------------------------------------------------
    # 4. Export Artifacts
    # --------------------------------------------------------------------------
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / "concurrency_benchmark_report.json"
    json_path.write_text(json.dumps(result.model_dump(), indent=2), encoding="utf-8")
    print(f"\n[3] Exported JSON Benchmark Metrics to: {json_path.resolve()}")

    html_content = service.render_benchmark_html(result)
    html_path = reports_dir / "concurrency_benchmark_report.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"[4] Exported HTML Benchmark Dashboard to: {html_path.resolve()}")

    print("\n" + "=" * 85)
    print("  PHASE 13 DEMO COMPLETED SUCCESSFULLY WITH ZERO ERRORS.")
    print("=" * 85)


if __name__ == "__main__":
    run_demo()
