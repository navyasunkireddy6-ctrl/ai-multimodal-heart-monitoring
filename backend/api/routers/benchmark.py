"""Phase 13: REST API router for high-concurrency multi-patient stress benchmarking."""

from fastapi import APIRouter, Depends, HTTPException, Response, status

from backend.api.deps import get_benchmark_service
from backend.models.schemas import (
    ConcurrencyBenchmarkRequest,
    ConcurrencyBenchmarkResult,
)
from backend.services.benchmark_service import BenchmarkService

router = APIRouter(prefix="/benchmark", tags=["Concurrency Benchmarking"])


@router.post(
    "/run",
    response_model=ConcurrencyBenchmarkResult,
    status_code=status.HTTP_200_OK,
    summary="Execute multi-patient high-concurrency telemetry stress benchmark",
)
async def execute_benchmark(
    request: ConcurrencyBenchmarkRequest = ConcurrencyBenchmarkRequest(),
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),
) -> ConcurrencyBenchmarkResult:
    """Run concurrent patient stream simulations and measure latency, jitter, and throughput."""
    return await benchmark_service.run_benchmark(request)


@router.get(
    "/last",
    response_model=ConcurrencyBenchmarkResult,
    summary="Retrieve the most recently recorded stress benchmark scorecard",
)
def get_latest_benchmark(
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),
) -> ConcurrencyBenchmarkResult:
    """Fetch the latest benchmark execution results."""
    res = benchmark_service.get_last_result()
    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No benchmark run has been recorded in this server session.",
        )
    return res


@router.get(
    "/html",
    response_class=Response,
    summary="Export standalone printable HTML benchmark dashboard",
)
async def get_benchmark_html(
    benchmark_service: BenchmarkService = Depends(get_benchmark_service),
) -> Response:
    """Generate or retrieve HTML performance dashboard."""
    res = benchmark_service.get_last_result()
    if not res:
        # Auto-run standard baseline benchmark if none exists
        res = await benchmark_service.run_benchmark(ConcurrencyBenchmarkRequest(concurrent_patients=10, frames_per_patient=20))

    html_content = benchmark_service.render_benchmark_html(res)
    return Response(content=html_content, media_type="text/html")
