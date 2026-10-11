"""Automated Clinical Diagnostic Report REST router."""

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import HTMLResponse

from backend.api.deps import get_report_service
from backend.models.schemas import (
    ClinicalReportGenerateRequest,
    ClinicalReportSummary,
)
from backend.services.report_service import ReportService

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/{session_id}/summary", response_model=ClinicalReportSummary)
def get_session_report_summary(
    session_id: str,
    service: ReportService = Depends(get_report_service),
) -> ClinicalReportSummary:
    """Compile and return structured clinical summary data for a recorded session."""
    report = service.build_report_for_session(session_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return report


@router.get("/{session_id}/html", response_class=HTMLResponse)
def get_session_report_html(
    session_id: str,
    service: ReportService = Depends(get_report_service),
) -> HTMLResponse:
    """Render a publication-ready, printable HTML medical diagnostic report."""
    report = service.build_report_for_session(session_id)
    if report is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    html_content = service.render_html_report(report)
    return HTMLResponse(content=html_content, status_code=status.HTTP_200_OK)


@router.post("/generate", response_model=ClinicalReportSummary)
def generate_adhoc_report(
    request: ClinicalReportGenerateRequest,
    service: ReportService = Depends(get_report_service),
) -> ClinicalReportSummary:
    """Generate an ad-hoc clinical report from a provided stream of measurements."""
    mode_str = request.mode if isinstance(request.mode, str) else request.mode.value
    return service.build_report_from_measurements(
        measurements=request.measurements,
        session_id=request.session_id,
        patient_notes=request.patient_notes,
        mode=mode_str,
    )
