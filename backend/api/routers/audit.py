"""Phase 15: REST API router for clinical audit compliance, data integrity, and system diagnostics."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse

from backend.api.deps import get_audit_service
from backend.models.schemas import (
    AuditIntegrityManifest,
    ComplianceStatementResponse,
    SystemDiagnosticsResponse,
)
from backend.services.audit_service import AuditService

router = APIRouter(prefix="/audit", tags=["Clinical Audit & Compliance"])


@router.get(
    "/integrity/{session_id}",
    response_model=AuditIntegrityManifest,
    status_code=status.HTTP_200_OK,
    summary="Generate tamper-evident cryptographic SHA-256 audit manifest for a session",
)
def get_session_integrity_manifest(
    session_id: str,
    audit_service: AuditService = Depends(get_audit_service),
) -> AuditIntegrityManifest:
    """Compute and verify SHA-256 cryptographic checksum of telemetry sequence."""
    sess = audit_service.session_service.get_session(session_id)
    if not sess:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Monitoring session '{session_id}' not found.",
        )
    return audit_service.compute_session_manifest(session_id, sess.measurements)


@router.get(
    "/system-diagnostics",
    response_model=SystemDiagnosticsResponse,
    status_code=status.HTTP_200_OK,
    summary="Run comprehensive multi-subsystem self-diagnostic health audit",
)
def get_system_diagnostics(
    audit_service: AuditService = Depends(get_audit_service),
) -> SystemDiagnosticsResponse:
    """Execute live diagnostic probes across all 10 core system modules."""
    return audit_service.run_deep_system_diagnostics()


@router.get(
    "/compliance",
    response_model=ComplianceStatementResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve clinical audit regulatory compliance framework statement",
)
def get_compliance_statement(
    audit_service: AuditService = Depends(get_audit_service),
) -> ComplianceStatementResponse:
    """Return regulatory research compliance metadata and audit guarantees."""
    return audit_service.get_compliance_statement()


@router.get(
    "/html",
    response_class=HTMLResponse,
    status_code=status.HTTP_200_OK,
    summary="Get standalone HTML clinical audit & system diagnostics dashboard",
)
def get_audit_html_dashboard(
    audit_service: AuditService = Depends(get_audit_service),
) -> HTMLResponse:
    """Render and serve live clinical audit dashboard HTML document."""
    diag = audit_service.run_deep_system_diagnostics()
    comp = audit_service.get_compliance_statement()
    html_content = audit_service.render_audit_html(diag=diag, comp=comp)
    return HTMLResponse(content=html_content)

