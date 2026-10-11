"""Real-time Cardiac Anomaly Alerting REST router."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.api.deps import get_alert_service
from backend.models.schemas import (
    AlertAcknowledgeResponse,
    AlertConfig,
    AlertListResponse,
    AlertSeverityEnum,
    AlertTypeEnum,
    CardiacAlert,
)
from backend.services.alert_service import AlertService

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("", response_model=AlertListResponse)
def list_alerts(
    severity: Optional[AlertSeverityEnum] = Query(None, description="Filter by severity level"),
    alert_type: Optional[AlertTypeEnum] = Query(None, description="Filter by alert category"),
    unacknowledged_only: bool = Query(False, description="Filter only unacknowledged alerts"),
    limit: int = Query(50, ge=1, le=500, description="Max alerts to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    service: AlertService = Depends(get_alert_service),
) -> AlertListResponse:
    """List recent cardiac anomaly alerts with optional filtering."""
    return service.list_alerts(
        severity=severity,
        alert_type=alert_type,
        unacknowledged_only=unacknowledged_only,
        limit=limit,
        offset=offset,
    )


@router.get("/active", response_model=List[CardiacAlert])
def get_active_alerts(
    service: AlertService = Depends(get_alert_service),
) -> List[CardiacAlert]:
    """Retrieve all active unacknowledged cardiac emergency alerts."""
    return service.get_active_alerts()


@router.post("/{alert_id}/acknowledge", response_model=AlertAcknowledgeResponse)
def acknowledge_alert(
    alert_id: str,
    service: AlertService = Depends(get_alert_service),
) -> AlertAcknowledgeResponse:
    """Mark an alert incident as reviewed and acknowledged."""
    res = service.acknowledge_alert(alert_id)
    if res is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert '{alert_id}' not found.",
        )
    return res


@router.get("/config", response_model=AlertConfig)
def get_alert_config(
    service: AlertService = Depends(get_alert_service),
) -> AlertConfig:
    """Get active clinical threshold configuration."""
    return service.get_config()


@router.put("/config", response_model=AlertConfig)
def update_alert_config(
    config: AlertConfig,
    service: AlertService = Depends(get_alert_service),
) -> AlertConfig:
    """Update active clinical threshold configuration."""
    return service.update_config(config)


@router.delete("", response_model=Dict[str, Any])
def clear_alerts(
    service: AlertService = Depends(get_alert_service),
) -> Dict[str, Any]:
    """Reset and clear all alert history (for test sessions)."""
    service.clear()
    return {"cleared": True, "message": "All alert records cleared."}
