"""Unit and integration tests for Phase 14: Unified Clinical Central Station & Production Deployment."""

from pathlib import Path
from fastapi.testclient import TestClient

from backend.api.app import app

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_workstation_page_serving():
    """Verify GET /workstation returns valid clinical station HTML."""
    client = TestClient(app)
    resp = client.get("/workstation")

    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    html = resp.text

    assert "BioQuest Central Station" in html
    assert "Bed 01 (ICU)" in html
    assert "Lead II Electrocardiogram" in html
    assert "Smartphone Optical Photoplethysmogram" in html
    assert "Emergency Alert Feed" in html
    assert "Generate Report" in html


def test_root_page_serves_workstation():
    """Verify root GET / serves the clinical workstation as primary interface."""
    client = TestClient(app)
    resp = client.get("/")

    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "BioQuest Central Station" in resp.text


def test_multimodal_and_viewer_subroutes():
    """Verify auxiliary telemetry pages /multimodal and /viewer remain accessible."""
    client = TestClient(app)

    resp_multi = client.get("/multimodal")
    assert resp_multi.status_code == 200
    assert "text/html" in resp_multi.headers.get("content-type", "")

    resp_view = client.get("/viewer")
    assert resp_view.status_code == 200
    assert "text/html" in resp_view.headers.get("content-type", "")


def test_docker_configuration_integrity():
    """Verify production containerization assets (Dockerfile, docker-compose, .dockerignore)."""
    dockerfile_path = ROOT_DIR / "Dockerfile"
    compose_path = ROOT_DIR / "docker-compose.yml"
    ignore_path = ROOT_DIR / ".dockerignore"

    assert dockerfile_path.exists(), "Dockerfile must exist in project root"
    df_content = dockerfile_path.read_text(encoding="utf-8")
    assert "FROM python:" in df_content
    assert "EXPOSE 8000" in df_content
    assert "HEALTHCHECK" in df_content
    assert "uvicorn" in df_content

    assert compose_path.exists(), "docker-compose.yml must exist in project root"
    compose_content = compose_path.read_text(encoding="utf-8")
    assert "bioquest-telemetry" in compose_content
    assert "8000:8000" in compose_content
    assert "healthcheck" in compose_content

    assert ignore_path.exists(), ".dockerignore must exist in project root"
    ignore_content = ignore_path.read_text(encoding="utf-8")
    assert ".git" in ignore_content
    assert "__pycache__" in ignore_content


def test_workstation_api_meta_endpoint():
    """Verify GET /api returns full endpoint discovery metadata."""
    client = TestClient(app)
    resp = client.get("/api")

    assert resp.status_code == 200
    data = resp.json()
    assert "service" in data
    assert "version" in data
    assert "/api/v1/health" in data["endpoints"].values()
    assert "/ws/v1/ppg" in data["endpoints"].values()
