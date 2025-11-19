import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """Test basic health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "katadia-ai-ml"
    assert "version" in data
    
def test_detailed_health_check():
    """Test detailed health check endpoint."""
    response = client.get("/api/v1/health/detailed")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data
    
    # Check that components exist (will be implemented in later sprints)
    components = data["components"]
    assert "database" in components
    assert "stt_service" in components
    assert "ml_models" in components
    assert "cache" in components
    
def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["service"] == "KataDia AI - ML Service"
    assert data["status"] == "running"
