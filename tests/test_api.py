"""Unit tests for CrackGraphAI API."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app, get_api_key

client = TestClient(app)


def create_test_image(size: tuple[int, int] = (256, 256)) -> io.BytesIO:
    """Create a test image in memory."""
    img = Image.new("RGB", size, color=(128, 128, 128))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes


class TestHealthEndpoints:
    """Test health and readiness endpoints."""

    def test_health_unhealthy_without_service(self, monkeypatch):
        """Test health check when service is not initialized."""
        import api.main
        # Temporarily set service to None
        original_service = api.main.service
        api.main.service = None

        response = client.get("/health")
        assert response.status_code == 503

        # Restore service
        api.main.service = original_service

    def test_ready_unhealthy_without_service(self, monkeypatch):
        """Test readiness probe when service is not initialized."""
        import api.main
        original_service = api.main.service
        api.main.service = None

        response = client.get("/ready")
        assert response.status_code == 503

        api.main.service = original_service

    def test_metrics_endpoint(self):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "prometheus" in response.headers.get("content-type", "")


class TestAuthentication:
    """Test authentication and authorization."""

    def test_dev_mode_allows_requests(self, monkeypatch):
        """Test that dev mode allows requests without auth."""
        monkeypatch.setenv("API_KEY", "dev-key-change-in-production")

        img_bytes = create_test_image()
        response = client.post(
            "/predict",
            files={"image": ("test.png", img_bytes, "image/png")},
        )
        # We expect 200 or 503 (if service not loaded in test)
        assert response.status_code in [200, 503]

    def test_missing_auth_header(self, monkeypatch):
        """Test request without auth header fails."""
        monkeypatch.setenv("API_KEY", "secure-key")

        img_bytes = create_test_image()
        response = client.post(
            "/predict",
            files={"image": ("test.png", img_bytes, "image/png")},
        )
        assert response.status_code == 401

    def test_invalid_api_key(self, monkeypatch):
        """Test request with invalid API key fails."""
        monkeypatch.setenv("API_KEY", "secure-key")

        img_bytes = create_test_image()
        response = client.post(
            "/predict",
            files={"image": ("test.png", img_bytes, "image/png")},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 401


class TestPrediction:
    """Test prediction endpoints."""

    def test_invalid_content_type(self):
        """Test prediction with invalid file type."""
        response = client.post(
            "/predict",
            files={"image": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_file_too_large(self, monkeypatch):
        """Test prediction with oversized file."""
        # Create a large fake image
        large_bytes = b"\x89PNG" + b"0" * (11 * 1024 * 1024)  # >10MB

        response = client.post(
            "/predict",
            files={"image": ("large.png", large_bytes, "image/png")},
        )
        assert response.status_code == 413

    def test_batch_too_large(self):
        """Test batch prediction with too many images."""
        files = [(f"image{i}", create_test_image(), "image/png") for i in range(15)]
        response = client.post(
            "/predict_batch",
            files=files,
        )
        assert response.status_code == 400


class TestRequestLogging:
    """Test request ID and logging headers."""

    def test_request_id_header(self):
        """Test that response includes X-Request-ID header."""
        response = client.get("/health")
        assert "x-request-id" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
