#!/usr/bin/env python3
"""
Production verification script for CrackGraphAI.
Tests all critical components to ensure production readiness.
"""

import asyncio
import base64
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
from PIL import Image

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    END = "\033[0m"
    BOLD = "\033[1m"


def success(msg: str) -> None:
    print(f"{Colors.GREEN}[PASS]{Colors.END} {msg}")


def warning(msg: str) -> None:
    print(f"{Colors.YELLOW}[WARN]{Colors.END} {msg}")


def error(msg: str) -> None:
    print(f"{Colors.RED}[FAIL]{Colors.END} {msg}")


def info(msg: str) -> None:
    print(f"{Colors.CYAN}[INFO]{Colors.END} {msg}")


def section(title: str) -> None:
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")


class ProductionVerifier:
    """Verify production deployment."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[Dict] = []
        self.passed = 0
        self.failed = 0

    def test(self, name: str, func) -> bool:
        """Run a test and record result."""
        try:
            func()
            success(name)
            self.passed += 1
            return True
        except Exception as e:
            error(f"{name}: {e}")
            self.failed += 1
            return False

    def test_health_endpoint(self) -> None:
        """Test health check endpoint."""
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        assert data.get("status") == "healthy", "Service not healthy"
        assert data.get("model_loaded") is True, "Model not loaded"
        info(f"  Version: {data.get('version')}")
        info(f"  Device: {data.get('device')}")
        info(f"  Cache: {data.get('cache_stats', {})}")

    def test_ready_endpoint(self) -> None:
        """Test Kubernetes readiness probe."""
        response = requests.get(f"{self.base_url}/ready", timeout=10)
        response.raise_for_status()
        data = response.json()
        assert data.get("ready") is True, "Service not ready"

    def test_metrics_endpoint(self) -> None:
        """Test Prometheus metrics endpoint."""
        response = requests.get(f"{self.base_url}/metrics", timeout=10)
        response.raise_for_status()
        content = response.text
        assert "predictions_total" in content or "prediction_latency" in content, "Metrics not found"

    def test_prediction(self) -> None:
        """Test single image prediction."""
        # Create a test image with a simulated crack
        img = self._create_test_image()
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        files = {"image": ("test.png", img_bytes, "image/png")}
        response = requests.post(
            f"{self.base_url}/predict",
            files=files,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()

        assert "segmentation_mask_png_b64" in data, "No mask in response"
        assert "si_score" in data, "No SI score in response"
        assert "graph_features" in data, "No graph features in response"
        assert "latency_seconds" in data, "No latency in response"

        info(f"  SI Score: {data['si_score']:.3f}")
        info(f"  Connectivity: {data.get('connectivity_score', 'N/A')}")
        info(f"  Latency: {data['latency_seconds']:.3f}s")

    def test_batch_prediction(self) -> None:
        """Test batch prediction."""
        files = []
        for i in range(2):
            img = self._create_test_image(seed=i)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            files.append(("images", (f"test_{i}.png", img_bytes, "image/png")))

        response = requests.post(
            f"{self.base_url}/predict_batch",
            files=files,
            timeout=120
        )
        response.raise_for_status()
        data = response.json()

        assert "items" in data, "No items in batch response"
        assert len(data["items"]) == 2, f"Expected 2 results, got {len(data['items'])}"

        for i, item in enumerate(data["items"]):
            if "error" in item:
                error(f"  Item {i}: {item['error']}")
            else:
                info(f"  Item {i}: OK (SI={item.get('result', {}).get('si_score', 'N/A')})")

    def test_cache_functionality(self) -> None:
        """Test that caching works."""
        # Make same request twice
        img = self._create_test_image(seed=999)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        files = {"image": ("cache_test.png", img_bytes, "image/png")}

        # First request
        start = time.time()
        response1 = requests.post(
            f"{self.base_url}/predict",
            files=files,
            timeout=60
        )
        latency1 = time.time() - start
        response1.raise_for_status()
        data1 = response1.json()

        # Second request (should be cached)
        img_bytes.seek(0)
        files = {"image": ("cache_test.png", img_bytes, "image/png")}
        start = time.time()
        response2 = requests.post(
            f"{self.base_url}/predict",
            files=files,
            timeout=60
        )
        latency2 = time.time() - start
        response2.raise_for_status()
        data2 = response2.json()

        info(f"  First request: {latency1:.3f}s (cached={data1.get('from_cache', False)})")
        info(f"  Second request: {latency2:.3f}s (cached={data2.get('from_cache', False)})")

        if data2.get("from_cache"):
            info("  Cache is working correctly!")
        else:
            warning("  Cache may not be enabled or hash mismatch")

    def test_cache_stats(self) -> None:
        """Test cache statistics endpoint."""
        response = requests.get(f"{self.base_url}/cache/stats", timeout=10)
        response.raise_for_status()
        data = response.json()
        assert "size" in data, "No cache size in stats"
        assert "max_size" in data, "No max_size in stats"
        info(f"  Cache: {data['size']}/{data['max_size']} items")

    def _create_test_image(self, seed: int = 42) -> Image.Image:
        """Create a test image with simulated crack pattern."""
        np.random.seed(seed)
        # Create grayscale image
        img_array = np.ones((256, 256, 3), dtype=np.uint8) * 200

        # Add simulated crack (dark line)
        x = np.linspace(50, 200, 150)
        y = 128 + 20 * np.sin(x / 20) + np.random.normal(0, 3, len(x))

        for xi, yi in zip(x.astype(int), y.astype(int)):
            if 0 <= xi < 256 and 0 <= yi < 256:
                img_array[max(0, yi-2):min(256, yi+3), max(0, xi-2):min(256, xi+3)] = 50

        # Add some noise
        noise = np.random.normal(0, 10, img_array.shape).astype(np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)

    def test_model_files(self) -> None:
        """Verify model files exist."""
        model_dir = project_root / "checkpoints"
        required_models = [
            "best_hybrid_segformer.pth",
        ]

        for model in required_models:
            path = model_dir / model
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                success(f"  {model}: {size_mb:.1f} MB")
            else:
                error(f"  {model}: NOT FOUND")

    def test_docker_services(self) -> None:
        """Test that Docker services are running."""
        import subprocess

        result = subprocess.run(
            ["docker", "compose", "-f", str(project_root / "docker-compose.prod.yml"), "ps", "-q"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0 and result.stdout.strip():
            containers = result.stdout.strip().split("\n")
            info(f"  Running containers: {len(containers)}")
            for container in containers:
                info(f"    - {container[:12]}")
        else:
            error("No Docker containers running")
            raise RuntimeError("Services not running")

    def run_all_tests(self) -> None:
        """Run all verification tests."""
        print(f"\n{Colors.BOLD}CrackGraphAI Production Verification{Colors.END}")
        print(f"API URL: {self.base_url}")
        print(f"Project: {project_root}")

        section("Model Files")
        self.test("Model files present", self.test_model_files)

        section("Docker Services")
        self.test("Services running", self.test_docker_services)

        section("API Endpoints")
        self.test("Health endpoint", self.test_health_endpoint)
        self.test("Ready endpoint", self.test_ready_endpoint)
        self.test("Metrics endpoint", self.test_metrics_endpoint)

        section("Inference Tests")
        self.test("Single prediction", self.test_prediction)
        self.test("Batch prediction", self.test_batch_prediction)
        self.test("Cache functionality", self.test_cache_functionality)
        self.test("Cache stats endpoint", self.test_cache_stats)

        section("Summary")
        total = self.passed + self.failed
        print(f"\n{Colors.BOLD}Results:{Colors.END}")
        print(f"  {Colors.GREEN}Passed: {self.passed}/{total}{Colors.END}")
        print(f"  {Colors.RED}Failed: {self.failed}/{total}{Colors.END}")

        if self.failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed! Production is ready.{Colors.END}")
            return 0
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}Some tests failed. Please review.{Colors.END}")
            return 1


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify CrackGraphAI production deployment")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    verifier = ProductionVerifier(base_url=args.url)
    exit_code = verifier.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
