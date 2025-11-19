"""
Production Optimizer Tests

Test suite untuk:
- Model quantization
- ONNX conversion
- Model profiling
- Latency analysis
- API endpoints
"""

import pytest
import asyncio
import torch
import tempfile
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from app.ml.optimization.production_optimizer import (
    EnglishModelOptimizer,
    ProductionOptimizationService,
    QuantizationConfig,
    OptimizationConfig,
    LatencyOptimization,
    PruningOptimizer,
    DistillationOptimizer,
    ENGLISH_OPTIMIZATION_CONFIGS,
    ModelMetrics,
    QuantizationType,
    ModelFormat
)


class TestOptimizationConfig:
    """Test OptimizationConfig dataclass."""
    
    def test_default_config(self):
        config = OptimizationConfig()
        assert config.quantization == True
        assert config.onnx_conversion == True
        assert config.torch_script == True
        assert config.model_compression == 0.5
        assert config.inference_time_target == 0.5
        assert config.memory_limit_mb == 512
    
    def test_custom_config(self):
        config = OptimizationConfig(
            quantization=False,
            model_compression=0.3,
            memory_limit_mb=256
        )
        assert config.quantization == False
        assert config.model_compression == 0.3
        assert config.memory_limit_mb == 256


class TestModelMetrics:
    """Test ModelMetrics dataclass."""
    
    def test_metrics_creation(self):
        metrics = ModelMetrics(
            model_size_mb=100.5,
            latency_ms=250.0,
            throughput_qps=4.0,
            memory_usage_mb=512.0
        )
        assert metrics.model_size_mb == 100.5
        assert metrics.latency_ms == 250.0
        assert metrics.throughput_qps == 4.0
        assert metrics.memory_usage_mb == 512.0
        assert metrics.accuracy_drop_percent == 0.0
    
    def test_metrics_to_dict(self):
        metrics = ModelMetrics(
            model_size_mb=100,
            latency_ms=250,
            throughput_qps=4.0,
            memory_usage_mb=512
        )
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert "model_size_mb" in metrics_dict
        assert "timestamp" in metrics_dict


class TestEnglishModelOptimizer:
    """Test EnglishModelOptimizer class."""
    
    @pytest.fixture
    def optimizer(self):
        config = OptimizationConfig(
            quantization=True,
            onnx_conversion=False,
            torch_script=False
        )
        return EnglishModelOptimizer(config)
    
    def test_optimizer_initialization(self, optimizer):
        assert optimizer.config is not None
        assert optimizer.optimized_models == {}
    
    @pytest.mark.asyncio
    async def test_benchmark_model_basic(self, optimizer):
        # Create a simple dummy model
        model = torch.nn.Linear(10, 5)
        model.eval()
        
        metrics = await optimizer._benchmark_model(model, "wav2vec2", num_samples=10)
        
        assert "avg_inference_time" in metrics
        assert "p95_inference_time" in metrics
        assert "p99_inference_time" in metrics
        assert metrics["avg_inference_time"] > 0
        assert metrics["p95_inference_time"] >= metrics["avg_inference_time"]
        assert metrics["p99_inference_time"] >= metrics["p95_inference_time"]
    
    def test_calculate_improvement(self, optimizer):
        base_metrics = {
            "avg_inference_time": 1.0,  # 1 second
            "peak_memory_mb": 512,
            "model_size_mb": 400
        }
        
        optimized_metrics = {
            "avg_inference_time": 0.5,  # 0.5 seconds
            "peak_memory_mb": 256,
            "model_size_mb": 100
        }
        
        improvements = optimizer._calculate_improvement(base_metrics, optimized_metrics)
        
        # Verify calculation: (1.0 - 0.5) / 1.0 * 100 = 50%
        assert improvements["inference_time_improvement"] == 50.0
        # Verify memory: (512 - 256) / 512 * 100 = 50%
        assert improvements["memory_improvement"] == 50.0
        # Verify size: (400 - 100) / 400 * 100 = 75%
        assert improvements["size_improvement"] == 75.0
    
    def test_find_best_optimized_model(self, optimizer):
        results = {
            "optimized_models": {
                "quantized": {
                    "path": "./quantized",
                    "metrics": {
                        "avg_inference_time": 0.2,
                        "peak_memory_mb": 128
                    }
                },
                "onnx": {
                    "path": "./onnx",
                    "metrics": {
                        "avg_inference_time": 0.3,
                        "peak_memory_mb": 100
                    }
                }
            }
        }
        
        best = optimizer._find_best_optimized_model(results)
        
        # Quantized should be best due to speed (more weight)
        assert best["type"] == "quantized"


class TestLatencyOptimization:
    """Test LatencyOptimization class."""
    
    @pytest.fixture
    def latency_analyzer(self):
        return LatencyOptimization()
    
    def test_initialization(self, latency_analyzer):
        assert latency_analyzer.latency_profile == {}
    
    def test_identify_bottlenecks(self, latency_analyzer):
        bottlenecks = [
            {
                "stage": "forward_pass",
                "latency_ms": 400,
                "percentage": 80.0
            }
        ]
        
        recommendations = latency_analyzer._generate_recommendations(bottlenecks)
        
        assert isinstance(recommendations, list)
        assert any("quantization" in r.lower() for r in recommendations)


class TestPruningOptimizer:
    """Test PruningOptimizer class."""
    
    def test_magnitude_pruning_ratio_validation(self):
        model = torch.nn.Linear(10, 5)
        
        # Test invalid pruning rates
        with pytest.raises(Exception):
            PruningOptimizer.magnitude_pruning(model, pruning_rate=-0.1)
        
        with pytest.raises(Exception):
            PruningOptimizer.magnitude_pruning(model, pruning_rate=1.5)


class TestDistillationOptimizer:
    """Test DistillationOptimizer class."""
    
    @pytest.fixture
    def distillation_setup(self):
        teacher = torch.nn.Linear(10, 5)
        student = torch.nn.Linear(10, 5)
        return DistillationOptimizer(teacher, student, temperature=3.0)
    
    def test_initialization(self, distillation_setup):
        assert distillation_setup.temperature == 3.0
        assert distillation_setup.teacher_model is not None
        assert distillation_setup.student_model is not None


class TestOptimizationProfiles:
    """Test optimization profiles."""
    
    def test_production_profile(self):
        config = ENGLISH_OPTIMIZATION_CONFIGS["production"]
        assert config.quantization == True
        assert config.onnx_conversion == True
        assert config.inference_time_target == 0.3
        assert config.memory_limit_mb == 256
    
    def test_mobile_profile(self):
        config = ENGLISH_OPTIMIZATION_CONFIGS["mobile"]
        assert config.quantization == True
        assert config.model_compression == 0.7
        assert config.memory_limit_mb == 128
    
    def test_development_profile(self):
        config = ENGLISH_OPTIMIZATION_CONFIGS["development"]
        assert config.quantization == False
        assert config.onnx_conversion == False
        assert config.memory_limit_mb == 1024


class TestModelFormats:
    """Test ModelFormat enum."""
    
    def test_pytorch_format(self):
        assert ModelFormat.PYTORCH.value == "pytorch"
    
    def test_onnx_format(self):
        assert ModelFormat.ONNX.value == "onnx"


class TestQuantizationTypes:
    """Test QuantizationType enum."""
    
    def test_int8_type(self):
        assert QuantizationType.INT8.value == "int8"
    
    def test_fp16_type(self):
        assert QuantizationType.FP16.value == "fp16"
    
    def test_dynamic_type(self):
        assert QuantizationType.DYNAMIC.value == "dynamic"


# ==================== API ENDPOINT TESTS ====================

@pytest.mark.asyncio
class TestOptimizerEndpoints:
    """Test optimizer API endpoints."""
    
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        response = client.get("/api/v1/optimizer/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "production_optimizer"
    
    def test_profiles_endpoint(self, client):
        response = client.get("/api/v1/optimizer/profiles")
        assert response.status_code == 200
        data = response.json()
        assert "production" in data
        assert "mobile" in data
        assert "development" in data
    
    def test_optimize_endpoint_missing_model(self, client):
        response = client.post(
            "/api/v1/optimizer/optimize",
            json={
                "model_path": "./nonexistent_model",
                "optimization_type": "comprehensive"
            }
        )
        assert response.status_code == 404
    
    def test_quantize_endpoint_invalid_rate(self, client):
        # Create a temporary model file
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            model_path = f.name
        
        try:
            response = client.post(
                "/api/v1/optimizer/quantize",
                json={
                    "model_path": model_path,
                    "quantization_type": "int8",
                    "output_path": "./output",
                    "pruning_rate": 1.5  # Invalid
                }
            )
            # Should handle invalid rate
        finally:
            Path(model_path).unlink(missing_ok=True)


class TestMetricsCalculations:
    """Test metrics calculations."""
    
    def test_latency_improvement_calculation(self):
        base_latency = 1000  # 1000 ms
        optimized_latency = 500  # 500 ms
        
        improvement = ((base_latency - optimized_latency) / base_latency) * 100
        
        assert improvement == 50.0  # 50% improvement
    
    def test_size_reduction_calculation(self):
        base_size = 400  # MB
        optimized_size = 100  # MB
        
        reduction = ((base_size - optimized_size) / base_size) * 100
        
        assert reduction == 75.0  # 75% reduction
    
    def test_speedup_calculation(self):
        base_latency = 500  # ms
        optimized_latency = 100  # ms
        
        speedup = base_latency / optimized_latency
        
        assert speedup == 5.0  # 5x speedup


class TestErrorHandling:
    """Test error handling."""
    
    @pytest.mark.asyncio
    async def test_missing_model_file(self):
        optimizer = EnglishModelOptimizer()
        
        with pytest.raises(Exception):
            await optimizer.optimize_for_production(
                model_path="./nonexistent_model",
                output_path="./output"
            )
    
    @pytest.mark.asyncio
    async def test_invalid_model_type(self):
        optimizer = EnglishModelOptimizer()
        
        with pytest.raises(ValueError):
            optimizer._load_model("./model", model_type="invalid_type")


# ==================== INTEGRATION TESTS ====================

@pytest.mark.asyncio
class TestIntegration:
    """Integration tests."""
    
    async def test_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        )
        
        # Initialize optimizer
        config = OptimizationConfig(
            quantization=True,
            onnx_conversion=False,
            torch_script=False
        )
        optimizer = EnglishModelOptimizer(config)
        
        # Save model temporarily
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            model.save_pretrained(str(model_path))
            
            # Run basic profiling
            loaded_model = optimizer._load_model(str(model_path), "wav2vec2")
            metrics = await optimizer._benchmark_model(loaded_model, "wav2vec2", num_samples=5)
            
            # Verify metrics
            assert metrics["avg_inference_time"] > 0
            assert "p95_inference_time" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
