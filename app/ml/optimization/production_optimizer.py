import torch
import onnx
import onnxruntime as ort
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import Dict, List, Tuple, Any
from pathlib import Path
import structlog
import numpy as np
import time
from dataclasses import dataclass
import json

logger = structlog.get_logger()

@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""
    quantization: bool = True
    onnx_conversion: bool = True
    torch_script: bool = True
    model_compression: float = 0.5  # Target compression ratio
    inference_time_target: float = 0.5  # seconds
    memory_limit_mb: int = 512  # Maximum memory usage

class EnglishModelOptimizer:
    """Optimization service for English pronunciation models."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.optimized_models = {}
        
    async def optimize_for_production(
        self, 
        model_path: str, 
        output_path: str,
        model_type: str = "wav2vec2"
    ) -> Dict[str, Any]:
        """
        Optimize English pronunciation model for production deployment.
        
        Args:
            model_path: Path to trained model
            output_path: Directory for optimized models
            model_type: Type of model to optimize
            
        Returns:
            Optimization results with performance metrics
        """
        try:
            logger.info(f"Starting production optimization for {model_type}")
            
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Load original model
            original_model = self._load_model(model_path, model_type)
            
            # Benchmark original model
            original_metrics = await self._benchmark_model(original_model, model_type)
            logger.info(f"Original model metrics: {original_metrics}")
            
            optimization_results = {
                "original_metrics": original_metrics,
                "optimized_models": {},
                "performance_improvements": {}
            }
            
            # 1. Dynamic Quantization
            if self.config.quantization:
                quantized_path = output_dir / f"{model_type}_quantized"
                quantized_model = await self._quantize_model(original_model, str(quantized_path))
                quantized_metrics = await self._benchmark_model(quantized_model, model_type)
                
                optimization_results["optimized_models"]["quantized"] = {
                    "path": str(quantized_path),
                    "metrics": quantized_metrics
                }
                
                improvement = self._calculate_improvement(original_metrics, quantized_metrics)
                optimization_results["performance_improvements"]["quantization"] = improvement
            
            # 2. ONNX Conversion
            if self.config.onnx_conversion:
                onnx_path = output_dir / f"{model_type}.onnx"
                await self._convert_to_onnx(original_model, str(onnx_path))
                
                # Create ONNX inference session
                onnx_session = self._create_onnx_session(str(onnx_path))
                onnx_metrics = await self._benchmark_onnx_session(onnx_session)
                
                optimization_results["optimized_models"]["onnx"] = {
                    "path": str(onnx_path),
                    "metrics": onnx_metrics
                }
                
                improvement = self._calculate_improvement(original_metrics, onnx_metrics)
                optimization_results["performance_improvements"]["onnx"] = improvement
            
            # 3. TorchScript Conversion
            if self.config.torch_script:
                script_path = output_dir / f"{model_type}_scripted.pt"
                scripted_model = await self._convert_to_torchscript(original_model, str(script_path))
                scripted_metrics = await self._benchmark_model(scripted_model, model_type)
                
                optimization_results["optimized_models"]["torchscript"] = {
                    "path": str(script_path),
                    "metrics": scripted_metrics
                }
                
                improvement = self._calculate_improvement(original_metrics, scripted_metrics)
                optimization_results["performance_improvements"]["torchscript"] = improvement
            
            # Find best performing optimized model
            best_model = self._find_best_optimized_model(optimization_results)
            optimization_results["recommended_model"] = best_model
            
            # Save optimization metadata
            metadata_path = output_dir / "optimization_metadata.json"
            self._save_metadata(optimization_results, str(metadata_path))
            
            logger.info(f"Optimization completed. Recommended: {best_model['type']}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            raise
    
    def _load_model(self, model_path: str, model_type: str):
        """Load original model for optimization."""
        try:
            if model_type == "wav2vec2":
                # Load Wav2Vec2 model
                model = Wav2Vec2ForCTC.from_pretrained(model_path)
                model.eval()  # Set to evaluation mode
                return model
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    async def _benchmark_model(
        self, 
        model, 
        model_type: str, 
        num_samples: int = 100
    ) -> Dict[str, float]:
        """Benchmark model performance metrics."""
        
        # Create dummy input
        if model_type == "wav2vec2":
            # Create dummy audio input
            dummy_input = torch.randn(1, 16000)  # 1 second of audio at 16kHz
        else:
            dummy_input = torch.randn(1, 16, 100)  # Generic dummy input
        
        metrics = {}
        
        # Inference time
        inference_times = []
        model.eval()
        
        with torch.no_grad():
            for _ in range(num_samples):
                start_time = time.perf_counter()
                _ = model(dummy_input)
                end_time = time.perf_counter()
                inference_times.append(end_time - start_time)
        
        metrics["avg_inference_time"] = float(np.mean(inference_times))
        metrics["p95_inference_time"] = float(np.percentile(inference_times, 95))
        metrics["p99_inference_time"] = float(np.percentile(inference_times, 99))
        
        # Memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()
            metrics["peak_memory_mb"] = float((peak_memory - start_memory) / 1024 / 1024)
        else:
            # CPU memory estimation (simplified)
            import psutil
            process = psutil.Process()
            baseline_memory = process.memory_info().rss
            _ = model(dummy_input)
            current_memory = process.memory_info().rss
            metrics["peak_memory_mb"] = float((current_memory - baseline_memory) / 1024 / 1024)
        
        # Model size
        if hasattr(model, 'parameters'):
            model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
            metrics["model_size_mb"] = float(model_size_mb)
        
        return metrics
    
    async def _quantize_model(self, model: torch.nn.Module, output_path: str) -> torch.nn.Module:
        """Apply dynamic quantization to the model."""
        logger.info(f"Applying dynamic quantization to model")
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # Quantize linear layers
            dtype=torch.qint8
        )
        
        # Save quantized model
        quantized_model.save_pretrained(output_path)
        
        logger.info(f"Quantized model saved to {output_path}")
        return quantized_model
    
    async def _convert_to_onnx(self, model: torch.nn.Module, output_path: str):
        """Convert PyTorch model to ONNX format."""
        logger.info(f"Converting model to ONNX format")
        
        # Create dummy input for ONNX export
        dummy_input = torch.randn(1, 16000)  # Adjust based on actual input
        
        # Set model to evaluation mode
        model.eval()
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,  # Use appropriate ONNX version
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'audio_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"ONNX model saved and verified: {output_path}")
    
    def _create_onnx_session(self, onnx_path: str) -> ort.InferenceSession:
        """Create ONNX Runtime inference session."""
        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 4  # Adjust based on CPU cores
        
        # Execution providers (GPU if available, otherwise CPU)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        return session
    
    async def _benchmark_onnx_session(self, session: ort.InferenceSession, num_samples: int = 100) -> Dict[str, float]:
        """Benchmark ONNX inference session."""
        
        # Create dummy input
        input_name = session.get_inputs()[0].name
        dummy_input = {
            input_name: np.random.randn(1, 16000).astype(np.float32)
        }
        
        # Warm up
        for _ in range(10):
            _ = session.run(None, dummy_input)
        
        # Benchmark inference
        inference_times = []
        for _ in range(num_samples):
            start_time = time.perf_counter()
            _ = session.run(None, dummy_input)
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)
        
        metrics = {
            "avg_inference_time": float(np.mean(inference_times)),
            "p95_inference_time": float(np.percentile(inference_times, 95)),
            "p99_inference_time": float(np.percentile(inference_times, 99)),
            "model_size_mb": float(Path(session._model_path).stat().st_size / 1024 / 1024) if hasattr(session, '_model_path') else 0
        }
        
        return metrics
    
    async def _convert_to_torchscript(self, model: torch.nn.Module, output_path: str) -> torch.jit.ScriptModule:
        """Convert PyTorch model to TorchScript."""
        logger.info(f"Converting model to TorchScript")
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 16000)
        
        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
        
        # Verify traced model
        traced_output = traced_model(dummy_input)
        original_output = model(dummy_input)
        
        assert torch.allclose(traced_output, original_output, atol=1e-5), "TorchScript tracing failed"
        
        # Save traced model
        torch.jit.save(traced_model, output_path)
        
        logger.info(f"TorchScript model saved: {output_path}")
        return traced_model
    
    def _calculate_improvement(self, base_metrics: Dict, optimized_metrics: Dict) -> Dict[str, float]:
        """Calculate performance improvements."""
        improvements = {}
        
        # Inference time improvement
        base_time = base_metrics.get("avg_inference_time", 0)
        opt_time = optimized_metrics.get("avg_inference_time", 0)
        if base_time > 0:
            improvements["inference_time_improvement"] = float((base_time - opt_time) / base_time * 100)
        
        # Memory usage improvement
        base_memory = base_metrics.get("peak_memory_mb", 0)
        opt_memory = optimized_metrics.get("peak_memory_mb", 0)
        if base_memory > 0:
            improvements["memory_improvement"] = float((base_memory - opt_memory) / base_memory * 100)
        
        # Model size improvement
        base_size = base_metrics.get("model_size_mb", 0)
        opt_size = optimized_metrics.get("model_size_mb", 0)
        if base_size > 0:
            improvements["size_improvement"] = float((base_size - opt_size) / base_size * 100)
        
        return improvements
    
    def _find_best_optimized_model(self, results: Dict) -> Dict[str, Any]:
        """Find the best performing optimized model."""
        best_model = None
        best_score = -1
        
        for model_type, model_data in results["optimized_models"].items():
            metrics = model_data["metrics"]
            
            # Calculate score based on speed and memory efficiency
            time_score = 1.0 / max(metrics.get("avg_inference_time", 1), 0.01)
            memory_score = 1.0 / max(metrics.get("peak_memory_mb", 1), 1)
            
            total_score = time_score * 0.6 + memory_score * 0.4  # Weight time more heavily
            
            if total_score > best_score:
                best_score = total_score
                best_model = {
                    "type": model_type,
                    "path": model_data["path"],
                    "metrics": metrics,
                    "score": total_score
                }
        
        return best_model or {"type": "none"}
    
    def _save_metadata(self, results: Dict, output_path: str):
        """Save optimization metadata."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Optimization metadata saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

class ProductionOptimizedModelLoader:
    """Loader for optimized models in production."""
    
    def __init__(self, optimized_model_dir: str):
        self.model_dir = Path(optimized_model_dir)
        self.model_cache = {}
        
    async def load_optimized_model(self, model_type: str = "wav2vec2") -> Any:
        """Load the best optimized model for production use."""
        try:
            # Load optimization metadata
            metadata_path = self.model_dir / "optimization_metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Optimization metadata not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            recommended = metadata.get("recommended_model")
            if not recommended or recommended["type"] == "none":
                raise ValueError("No optimized model available")
            
            model_path = recommended["path"]
            
            # Load based on optimization type
            if recommended["type"] == "onnx":
                return self._load_onnx_model(model_path)
            elif recommended["type"] == "torchscript":
                return self._load_torchscript_model(model_path)
            else:
                return self._load_quantized_model(model_path, model_type)
                
        except Exception as e:
            logger.error(f"Failed to load optimized model: {e}")
            raise
    
    def _load_onnx_model(self, model_path: str):
        """Load ONNX model for inference."""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        
        session = ort.InferenceSession(model_path, sess_options, providers=providers)
        return session
    
    def _load_torchscript_model(self, model_path: str):
        """Load TorchScript model."""
        return torch.jit.load(model_path, map_location='cpu')
    
    def _load_quantized_model(self, model_path: str, model_type: str):
        """Load quantized PyTorch model."""
        if model_type == "wav2vec2":
            return Wav2Vec2ForCTC.from_pretrained(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

# English-specific optimization configurations
ENGLISH_OPTIMIZATION_CONFIGS = {
    "production": OptimizationConfig(
        quantization=True,
        onnx_conversion=True,
        torch_script=True,
        model_compression=0.3,
        inference_time_target=0.3,
        memory_limit_mb=256
    ),
    "development": OptimizationConfig(
        quantization=False,
        onnx_conversion=False,
        torch_script=False,
        model_compression=0.1,
        inference_time_target=1.0,
        memory_limit_mb=1024
    ),
    "mobile": OptimizationConfig(
        quantization=True,
        onnx_conversion=False,
        torch_script=False,
        model_compression=0.7,
        inference_time_target=0.2,
        memory_limit_mb=128
    )
}

async def main():
    """Main function for model optimization."""
    # Example usage
    model_path = "./models/wav2vec2-english-v1"
    output_path = "./models/optimized"
    
    optimizer = EnglishModelOptimizer(ENGLISH_OPTIMIZATION_CONFIGS["production"])
    
    try:
        results = await optimizer.optimize_for_production(
            model_path=model_path,
            output_path=output_path,
            model_type="wav2vec2"
        )
        
        logger.info("Optimization completed successfully!")
        logger.info(f"Recommended model: {results['recommended_model']['type']}")
        logger.info(f"Inference time: {results['recommended_model']['metrics']['avg_inference_time']:.3f}s")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
