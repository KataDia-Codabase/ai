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
            logger.info("Starting production optimization", model_type=model_type)
            
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            original_model = self._load_model(model_path, model_type)
            
            original_metrics = await self._benchmark_model(original_model, model_type)
            logger.info("Original model metrics", metrics=original_metrics)
            
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
            
            # ONNX Conversion
            if self.config.onnx_conversion:
                try:
                    onnx_path = output_dir / f"{model_type}.onnx"
                    await self._convert_to_onnx(original_model, str(onnx_path))
                    
                    onnx_session = self._create_onnx_session(str(onnx_path))
                    onnx_metrics = await self._benchmark_onnx_session(onnx_session)
                    
                    optimization_results["optimized_models"]["onnx"] = {
                        "path": str(onnx_path),
                        "metrics": onnx_metrics
                    }
                    
                    improvement = self._calculate_improvement(original_metrics, onnx_metrics)
                    optimization_results["performance_improvements"]["onnx"] = improvement
                except Exception as e:
                    logger.warning("ONNX conversion skipped", error=str(e))
            
            # TorchScript Conversion
            if self.config.torch_script:
                try:
                    script_path = output_dir / f"{model_type}_scripted.pt"
                    scripted_model = await self._convert_to_torchscript(original_model, str(script_path))
                    scripted_metrics = await self._benchmark_model(scripted_model, model_type)
                    
                    optimization_results["optimized_models"]["torchscript"] = {
                        "path": str(script_path),
                        "metrics": scripted_metrics
                    }
                    
                    improvement = self._calculate_improvement(original_metrics, scripted_metrics)
                    optimization_results["performance_improvements"]["torchscript"] = improvement
                except Exception as e:
                    logger.warning("TorchScript conversion skipped", error=str(e))
            
            # Find best performing optimized model
            best_model = self._find_best_optimized_model(optimization_results)
            optimization_results["recommended_model"] = best_model
            
            # Save optimization metadata
            metadata_path = output_dir / "optimization_metadata.json"
            self._save_metadata(optimization_results, str(metadata_path))
            
            logger.info("Optimization completed", recommended=best_model['type'])
            return optimization_results
            
        except Exception as e:
            logger.error("Model optimization failed", error=str(e))
            raise
    
    def _load_model(self, model_path: str, model_type: str):
        """Load original model for optimization."""
        try:
            if model_type == "wav2vec2":
                model = Wav2Vec2ForCTC.from_pretrained(model_path)
                model.eval()
                return model
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            logger.error("Failed to load model", path=model_path, error=str(e))
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
        logger.info("Applying dynamic quantization to model")
        
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            quantized_model.save_pretrained(output_path)
        except AttributeError:
            torch.save(quantized_model.state_dict(), f"{output_path}.pth")
            logger.info("Quantized model saved", path=f"{output_path}.pth")
        
        logger.info("Quantized model saved", path=output_path)
        return quantized_model
    
    async def _convert_to_onnx(self, model: torch.nn.Module, output_path: str):
        """Convert PyTorch model to ONNX format."""
        try:
            logger.info("Converting model to ONNX format")
            
            dummy_input = torch.randn(1, 16000)
            
            model.eval()
            
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size', 1: 'audio_length'},
                    'output': {0: 'batch_size', 1: 'sequence_length'}
                },
                verbose=False
            )
            
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            logger.info("ONNX model saved and verified", path=output_path)
        except Exception as e:
            logger.warning("ONNX conversion failed", error=str(e))
            raise
    
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
        logger.info("Converting model to TorchScript")
        
        model.eval()
        
        dummy_input = torch.randn(1, 16000)
        
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
        
        traced_output = traced_model(dummy_input)
        original_output = model(dummy_input)
        
        assert torch.allclose(traced_output, original_output, atol=1e-5), "TorchScript tracing failed"
        
        torch.jit.save(traced_model, output_path)
        
        logger.info("TorchScript model saved", path=output_path)
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

# ==================== ADVANCED OPTIMIZATION FEATURES ====================

class DistillationOptimizer:
    """Knowledge distillation for model compression."""
    
    def __init__(self, teacher_model, student_model, temperature: float = 3.0):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        
    def distillation_loss(self, logits_student, logits_teacher):
        """Calculate distillation loss."""
        # Soft targets from teacher
        soft_targets = torch.nn.functional.softmax(logits_teacher / self.temperature, dim=-1)
        
        # KL divergence loss
        distillation_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(logits_student / self.temperature, dim=-1),
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return distillation_loss
    
    async def distill(self, train_loader, num_epochs: int = 10, learning_rate: float = 1e-4):
        """Train student model with knowledge distillation."""
        optimizer = torch.optim.Adam(self.student_model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Forward pass through both models
                with torch.no_grad():
                    teacher_output = self.teacher_model(**batch)
                
                student_output = self.student_model(**batch)
                
                # Calculate distillation loss
                loss = self.distillation_loss(
                    student_output.logits,
                    teacher_output.logits
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Distillation epoch {epoch + 1}/{num_epochs}, loss: {avg_loss:.4f}")
        
        return self.student_model


class PruningOptimizer:
    """Structured and unstructured pruning for model optimization."""
    
    @staticmethod
    def structured_pruning(model, pruning_rate: float = 0.3):
        """Remove entire filters or channels."""
        import torch.nn.utils.prune as prune
        
        pruned_model = model
        
        for module in pruned_model.modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.structured_prune(
                    module,
                    pruning_method=prune.LnStructured,
                    amount=pruning_rate,
                    n=2,
                    dim=0
                )
        
        return pruned_model
    
    @staticmethod
    def unstructured_pruning(model, pruning_rate: float = 0.3):
        """Remove individual weights."""
        import torch.nn.utils.prune as prune
        
        parameters_to_prune = [
            (module, 'weight') for module in model.modules()
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))
        ]
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_rate
        )
        
        # Remove reparameterization to make pruning permanent
        for module, name in parameters_to_prune:
            prune.remove(module, name)
        
        return model


class BatchOptimization:
    """Batch processing optimization."""
    
    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.optimal_batch_size = None
    
    async def find_optimal_batch_size(
        self,
        sample_input,
        max_memory_mb: float = 2048,
        min_batch_size: int = 1,
        max_batch_size: int = 256
    ) -> int:
        """Find optimal batch size for GPU/CPU."""
        
        current_batch_size = min_batch_size
        optimal_size = min_batch_size
        
        while current_batch_size <= max_batch_size:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                # Create batched input
                batch_input = torch.cat([sample_input] * current_batch_size, dim=0)
                
                # Test forward pass
                with torch.no_grad():
                    _ = self.model(batch_input)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024
                    
                    if memory_used <= max_memory_mb:
                        optimal_size = current_batch_size
                
                current_batch_size *= 2
                
            except RuntimeError:
                # Out of memory
                break
        
        self.optimal_batch_size = optimal_size
        logger.info(f"Optimal batch size found: {optimal_size}")
        return optimal_size
    
    async def optimize_batch_processing(self, data_loader, batch_multiplier: int = 4):
        """Optimize batch processing through gradient accumulation."""
        
        accumulation_steps = batch_multiplier
        logger.info(f"Gradient accumulation with {accumulation_steps} steps")
        
        return accumulation_steps


class LatencyOptimization:
    """End-to-end latency optimization."""
    
    def __init__(self):
        self.latency_profile = {}
    
    async def profile_latency_breakdown(self, model, input_data):
        """Profile latency at different stages."""
        
        stages = {
            "preprocessing": 0.0,
            "forward_pass": 0.0,
            "postprocessing": 0.0
        }
        
        # Profile preprocessing
        start = time.perf_counter()
        # Preprocessing operations
        preprocessed = self._preprocess(input_data)
        stages["preprocessing"] = time.perf_counter() - start
        
        # Profile forward pass
        start = time.perf_counter()
        with torch.no_grad():
            output = model(preprocessed)
        stages["forward_pass"] = time.perf_counter() - start
        
        # Profile postprocessing
        start = time.perf_counter()
        result = self._postprocess(output)
        stages["postprocessing"] = time.perf_counter() - start
        
        return stages
    
    def _preprocess(self, input_data):
        """Placeholder for preprocessing."""
        return input_data
    
    def _postprocess(self, output):
        """Placeholder for postprocessing."""
        return output
    
    async def identify_bottlenecks(self, model, input_data, threshold_percent: float = 20.0):
        """Identify bottlenecks in the pipeline."""
        
        stages = await self.profile_latency_breakdown(model, input_data)
        total_time = sum(stages.values())
        
        bottlenecks = []
        for stage, latency in stages.items():
            percent = (latency / total_time) * 100
            if percent > threshold_percent:
                bottlenecks.append({
                    "stage": stage,
                    "latency_ms": latency * 1000,
                    "percentage": percent
                })
        
        logger.info(f"Identified {len(bottlenecks)} bottlenecks")
        return bottlenecks
    
    def _generate_recommendations(self, bottlenecks: List[Dict]) -> List[str]:
        """Generate recommendations based on bottlenecks."""
        recommendations = []
        
        for bottleneck in bottlenecks:
            stage = bottleneck["stage"]
            percent = bottleneck["percentage"]
            
            if stage == "preprocessing" and percent > 20:
                recommendations.append("Consider optimizing audio preprocessing pipeline")
            elif stage == "forward_pass" and percent > 20:
                recommendations.append("Consider model quantization or pruning")
            elif stage == "postprocessing" and percent > 20:
                recommendations.append("Optimize post-processing algorithms")
        
        return recommendations


class EnhancedProductionOptimizer:
    """Enhanced production optimizer with advanced techniques."""
    
    def __init__(self):
        self.base_optimizer = EnglishModelOptimizer()
        self.distillation_optimizer = None
        self.pruning_optimizer = PruningOptimizer()
        self.batch_optimizer = None
        self.latency_optimizer = LatencyOptimization()
    
    async def comprehensive_optimization(
        self,
        model_path: str,
        output_dir: str,
        enable_distillation: bool = False,
        enable_pruning: bool = True,
        enable_batch_opt: bool = True,
        teacher_model_path: str = None
    ) -> Dict[str, Any]:
        """Comprehensive optimization pipeline."""
        
        results = {
            "base_optimization": None,
            "distillation": None,
            "pruning": None,
            "batch_optimization": None,
            "latency_analysis": None,
            "final_metrics": None
        }
        
        # Step 1: Base optimization
        logger.info("Step 1: Running base optimization...")
        results["base_optimization"] = await self.base_optimizer.optimize_for_production(
            model_path, output_dir
        )
        
        # Step 2: Knowledge distillation (optional)
        if enable_distillation and teacher_model_path:
            logger.info("Step 2: Running knowledge distillation...")
            teacher_model = self.base_optimizer._load_model(teacher_model_path, "wav2vec2")
            student_model = self.base_optimizer._load_model(model_path, "wav2vec2")
            
            self.distillation_optimizer = DistillationOptimizer(teacher_model, student_model)
            # Would need train_loader from actual training data
            # distilled_model = await self.distillation_optimizer.distill(train_loader)
            results["distillation"] = {"status": "configured"}
        
        # Step 3: Pruning (optional)
        if enable_pruning:
            logger.info("Step 3: Running pruning...")
            loaded_model = self.base_optimizer._load_model(model_path, "wav2vec2")
            pruned_model = self.pruning_optimizer.unstructured_pruning(loaded_model, 0.3)
            
            pruned_path = Path(output_dir) / "model_pruned.pt"
            
            # Save pruned model
            try:
                pruned_model.save_pretrained(str(pruned_path))
            except AttributeError:
                torch.save(pruned_model.state_dict(), str(pruned_path))
            
            pruned_metrics = await self.base_optimizer._benchmark_model(pruned_model, "wav2vec2")
            results["pruning"] = {
                "path": str(pruned_path),
                "metrics": pruned_metrics
            }
        
        # Step 4: Batch optimization (optional)
        if enable_batch_opt:
            logger.info("Step 4: Running batch optimization...")
            loaded_model = self.base_optimizer._load_model(model_path, "wav2vec2")
            self.batch_optimizer = BatchOptimization(loaded_model)
            
            sample_input = torch.randn(1, 16000)
            optimal_batch_size = await self.batch_optimizer.find_optimal_batch_size(sample_input)
            results["batch_optimization"] = {"optimal_batch_size": optimal_batch_size}
        
        # Step 5: Latency analysis
        logger.info("Step 5: Running latency analysis...")
        loaded_model = self.base_optimizer._load_model(model_path, "wav2vec2")
        sample_input = torch.randn(1, 16000)
        
        bottlenecks = await self.latency_optimizer.identify_bottlenecks(loaded_model, sample_input)
        results["latency_analysis"] = {
            "bottlenecks": bottlenecks,
            "recommendations": self._generate_latency_recommendations(bottlenecks)
        }
        
        # Final metrics
        logger.info("Computing final metrics...")
        results["final_metrics"] = await self.base_optimizer._benchmark_model(
            loaded_model, "wav2vec2", num_samples=200
        )
        
        return results
    
    def _generate_latency_recommendations(self, bottlenecks: List[Dict]) -> List[str]:
        """Generate recommendations based on bottlenecks."""
        recommendations = []
        
        for bottleneck in bottlenecks:
            stage = bottleneck["stage"]
            percent = bottleneck["percentage"]
            
            if stage == "preprocessing" and percent > 20:
                recommendations.append("Consider optimizing audio preprocessing pipeline")
            elif stage == "forward_pass" and percent > 20:
                recommendations.append("Consider model quantization or pruning")
            elif stage == "postprocessing" and percent > 20:
                recommendations.append("Optimize post-processing algorithms")
        
        return recommendations


# ==================== PRODUCTION SERVICE ====================

class ProductionOptimizationService:
    """Service for production optimization endpoints."""
    
    def __init__(self):
        self.optimizer = EnhancedProductionOptimizer()
        self.loader = ProductionOptimizedModelLoader("./models/optimized")
    
    async def optimize_model_endpoint(
        self,
        model_path: str,
        optimization_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """API endpoint for model optimization."""
        
        try:
            logger.info(f"Received optimization request: {optimization_type}")
            
            if optimization_type == "comprehensive":
                results = await self.optimizer.comprehensive_optimization(
                    model_path=model_path,
                    output_dir="./models/optimized",
                    enable_distillation=False,
                    enable_pruning=True,
                    enable_batch_opt=True
                )
            elif optimization_type == "fast":
                results = await EnglishModelOptimizer().optimize_for_production(
                    model_path=model_path,
                    output_path="./models/optimized"
                )
            else:
                raise ValueError(f"Unknown optimization type: {optimization_type}")
            
            return {
                "success": True,
                "optimization_type": optimization_type,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_model_metrics(self, model_type: str = "onnx") -> Dict[str, Any]:
        """Get metrics for optimized model."""
        
        try:
            model = await self.loader.load_optimized_model()
            
            if model_type == "onnx":
                metrics = await EnglishModelOptimizer()._benchmark_onnx_session(model)
            else:
                metrics = await EnglishModelOptimizer()._benchmark_model(model, "wav2vec2")
            
            return {
                "success": True,
                "model_type": model_type,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {
                "success": False,
                "error": str(e)
            }


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
