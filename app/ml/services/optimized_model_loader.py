"""
Optimized Model Loader Service
Automatically loads optimized models if available, falls back to original model otherwise.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import structlog
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logger = structlog.get_logger()

class OptimizedModelLoader:
    """
    Intelligently loads either optimized or original models.
    Priority: Quantized INT8 > ONNX > TorchScript > Pruned > Original
    """
    
    def __init__(self, base_model_path: str = "./models/wav2vec2-english-finetuned"):
        self.base_model_path = Path(base_model_path)
        self.optimization_dir = self.base_model_path.parent / "optimized"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimization_metadata: Dict[str, Any] = {}
        self._load_optimization_metadata()
    
    def _load_optimization_metadata(self):
        """Load optimization metadata if available."""
        try:
            metadata_path = self.optimization_dir / "optimization_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.optimization_metadata = json.load(f)
                    logger.info(
                        "Loaded optimization metadata",
                        optimizations_available=list(self.optimization_metadata.get('models', {}).keys())
                    )
        except Exception as e:
            logger.warning("Failed to load optimization metadata", error=str(e))
            self.optimization_metadata = {}
    
    async def load_model_with_processor(
        self,
        language: str = "en-US"
    ) -> Tuple[Optional[Wav2Vec2ForCTC], Optional[Wav2Vec2Processor], Dict[str, Any]]:
        """
        Load model and processor with automatic optimization detection.
        
        Args:
            language: Language code (en-US for English)
            
        Returns:
            Tuple of (model, processor, metadata)
            Metadata includes:
                - model_type: Type of model loaded (quantized/onnx/torchscript/original)
                - optimization_level: Compression/speed optimization level
                - performance_metrics: Available performance metrics from optimization
                - device: Device model is on
        """
        try:
            logger.info("Loading model with optimization detection", language=language)
            
            # Try to load optimized model
            model_info = await self._try_load_optimized_model(language)
            
            if model_info['model'] is not None:
                # Successfully loaded optimized model
                logger.info(
                    "Optimized model loaded successfully",
                    model_type=model_info['type'],
                    optimization_level=model_info.get('optimization_level', 'unknown')
                )
                
                # Load processor (always use original processor)
                processor = await self._load_processor(language)
                return model_info['model'], processor, model_info['metadata']
            
            # Fallback to original model
            logger.info("Loading original model (no optimized version available)")
            model, processor = await self._load_original_model(language)
            
            metadata = {
                'model_type': 'original',
                'optimization_level': 0,
                'device': str(self.device),
                'performance_metrics': {}
            }
            
            return model, processor, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}", language=language)
            return None, None, {}
    
    async def _try_load_optimized_model(self, language: str) -> Dict[str, Any]:
        """Try to load optimized model in priority order."""
        
        result = {
            'model': None,
            'type': None,
            'metadata': {},
            'optimization_level': 0
        }
        
        quantized_model = await self._try_load_quantized_model()
        if quantized_model is not None:
            result['model'] = quantized_model
            result['type'] = 'quantized_int8'
            result['optimization_level'] = 2
            result['metadata'] = {
                'model_type': 'quantized_int8',
                'optimization_level': 2,
                'device': str(self.device),
                'performance_metrics': self.optimization_metadata.get('models', {}).get('quantized', {})
            }
            return result
        
        onnx_model = await self._try_load_onnx_model()
        if onnx_model is not None:
            result['model'] = onnx_model
            result['type'] = 'onnx'
            result['optimization_level'] = 2
            result['metadata'] = {
                'model_type': 'onnx',
                'optimization_level': 2,
                'device': str(self.device),
                'performance_metrics': self.optimization_metadata.get('models', {}).get('onnx', {})
            }
            return result
        
        torchscript_model = await self._try_load_torchscript_model()
        if torchscript_model is not None:
            result['model'] = torchscript_model
            result['type'] = 'torchscript'
            result['optimization_level'] = 1
            result['metadata'] = {
                'model_type': 'torchscript',
                'optimization_level': 1,
                'device': str(self.device),
                'performance_metrics': self.optimization_metadata.get('models', {}).get('torchscript', {})
            }
            return result
        
        pruned_model = await self._try_load_pruned_model()
        if pruned_model is not None:
            result['model'] = pruned_model
            result['type'] = 'pruned'
            result['optimization_level'] = 1
            result['metadata'] = {
                'model_type': 'pruned',
                'optimization_level': 1,
                'device': str(self.device),
                'performance_metrics': self.optimization_metadata.get('models', {}).get('pruned', {})
            }
            return result
        
        return result
    
    async def _try_load_quantized_model(self) -> Optional[torch.nn.Module]:
        """Try to load INT8 quantized model."""
        try:
            quantized_path = self.optimization_dir / "quantized_model.pt"
            if not quantized_path.exists():
                logger.debug("Quantized model not found", path=str(quantized_path))
                return None
            
            logger.info("Loading quantized INT8 model", path=str(quantized_path))
            
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None,
                lambda: torch.load(str(quantized_path), map_location=self.device)
            )
            
            model.to(self.device)
            model.eval()
            
            logger.info("Quantized model loaded successfully")
            return model
            
        except Exception as e:
            logger.warning("Failed to load quantized model", error=str(e))
            return None
    
    async def _try_load_onnx_model(self) -> Optional[Any]:
        """Try to load ONNX optimized model."""
        try:
            onnx_path = self.optimization_dir / "model_optimized.onnx"
            if not onnx_path.exists():
                logger.debug("ONNX model not found", path=str(onnx_path))
                return None
            
            logger.info("Loading ONNX optimized model", path=str(onnx_path))
            
            import onnxruntime as ort
            
            loop = asyncio.get_event_loop()
            session = await loop.run_in_executor(
                None,
                lambda: ort.InferenceSession(
                    str(onnx_path),
                    providers=['CPUExecutionProvider']
                )
            )
            
            logger.info("ONNX model loaded successfully")
            return session
            
        except Exception as e:
            logger.warning("Failed to load ONNX model", error=str(e))
            return None
    
    async def _try_load_torchscript_model(self) -> Optional[torch.jit.ScriptModule]:
        """Try to load TorchScript compiled model."""
        try:
            ts_path = self.optimization_dir / "model_optimized.ts"
            if not ts_path.exists():
                logger.debug("TorchScript model not found", path=str(ts_path))
                return None
            
            logger.info("Loading TorchScript compiled model", path=str(ts_path))
            
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None,
                lambda: torch.jit.load(str(ts_path), map_location=self.device)
            )
            
            model.to(self.device)
            model.eval()
            
            logger.info("TorchScript model loaded successfully")
            return model
            
        except Exception as e:
            logger.warning("Failed to load TorchScript model", error=str(e))
            return None
    
    async def _try_load_pruned_model(self) -> Optional[Wav2Vec2ForCTC]:
        """Try to load pruned model."""
        try:
            pruned_path = self.optimization_dir / "pruned_model.pt"
            if not pruned_path.exists():
                logger.debug("Pruned model not found", path=str(pruned_path))
                return None
            
            logger.info("Loading pruned model", path=str(pruned_path))
            
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None,
                lambda: torch.load(str(pruned_path), map_location=self.device)
            )
            
            model.to(self.device)
            model.eval()
            
            logger.info("Pruned model loaded successfully")
            return model
            
        except Exception as e:
            logger.warning("Failed to load pruned model", error=str(e))
            return None
    
    async def _load_processor(self, language: str) -> Optional[Wav2Vec2Processor]:
        """Load processor from original model path."""
        try:
            logger.info("Loading processor", language=language)
            
            processor = Wav2Vec2Processor.from_pretrained(
                str(self.base_model_path)
            )
            
            logger.info("Processor loaded successfully")
            return processor
            
        except Exception as e:
            logger.error("Failed to load processor", error=str(e))
            return None
    
    async def _load_original_model(self, language: str) -> Tuple[Optional[Wav2Vec2ForCTC], Optional[Wav2Vec2Processor]]:
        """Load original model from HuggingFace or local path."""
        try:
            logger.info("Loading original model from HuggingFace", path=str(self.base_model_path))
            
            loop = asyncio.get_event_loop()
            
            model = await loop.run_in_executor(
                None,
                lambda: Wav2Vec2ForCTC.from_pretrained(str(self.base_model_path))
            )
            
            processor = await self._load_processor(language)
            
            model.to(self.device)
            model.eval()
            
            logger.info("Original model loaded successfully")
            return model, processor
            
        except Exception as e:
            logger.error("Failed to load original model", error=str(e))
            return None, None
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """Get information about available optimizations."""
        return {
            'base_model_path': str(self.base_model_path),
            'optimization_dir': str(self.optimization_dir),
            'device': str(self.device),
            'metadata': self.optimization_metadata
        }


class ModelCache:
    """Cache for loaded models to avoid reloading."""
    
    def __init__(self):
        self._cache: Dict[str, Tuple[Any, Any, Dict]] = {}
        self._lock = asyncio.Lock()
    
    async def get_or_load(
        self,
        language: str,
        loader: OptimizedModelLoader
    ) -> Tuple[Optional[Any], Optional[Any], Dict]:
        """Get model from cache or load if not cached."""
        
        async with self._lock:
            if language in self._cache:
                logger.debug("Loading model from cache", language=language)
                return self._cache[language]
            
            logger.info("Model not in cache, loading...", language=language)
            model, processor, metadata = await loader.load_model_with_processor(language)
            
            if model is not None:
                self._cache[language] = (model, processor, metadata)
            
            return model, processor, metadata
    
    def clear_cache(self, language: Optional[str] = None):
        """Clear cache for specific language or all."""
        if language:
            self._cache.pop(language, None)
            logger.info("Cache cleared for language", language=language)
        else:
            self._cache.clear()
            logger.info("All cache cleared")


# Global model cache
_model_cache = ModelCache()
_model_loader = None


def initialize_model_loader(base_model_path: str = "./models/wav2vec2-english-finetuned"):
    """Initialize global model loader."""
    global _model_loader
    _model_loader = OptimizedModelLoader(base_model_path)
    logger.info("Model loader initialized", base_path=base_model_path)


async def get_optimized_model_and_processor(
    language: str = "en-US"
) -> Tuple[Optional[Any], Optional[Any], Dict]:
    """
    Get model and processor with optimization detection.
    
    This is the main function to use for loading models.
    Automatically detects and loads optimized versions if available.
    """
    
    global _model_loader
    
    if _model_loader is None:
        initialize_model_loader()
    
    return await _model_cache.get_or_load(language, _model_loader)


def get_loader_info() -> Dict[str, Any]:
    """Get information about current model loader."""
    global _model_loader
    if _model_loader is None:
        return {'status': 'not_initialized'}
    return _model_loader.get_optimization_info()
