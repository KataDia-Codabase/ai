"""
Production Optimization Module

This module provides comprehensive ML model optimization capabilities for production deployment.
"""

from app.ml.optimization.production_optimizer import (
    # Main classes
    EnglishModelOptimizer,
    ProductionOptimizedModelLoader,
    ProductionOptimizationService,
    EnhancedProductionOptimizer,
    
    # Specialized optimizers
    DistillationOptimizer,
    PruningOptimizer,
    BatchOptimization,
    LatencyOptimization,
    
    # Configuration
    OptimizationConfig,
    ENGLISH_OPTIMIZATION_CONFIGS,
)

__all__ = [
    # Main classes
    "EnglishModelOptimizer",
    "ProductionOptimizedModelLoader",
    "ProductionOptimizationService",
    "EnhancedProductionOptimizer",
    
    # Specialized optimizers
    "DistillationOptimizer",
    "PruningOptimizer",
    "BatchOptimization",
    "LatencyOptimization",
    
    # Configuration
    "OptimizationConfig",
    "ENGLISH_OPTIMIZATION_CONFIGS",
]

__version__ = "1.0.0"
__author__ = "KataDia AI Team"
__description__ = "Production-grade ML model optimization service"
