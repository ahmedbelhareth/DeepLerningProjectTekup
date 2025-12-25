"""
Module MLOps - Pipeline complet de Machine Learning Operations

Ce module contient tous les composants MLOps pour le projet CIFAR-10:
- Model Registry: Versioning et gestion des modèles
- Drift Detection: Détection de dérive des données et du modèle
- Data Validation: Validation de la qualité des données
- Monitoring: Métriques Prometheus et logging structuré
- Feature Store: Gestion des features
- A/B Testing: Infrastructure de tests A/B
- Model Optimization: Export ONNX et optimisation
"""

from .model_registry import ModelRegistry
from .drift_detection import DriftDetector
from .data_validation import DataValidator
from .monitoring import MetricsCollector, StructuredLogger
from .feature_store import FeatureStore
from .ab_testing import ABTestManager
from .model_optimization import ModelOptimizer

__all__ = [
    'ModelRegistry',
    'DriftDetector',
    'DataValidator',
    'MetricsCollector',
    'StructuredLogger',
    'FeatureStore',
    'ABTestManager',
    'ModelOptimizer'
]
