"""
Tests pour les modules MLOps

Ce module contient les tests unitaires pour:
- Model Registry
- Drift Detection
- Data Validation
- Monitoring
- Feature Store
- A/B Testing
- Model Optimization
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import pytest
import numpy as np
import torch
import torch.nn as nn

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# Tests Model Registry
# =============================================================================

class TestModelRegistry:
    """Tests pour le Model Registry."""

    @pytest.fixture
    def temp_registry(self, tmp_path):
        """Crée un registry temporaire."""
        from src.mlops.model_registry import ModelRegistry
        registry_path = tmp_path / "test_registry"
        return ModelRegistry(registry_path=str(registry_path))

    @pytest.fixture
    def sample_model(self, tmp_path):
        """Crée un modèle de test."""
        model = nn.Linear(10, 5)
        model_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), model_path)
        return str(model_path)

    def test_registry_initialization(self, temp_registry):
        """Test l'initialisation du registry."""
        assert temp_registry.registry_path.exists()
        assert temp_registry.models_path.exists()
        assert "models" in temp_registry.registry

    def test_register_model(self, temp_registry, sample_model):
        """Test l'enregistrement d'un modèle."""
        version = temp_registry.register_model(
            model_path=sample_model,
            model_name="test-model",
            metrics={"accuracy": 0.95},
            tags={"framework": "pytorch"},
            description="Test model"
        )

        assert version == "1.0.0"
        assert "test-model" in temp_registry.registry["models"]

    def test_register_multiple_versions(self, temp_registry, sample_model):
        """Test l'enregistrement de plusieurs versions."""
        v1 = temp_registry.register_model(sample_model, "test-model")
        v2 = temp_registry.register_model(sample_model, "test-model")
        v3 = temp_registry.register_model(sample_model, "test-model", bump="minor")

        assert v1 == "1.0.0"
        assert v2 == "1.0.1"
        assert v3 == "1.1.0"

    def test_promote_model(self, temp_registry, sample_model):
        """Test la promotion d'un modèle."""
        version = temp_registry.register_model(sample_model, "test-model")
        temp_registry.promote_model("test-model", version, "production")

        model_info = temp_registry.registry["models"]["test-model"]
        assert model_info["production_version"] == version

    def test_list_models(self, temp_registry, sample_model):
        """Test le listing des modèles."""
        temp_registry.register_model(sample_model, "model-1")
        temp_registry.register_model(sample_model, "model-2")

        models = temp_registry.list_models()
        assert len(models) == 2

    def test_compare_versions(self, temp_registry, sample_model):
        """Test la comparaison de versions."""
        temp_registry.register_model(
            sample_model, "test-model",
            metrics={"accuracy": 0.90}
        )
        temp_registry.register_model(
            sample_model, "test-model",
            metrics={"accuracy": 0.95}
        )

        comparison = temp_registry.compare_versions("test-model", "1.0.0", "1.0.1")
        assert comparison["version1"] == "1.0.0"
        assert comparison["version2"] == "1.0.1"

    def test_export_model_card(self, temp_registry, sample_model):
        """Test l'export de la Model Card."""
        temp_registry.register_model(
            sample_model, "test-model",
            metrics={"accuracy": 0.95},
            description="A test model"
        )

        card = temp_registry.export_model_card("test-model", "1.0.0")
        assert "# Model Card: test-model" in card
        assert "accuracy" in card


# =============================================================================
# Tests Drift Detection
# =============================================================================

class TestDriftDetection:
    """Tests pour le Drift Detection."""

    @pytest.fixture
    def temp_detector(self, tmp_path):
        """Crée un détecteur temporaire."""
        from src.mlops.drift_detection import DriftDetector
        return DriftDetector(baseline_path=str(tmp_path / "baselines"))

    @pytest.fixture
    def mock_dataloader(self):
        """Crée un DataLoader mocké."""
        # Créer des données de test
        data = [
            (torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,)))
            for _ in range(5)
        ]
        return data

    def test_detector_initialization(self, temp_detector):
        """Test l'initialisation du détecteur."""
        assert temp_detector.baseline_path.exists()
        assert temp_detector.drift_threshold == 0.1

    def test_compute_baseline(self, temp_detector, mock_dataloader):
        """Test le calcul de la baseline."""
        # Mock du modèle
        model = nn.Sequential(nn.Flatten(), nn.Linear(3*32*32, 10))
        model.eval()

        baseline = temp_detector.compute_baseline(mock_dataloader, model)

        assert "feature_stats" in baseline
        assert "label_distribution" in baseline
        assert baseline["num_samples"] > 0

    def test_check_data_drift_no_baseline(self, temp_detector, mock_dataloader):
        """Test la détection de drift sans baseline."""
        report = temp_detector.check_data_drift(mock_dataloader)
        assert report["drift_detected"] == False
        assert "No baseline available" in report["message"]

    def test_calculate_psi(self, temp_detector):
        """Test le calcul du PSI."""
        baseline_dist = {"0": 0.5, "1": 0.5}
        current_dist = {"0": 0.6, "1": 0.4}

        psi = temp_detector._calculate_psi(baseline_dist, current_dist)
        assert isinstance(psi, float)
        assert psi >= 0


# =============================================================================
# Tests Data Validation
# =============================================================================

class TestDataValidation:
    """Tests pour la Data Validation."""

    @pytest.fixture
    def validator(self):
        """Crée un validateur."""
        from src.mlops.data_validation import DataValidator
        return DataValidator()

    @pytest.fixture
    def mock_dataloaders(self):
        """Crée des DataLoaders mockés."""
        train = [(torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))) for _ in range(5)]
        val = [(torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))) for _ in range(2)]
        test = [(torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))) for _ in range(2)]
        return train, val, test

    def test_validator_initialization(self, validator):
        """Test l'initialisation du validateur."""
        assert validator.num_classes == 10
        assert validator.expected_image_size == (3, 32, 32)

    def test_validate_dataset(self, validator, mock_dataloaders):
        """Test la validation du dataset."""
        train, val, test = mock_dataloaders
        report = validator.validate_dataset(train, val, test)

        assert "train_samples" in report
        assert "data_integrity_ok" in report
        assert "class_balance_ok" in report

    def test_validate_single_image_valid(self, validator):
        """Test la validation d'une image valide."""
        image = torch.randn(3, 32, 32)
        result = validator.validate_single_image(image)
        assert result["valid"] == True

    def test_validate_single_image_with_nan(self, validator):
        """Test la validation d'une image avec NaN."""
        image = torch.randn(3, 32, 32)
        image[0, 0, 0] = float('nan')
        result = validator.validate_single_image(image)
        assert result["valid"] == False

    def test_generate_validation_report(self, validator):
        """Test la génération du rapport."""
        validation_result = {
            "timestamp": datetime.now().isoformat(),
            "train_samples": 1000,
            "val_samples": 200,
            "test_samples": 200,
            "data_integrity_ok": True,
            "class_balance_ok": True,
            "image_quality_ok": True,
            "validation_passed": True,
            "issues": []
        }

        report = validator.generate_validation_report(validation_result)
        assert "Data Validation Report" in report


# =============================================================================
# Tests Monitoring
# =============================================================================

class TestMonitoring:
    """Tests pour le Monitoring."""

    @pytest.fixture
    def metrics_collector(self):
        """Crée un collecteur de métriques."""
        from src.mlops.monitoring import MetricsCollector
        return MetricsCollector(namespace="test")

    @pytest.fixture
    def structured_logger(self, tmp_path):
        """Crée un logger structuré."""
        from src.mlops.monitoring import StructuredLogger
        return StructuredLogger(
            name="test",
            log_dir=str(tmp_path / "logs"),
            console_output=False
        )

    def test_counter_increment(self, metrics_collector):
        """Test l'incrémentation d'un compteur."""
        metrics_collector.counter_inc("requests_total")
        metrics_collector.counter_inc("requests_total", 5)

        metrics = metrics_collector.get_metrics()
        assert metrics["counters"]["test_requests_total"] == 6

    def test_gauge_operations(self, metrics_collector):
        """Test les opérations sur une gauge."""
        metrics_collector.gauge_set("active_connections", 10)
        metrics_collector.gauge_inc("active_connections", 5)
        metrics_collector.gauge_dec("active_connections", 3)

        metrics = metrics_collector.get_metrics()
        assert metrics["gauges"]["test_active_connections"] == 12

    def test_histogram_observe(self, metrics_collector):
        """Test l'observation d'un histogramme."""
        for i in range(10):
            metrics_collector.histogram_observe("latency", i * 0.1)

        metrics = metrics_collector.get_metrics()
        assert "test_latency" in metrics["histograms"]
        assert metrics["histograms"]["test_latency"]["count"] == 10

    def test_prometheus_export(self, metrics_collector):
        """Test l'export au format Prometheus."""
        metrics_collector.counter_inc("requests")
        metrics_collector.gauge_set("memory", 100)

        output = metrics_collector.export_prometheus_format()
        assert "test_requests" in output
        assert "test_memory" in output

    def test_structured_logger_info(self, structured_logger):
        """Test le logging structuré."""
        structured_logger.info("Test message", key="value")
        # Le test vérifie que le logging ne génère pas d'erreur

    def test_log_prediction(self, structured_logger):
        """Test le logging d'une prédiction."""
        structured_logger.log_prediction(
            prediction="cat",
            confidence=0.95,
            latency_ms=45.2
        )


# =============================================================================
# Tests Feature Store
# =============================================================================

class TestFeatureStore:
    """Tests pour le Feature Store."""

    @pytest.fixture
    def temp_store(self, tmp_path):
        """Crée un Feature Store temporaire."""
        from src.mlops.feature_store import FeatureStore
        return FeatureStore(store_path=str(tmp_path / "feature_store"))

    def test_store_initialization(self, temp_store):
        """Test l'initialisation du store."""
        assert temp_store.store_path.exists()
        assert temp_store.features_path.exists()

    def test_register_feature_group(self, temp_store):
        """Test l'enregistrement d'un groupe de features."""
        group_id = temp_store.register_feature_group(
            name="test_features",
            description="Test feature group",
            features=[
                {"name": "feature1", "type": "float"},
                {"name": "feature2", "type": "float"}
            ]
        )

        assert "test_features" in group_id
        assert "test_features" in temp_store.metadata["feature_groups"]

    def test_store_and_get_features(self, temp_store):
        """Test le stockage et la récupération des features."""
        features = {"feature1": 0.5, "feature2": 1.0}

        storage_id = temp_store.store_features(
            feature_group="test",
            entity_id="entity_1",
            features=features
        )

        retrieved = temp_store.get_features("test", "entity_1")
        assert retrieved["feature1"] == 0.5
        assert retrieved["feature2"] == 1.0

    def test_compute_image_features(self, temp_store):
        """Test le calcul des features d'une image."""
        image = torch.randn(3, 32, 32)
        features = temp_store.compute_image_features(image)

        assert "R_mean" in features
        assert "global_std" in features
        assert "brightness" in features

    def test_cache_operations(self, temp_store):
        """Test les opérations de cache."""
        temp_store.store_features("test", "entity_1", {"f1": 1.0})
        stats = temp_store.get_cache_stats()

        assert stats["cache_size"] >= 1
        temp_store.clear_cache()
        assert temp_store.get_cache_stats()["cache_size"] == 0


# =============================================================================
# Tests A/B Testing
# =============================================================================

class TestABTesting:
    """Tests pour l'A/B Testing."""

    @pytest.fixture
    def temp_ab_manager(self, tmp_path):
        """Crée un gestionnaire A/B temporaire."""
        from src.mlops.ab_testing import ABTestManager
        return ABTestManager(experiment_path=str(tmp_path / "experiments"))

    def test_create_experiment(self, temp_ab_manager):
        """Test la création d'une expérience."""
        exp_id = temp_ab_manager.create_experiment(
            name="test_experiment",
            description="Test A/B experiment",
            variants=[
                {"name": "control", "model_path": "model_a.pth"},
                {"name": "treatment", "model_path": "model_b.pth"}
            ]
        )

        assert "test_experiment" in exp_id
        assert exp_id in temp_ab_manager.experiments

    def test_get_variant(self, temp_ab_manager):
        """Test la sélection de variante."""
        exp_id = temp_ab_manager.create_experiment(
            name="test",
            description="Test",
            variants=[
                {"name": "A"},
                {"name": "B"}
            ]
        )

        variant = temp_ab_manager.get_variant(exp_id)
        assert variant["name"] in ["A", "B"]

    def test_record_result(self, temp_ab_manager):
        """Test l'enregistrement des résultats."""
        exp_id = temp_ab_manager.create_experiment(
            name="test",
            description="Test",
            variants=[{"name": "A"}, {"name": "B"}]
        )

        temp_ab_manager.record_result(exp_id, "A", {"accuracy": 0.9})
        temp_ab_manager.record_result(exp_id, "A", {"accuracy": 0.92})

        results = temp_ab_manager.experiments[exp_id]["results"]["A"]
        assert results["samples"] == 2

    def test_analyze_experiment(self, temp_ab_manager):
        """Test l'analyse d'une expérience."""
        exp_id = temp_ab_manager.create_experiment(
            name="test",
            description="Test",
            variants=[{"name": "A"}, {"name": "B"}]
        )

        for _ in range(50):
            temp_ab_manager.record_result(exp_id, "A", {"accuracy": 0.9})
            temp_ab_manager.record_result(exp_id, "B", {"accuracy": 0.85})

        # Réduire le min_sample_size pour le test
        temp_ab_manager.min_sample_size = 10

        analysis = temp_ab_manager.analyze_experiment(exp_id)
        assert "variants" in analysis
        assert "statistical_tests" in analysis

    def test_conclude_experiment(self, temp_ab_manager):
        """Test la conclusion d'une expérience."""
        exp_id = temp_ab_manager.create_experiment(
            name="test",
            description="Test",
            variants=[{"name": "A"}, {"name": "B"}]
        )

        result = temp_ab_manager.conclude_experiment(exp_id, winner="A")
        assert result["winner"] == "A"
        assert temp_ab_manager.experiments[exp_id]["status"] == "completed"


# =============================================================================
# Tests Model Optimization
# =============================================================================

class TestModelOptimization:
    """Tests pour l'optimisation des modèles."""

    @pytest.fixture
    def temp_optimizer(self, tmp_path):
        """Crée un optimiseur temporaire."""
        from src.mlops.model_optimization import ModelOptimizer
        return ModelOptimizer(output_dir=str(tmp_path / "optimized"))

    @pytest.fixture
    def simple_model(self):
        """Crée un modèle simple pour les tests."""
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10)
        )

    def test_optimizer_initialization(self, temp_optimizer):
        """Test l'initialisation de l'optimiseur."""
        assert temp_optimizer.output_dir.exists()

    def test_get_model_size(self, temp_optimizer, simple_model):
        """Test le calcul de la taille du modèle."""
        size = temp_optimizer._get_model_size(simple_model)
        assert size > 0
        assert isinstance(size, float)

    def test_benchmark_inference(self, temp_optimizer, simple_model):
        """Test le benchmark d'inférence."""
        benchmark = temp_optimizer.benchmark_inference(
            simple_model,
            input_shape=(1, 3, 32, 32),
            num_iterations=10,
            warmup_iterations=2
        )

        assert "mean_latency_ms" in benchmark
        assert "p50_latency_ms" in benchmark
        assert "throughput_fps" in benchmark

    def test_benchmark_batch_sizes(self, temp_optimizer, simple_model):
        """Test le benchmark de différentes tailles de batch."""
        results = temp_optimizer.benchmark_batch_sizes(
            simple_model,
            batch_sizes=[1, 2, 4],
            num_iterations=5
        )

        assert "batch_benchmarks" in results
        assert "optimal_batch_size" in results

    def test_quantize_dynamic(self, temp_optimizer, simple_model):
        """Test la quantification dynamique."""
        simple_model.eval()
        quantized, report = temp_optimizer.quantize_dynamic(simple_model)

        assert "original_size_mb" in report
        assert "quantized_size_mb" in report
        assert "size_reduction" in report

    @pytest.mark.skipif(
        not hasattr(torch, 'onnx'),
        reason="ONNX export not available"
    )
    def test_export_onnx(self, temp_optimizer, simple_model, tmp_path):
        """Test l'export ONNX."""
        output_path = tmp_path / "model.onnx"
        result = temp_optimizer.export_onnx(
            simple_model,
            input_shape=(1, 3, 32, 32),
            output_path=str(output_path)
        )

        assert Path(result).exists()


# =============================================================================
# Tests d'intégration
# =============================================================================

class TestMLOpsIntegration:
    """Tests d'intégration pour les modules MLOps."""

    def test_full_pipeline_simulation(self, tmp_path):
        """Simule un pipeline MLOps complet."""
        from src.mlops.model_registry import ModelRegistry
        from src.mlops.monitoring import MetricsCollector, StructuredLogger

        # 1. Initialiser les composants
        registry = ModelRegistry(str(tmp_path / "registry"))
        metrics = MetricsCollector("integration_test")
        logger = StructuredLogger("integration", str(tmp_path / "logs"), console_output=False)

        # 2. Créer et enregistrer un modèle
        model = nn.Linear(10, 5)
        model_path = tmp_path / "model.pth"
        torch.save(model.state_dict(), model_path)

        version = registry.register_model(
            str(model_path),
            "integration-test-model",
            metrics={"accuracy": 0.95}
        )

        # 3. Logger l'événement
        logger.log_model_loaded(str(model_path), "integration-test-model")

        # 4. Enregistrer des métriques
        metrics.counter_inc("models_registered")
        metrics.gauge_set("current_accuracy", 0.95)

        # 5. Promouvoir le modèle
        registry.promote_model("integration-test-model", version, "production")

        # Vérifications
        assert version == "1.0.0"
        loaded = registry.load_model("integration-test-model", stage="production")
        assert loaded["version"] == version


# =============================================================================
# Point d'entrée
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
