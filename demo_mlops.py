"""
Demo MLOps - Demonstration des fonctionnalites MLOps
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import tempfile
import random

print("=" * 60)
print("        DEMO PIPELINE MLOPS - CIFAR-10")
print("=" * 60)

# 1. Model Registry
print("\n[1] MODEL REGISTRY - Versioning des modeles")
print("-" * 40)
from src.mlops.model_registry import ModelRegistry

registry = ModelRegistry()

# Creer un modele temporaire pour la demo
with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
    model = nn.Linear(10, 5)
    torch.save(model.state_dict(), f.name)
    temp_model = f.name

# Enregistrer le modele
version = registry.register_model(
    model_path=temp_model,
    model_name='demo-resnet18',
    metrics={'accuracy': 0.87, 'f1_score': 0.865},
    tags={'environment': 'demo', 'framework': 'pytorch'},
    description='Demo ResNet18 pour CIFAR-10'
)
print(f"  Modele enregistre: demo-resnet18 v{version}")

# Lister les modeles
models = registry.list_models()
print(f"  Modeles dans le registry: {len(models)}")

# Promouvoir en staging
registry.promote_model('demo-resnet18', version, 'staging')
print(f"  Modele promu en: staging")

os.unlink(temp_model)

# 2. Data Validation
print("\n[2] DATA VALIDATION - Qualite des donnees")
print("-" * 40)
from src.mlops.data_validation import DataValidator

validator = DataValidator(num_classes=10)
print(f"  Classes attendues: {validator.num_classes}")
print(f"  Taille image: {validator.expected_image_size}")

# Valider une image
image = torch.randn(3, 32, 32)
result = validator.validate_single_image(image)
print(f"  Image valide: {result['valid']}")

# 3. Monitoring
print("\n[3] MONITORING - Metriques Prometheus")
print("-" * 40)
from src.mlops.monitoring import MetricsCollector

metrics = MetricsCollector(namespace='cifar10_demo')
metrics.counter_inc('predictions_total', 150)
metrics.counter_inc('predictions_total', 50)
metrics.gauge_set('model_accuracy', 0.87)
metrics.gauge_set('active_models', 2)

for i in range(10):
    metrics.histogram_observe('inference_latency_ms', 45 + i*2)

m = metrics.get_metrics()
print(f"  Predictions totales: {m['counters']['cifar10_demo_predictions_total']}")
print(f"  Accuracy actuelle: {m['gauges']['cifar10_demo_model_accuracy']}")
print(f"  Latence moyenne: {m['histograms']['cifar10_demo_inference_latency_ms']['mean']:.1f} ms")

# 4. Feature Store
print("\n[4] FEATURE STORE - Gestion des features")
print("-" * 40)
from src.mlops.feature_store import FeatureStore

store = FeatureStore()
image = torch.randn(3, 32, 32)
features = store.compute_image_features(image)
print(f"  Features extraites: {len(features)} features")
print(f"  - R_mean: {features['R_mean']:.4f}")
print(f"  - Brightness: {features['brightness']:.4f}")
print(f"  - Contrast: {features['contrast']:.4f}")

# 5. A/B Testing
print("\n[5] A/B TESTING - Tests de modeles")
print("-" * 40)
from src.mlops.ab_testing import ABTestManager

ab = ABTestManager()
exp_id = ab.create_experiment(
    name='resnet_comparison',
    description='Compare ResNet18 vs ResNet34',
    variants=[
        {'name': 'resnet18', 'model': 'models/resnet18.pth'},
        {'name': 'resnet34', 'model': 'models/resnet34.pth'}
    ],
    traffic_weights=[0.5, 0.5]
)
print(f"  Experience creee: {exp_id[:40]}...")

# Simuler des resultats
for _ in range(100):
    ab.record_result(exp_id, 'resnet18', {'accuracy': 0.85 + random.uniform(0, 0.05)})
    ab.record_result(exp_id, 'resnet34', {'accuracy': 0.83 + random.uniform(0, 0.05)})

ab.min_sample_size = 50
analysis = ab.analyze_experiment(exp_id)
print(f"  Echantillons collectes: {sum(v['samples'] for v in analysis['variants'].values())}")
print(f"  ResNet18 accuracy: {analysis['variants']['resnet18']['metrics']['accuracy']['mean']:.4f}")
print(f"  ResNet34 accuracy: {analysis['variants']['resnet34']['metrics']['accuracy']['mean']:.4f}")

# 6. Model Optimization
print("\n[6] MODEL OPTIMIZATION - Benchmark")
print("-" * 40)
from src.mlops.model_optimization import ModelOptimizer

optimizer = ModelOptimizer()
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(16, 10)
)
size = optimizer._get_model_size(model)
print(f"  Taille modele: {size:.4f} MB")

benchmark = optimizer.benchmark_inference(model, num_iterations=20, warmup_iterations=5)
print(f"  Latence moyenne: {benchmark['mean_latency_ms']:.2f} ms")
print(f"  Throughput: {benchmark['throughput_fps']:.1f} FPS")

# 7. Drift Detection
print("\n[7] DRIFT DETECTION - Detection de derive")
print("-" * 40)
from src.mlops.drift_detection import DriftDetector

detector = DriftDetector()
print(f"  Seuil de drift: {detector.drift_threshold}")
print(f"  Seuil de performance: {detector.performance_threshold}")
print(f"  Pret pour la detection de derive")

print("\n" + "=" * 60)
print("        SERVICES DISPONIBLES")
print("=" * 60)
print()
print("  API FastAPI:    http://localhost:8000/docs")
print("  MLflow UI:      http://localhost:5000")
print("  Streamlit:      http://localhost:8501")
print()
print("=" * 60)
