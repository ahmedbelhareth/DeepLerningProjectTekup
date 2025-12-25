# Guide MLOps - Projet CIFAR-10

## Vue d'ensemble

Ce projet implémente un pipeline MLOps complet pour la classification d'images CIFAR-10 avec ResNet18. Ce guide détaille tous les composants MLOps disponibles.

## Architecture MLOps

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Pipeline MLOps Complet                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │   GitHub     │───▶│     CI/CD    │───▶│    Docker    │───▶│ Production │ │
│  │   Actions    │    │   Pipeline   │    │    Build     │    │   Deploy   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│         │                   │                   │                   │        │
│         ▼                   ▼                   ▼                   ▼        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │  Pre-commit  │    │    Tests     │    │   Registry   │    │ Monitoring │ │
│  │    Hooks     │    │   Pytest     │    │    GHCR      │    │ Prometheus │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Composants ML                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │    Data      │───▶│   Feature    │───▶│   Model      │───▶│    Model   │ │
│  │  Validation  │    │    Store     │    │  Training    │    │  Registry  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│         │                   │                   │                   │        │
│         ▼                   ▼                   ▼                   ▼        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │    Drift     │    │   MLflow     │    │    ONNX      │    │  A/B Test  │ │
│  │  Detection   │    │  Tracking    │    │   Export     │    │   Manager  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Composants MLOps

### 1. CI/CD Pipeline (GitHub Actions)

**Fichiers:** `.github/workflows/`

#### Pipeline Principal (`ci-cd.yml`)
- **Code Quality:** Black, isort, Flake8, Bandit
- **Unit Tests:** pytest avec coverage
- **Integration Tests:** Tests API
- **Model Validation:** Vérification architecture
- **Docker Build:** Images API et Streamlit
- **Deployment:** Staging et Production

```bash
# Déclencher manuellement le training
gh workflow run model-training.yml -f model_name=resnet18 -f epochs=50
```

#### Pipeline Training (`model-training.yml`)
- Training automatique ou planifié
- Validation des données
- Enregistrement du modèle
- Export des artifacts

#### Drift Detection (`drift-detection.yml`)
- Exécution quotidienne à 6h
- Détection de data drift
- Détection de model drift
- Déclenchement automatique du retraining

### 2. Model Registry

**Module:** `src/mlops/model_registry.py`

```python
from src.mlops import ModelRegistry

registry = ModelRegistry()

# Enregistrer un modèle
version = registry.register_model(
    model_path="models/best_model.pth",
    model_name="cifar10-resnet18",
    metrics={"accuracy": 0.87, "f1_score": 0.87},
    tags={"framework": "pytorch"},
    description="ResNet18 trained on CIFAR-10"
)

# Promouvoir en production
registry.promote_model("cifar10-resnet18", version, "production")

# Charger le modèle de production
model_data = registry.load_model("cifar10-resnet18", stage="production")

# Comparer deux versions
comparison = registry.compare_versions("cifar10-resnet18", "1.0.0", "1.1.0")
```

### 3. Drift Detection

**Module:** `src/mlops/drift_detection.py`

```python
from src.mlops import DriftDetector

detector = DriftDetector()

# Calculer la baseline
detector.compute_baseline(train_loader, model)

# Vérifier la dérive des données
data_report = detector.check_data_drift(new_data_loader)
print(f"Drift détecté: {data_report['drift_detected']}")

# Vérifier la dégradation du modèle
model_report = detector.check_model_drift(model, test_loader)
print(f"Performance dégradée: {model_report['degraded']}")

# Générer un rapport complet
report = detector.generate_drift_report(data_report, model_report)
```

### 4. Data Validation

**Module:** `src/mlops/data_validation.py`

```python
from src.mlops import DataValidator

validator = DataValidator()

# Valider le dataset complet
report = validator.validate_dataset(train_loader, val_loader, test_loader)

print(f"Validation passée: {report['validation_passed']}")
print(f"Intégrité: {report['data_integrity_ok']}")
print(f"Équilibre classes: {report['class_balance_ok']}")

# Vérifier les fuites de données
leakage = validator.check_data_leakage(train_loader, test_loader)
```

### 5. Monitoring & Logging

**Module:** `src/mlops/monitoring.py`

```python
from src.mlops import MetricsCollector, StructuredLogger

# Métriques Prometheus
metrics = MetricsCollector(namespace="cifar10")
metrics.counter_inc("predictions_total")
metrics.histogram_observe("inference_latency", 0.05)
metrics.gauge_set("model_accuracy", 0.87)

# Export au format Prometheus
prometheus_output = metrics.export_prometheus_format()

# Logging structuré JSON
logger = StructuredLogger("api")
logger.log_prediction(
    prediction="cat",
    confidence=0.95,
    latency_ms=45.2
)
```

### 6. Feature Store

**Module:** `src/mlops/feature_store.py`

```python
from src.mlops import FeatureStore

store = FeatureStore()

# Enregistrer un groupe de features
store.register_feature_group(
    name="image_features",
    description="Features extraites des images",
    features=[
        {"name": "brightness", "type": "float"},
        {"name": "contrast", "type": "float"}
    ]
)

# Calculer et stocker des features
features = store.compute_image_features(image_tensor)
store.store_features("image_features", "image_001", features)

# Récupérer des features
stored_features = store.get_features("image_features", "image_001")
```

### 7. A/B Testing

**Module:** `src/mlops/ab_testing.py`

```python
from src.mlops import ABTestManager

ab_manager = ABTestManager()

# Créer une expérience
exp_id = ab_manager.create_experiment(
    name="model_comparison",
    description="Compare ResNet18 vs ResNet34",
    variants=[
        {"name": "control", "model_path": "models/resnet18.pth"},
        {"name": "treatment", "model_path": "models/resnet34.pth"}
    ],
    traffic_weights=[0.5, 0.5]
)

# Router le trafic
variant = ab_manager.get_variant(exp_id, user_id="user_123")

# Enregistrer les résultats
ab_manager.record_result(exp_id, variant["name"], {
    "accuracy": 0.89,
    "latency_ms": 45
})

# Analyser l'expérience
analysis = ab_manager.analyze_experiment(exp_id)
```

### 8. Model Optimization

**Module:** `src/mlops/model_optimization.py`

```python
from src.mlops import ModelOptimizer

optimizer = ModelOptimizer()

# Export ONNX
onnx_path = optimizer.export_onnx(model, output_path="model.onnx")

# Quantification dynamique
quantized_model, report = optimizer.quantize_dynamic(model)
print(f"Réduction taille: {report['size_reduction']}")

# Benchmark
benchmark = optimizer.benchmark_inference(model)
print(f"Latence moyenne: {benchmark['mean_latency_ms']:.2f} ms")

# Optimisation TorchScript
optimized_model, opt_report = optimizer.optimize_for_inference(model)
print(f"Speedup: {opt_report['speedup']}")
```

### 9. Pipeline de Retraining

**Module:** `src/mlops/retraining.py`

```python
from src.mlops.retraining import RetrainingPipeline

pipeline = RetrainingPipeline()

# Vérifier si retraining nécessaire
check = pipeline.check_retraining_needed(data_loader, model)
if check["needs_retraining"]:
    print(f"Raisons: {check['reasons']}")

# Lancer le retraining
result = pipeline.run_retraining(train_loader, val_loader, test_loader)
print(f"Nouvelle version: {result['new_version']}")
```

## DVC Pipeline

**Fichier:** `dvc.yaml`

```bash
# Initialiser DVC
dvc init

# Exécuter le pipeline complet
dvc repro

# Exécuter une étape spécifique
dvc repro train

# Visualiser le DAG
dvc dag
```

Étapes du pipeline:
1. `prepare_data` - Préparation des données
2. `validate_data` - Validation qualité
3. `extract_features` - Extraction features
4. `train` - Entraînement
5. `evaluate` - Évaluation
6. `register` - Enregistrement
7. `export_onnx` - Export ONNX

## Pre-commit Hooks

**Fichier:** `.pre-commit-config.yaml`

```bash
# Installer les hooks
pip install pre-commit
pre-commit install

# Exécuter manuellement
pre-commit run --all-files
```

Hooks configurés:
- **Black:** Formatage du code
- **isort:** Tri des imports
- **Flake8:** Linting
- **Bandit:** Sécurité
- **MyPy:** Type checking
- **nbstripout:** Nettoyage notebooks

## Configuration

### Variables d'environnement

```bash
# MLflow
export MLFLOW_TRACKING_URI=./mlruns
export MLFLOW_EXPERIMENT_NAME=CIFAR10_Classification

# API
export API_HOST=0.0.0.0
export API_PORT=8000

# Monitoring
export PROMETHEUS_PORT=9090
export LOG_LEVEL=INFO
```

### Fichiers de configuration

- `pyproject.toml` - Configuration projet Python
- `.pre-commit-config.yaml` - Pre-commit hooks
- `dvc.yaml` - Pipeline DVC
- `retraining_config.json` - Configuration retraining

## Commandes utiles

```bash
# Démarrer l'API
uvicorn src.api.main:app --reload --port 8000

# Démarrer Streamlit
streamlit run streamlit_app/app.py

# Lancer les tests
pytest tests/ -v --cov=src

# Lancer MLflow UI
mlflow ui --port 5000

# Docker Compose
docker-compose -f docker/docker-compose.yml up -d

# Exécuter le pipeline DVC
dvc repro

# Vérifier la qualité du code
pre-commit run --all-files
```

## Métriques de monitoring

| Métrique | Description | Type |
|----------|-------------|------|
| `cifar10_predictions_total` | Nombre total de prédictions | Counter |
| `cifar10_inference_latency` | Latence d'inférence | Histogram |
| `cifar10_model_accuracy` | Accuracy du modèle | Gauge |
| `cifar10_data_drift_score` | Score de dérive données | Gauge |
| `cifar10_api_requests_total` | Requêtes API totales | Counter |

## Bonnes pratiques

1. **Versioning:** Toujours versionner les modèles avec le Model Registry
2. **Monitoring:** Surveiller les métriques de drift quotidiennement
3. **Tests:** Exécuter les tests avant chaque push
4. **Documentation:** Mettre à jour la documentation avec les changements
5. **Review:** Utiliser les pull requests pour les changements majeurs

## Troubleshooting

### Problèmes courants

**Erreur de drift detection:**
```python
# Recalculer la baseline
detector.compute_baseline(train_loader, model)
```

**Modèle non trouvé dans le registry:**
```python
# Lister les modèles disponibles
models = registry.list_models()
```

**Échec du pipeline CI/CD:**
```bash
# Vérifier les logs
gh run view <run-id> --log
```

## Ressources

- [Documentation PyTorch](https://pytorch.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [DVC Documentation](https://dvc.org/doc)
- [GitHub Actions](https://docs.github.com/en/actions)
