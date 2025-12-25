"""
Retraining Pipeline - Pipeline de retraining automatique

Ce module fournit:
- Orchestration du retraining
- Triggers basés sur les métriques
- Validation du nouveau modèle
- Promotion automatique
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import torch

from .drift_detection import DriftDetector
from .data_validation import DataValidator
from .model_registry import ModelRegistry
from .monitoring import StructuredLogger, MetricsCollector


class RetrainingPipeline:
    """
    Pipeline de retraining automatique.

    Fonctionnalités:
    - Détection automatique du besoin de retraining
    - Exécution du pipeline d'entraînement
    - Validation du nouveau modèle
    - Promotion conditionnelle
    """

    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        drift_detector: Optional[DriftDetector] = None,
        data_validator: Optional[DataValidator] = None,
        logger: Optional[StructuredLogger] = None,
        config_path: str = "retraining_config.json"
    ):
        """
        Initialise le pipeline de retraining.

        Args:
            model_registry: Registre de modèles
            drift_detector: Détecteur de dérive
            data_validator: Validateur de données
            logger: Logger structuré
            config_path: Chemin de configuration
        """
        self.model_registry = model_registry or ModelRegistry()
        self.drift_detector = drift_detector or DriftDetector()
        self.data_validator = data_validator or DataValidator()
        self.logger = logger or StructuredLogger("retraining")
        self.metrics = MetricsCollector("retraining")

        self.config_path = Path(config_path)
        self._load_config()

    def _load_config(self):
        """Charge la configuration du pipeline."""
        default_config = {
            "min_accuracy_improvement": 0.005,
            "max_performance_drop": 0.02,
            "retraining_cooldown_hours": 24,
            "auto_promote": False,
            "notification_channels": ["log"],
            "training_config": {
                "epochs": 50,
                "learning_rate": 0.001,
                "batch_size": 128,
                "model_name": "resnet18",
                "pretrained": True
            }
        }

        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = {**default_config, **json.load(f)}
        else:
            self.config = default_config
            self._save_config()

    def _save_config(self):
        """Sauvegarde la configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def check_retraining_needed(
        self,
        data_loader,
        model: torch.nn.Module,
        baseline_accuracy: float = 0.87
    ) -> Dict[str, Any]:
        """
        Vérifie si un retraining est nécessaire.

        Args:
            data_loader: DataLoader pour l'évaluation
            model: Modèle actuel
            baseline_accuracy: Accuracy de référence

        Returns:
            Rapport avec décision de retraining
        """
        self.logger.info("Checking if retraining is needed")

        # Vérifier la dérive des données
        data_drift_report = self.drift_detector.check_data_drift(data_loader)

        # Vérifier la dégradation du modèle
        model_drift_report = self.drift_detector.check_model_drift(
            model, data_loader, baseline_accuracy
        )

        # Décision
        needs_retraining = (
            data_drift_report["drift_detected"] or
            model_drift_report["degraded"]
        )

        report = {
            "timestamp": datetime.now().isoformat(),
            "needs_retraining": needs_retraining,
            "data_drift": data_drift_report,
            "model_drift": model_drift_report,
            "reasons": []
        }

        if data_drift_report["drift_detected"]:
            report["reasons"].append(f"Data drift detected (score: {data_drift_report['drift_score']:.4f})")

        if model_drift_report["degraded"]:
            report["reasons"].append(f"Model performance degraded (drop: {model_drift_report['performance_drop']:.4f})")

        self.logger.info(
            "Retraining check completed",
            needs_retraining=needs_retraining,
            reasons=report["reasons"]
        )

        return report

    def run_retraining(
        self,
        train_loader,
        val_loader,
        test_loader,
        training_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Exécute le pipeline de retraining complet.

        Args:
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
            test_loader: DataLoader de test
            training_config: Configuration d'entraînement

        Returns:
            Rapport de retraining
        """
        self.logger.info("Starting retraining pipeline")
        self.metrics.counter_inc("retraining_runs_total")

        config = training_config or self.config["training_config"]

        report = {
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "steps": []
        }

        try:
            # Étape 1: Validation des données
            self.logger.info("Step 1: Data validation")
            validation_report = self.data_validator.validate_dataset(
                train_loader, val_loader, test_loader
            )
            report["steps"].append({
                "name": "data_validation",
                "status": "completed" if validation_report["validation_passed"] else "warning",
                "details": validation_report
            })

            if not validation_report["validation_passed"]:
                self.logger.warning("Data validation failed", issues=validation_report["issues"])

            # Étape 2: Entraînement
            self.logger.info("Step 2: Model training")
            from src.models.training import train_model

            model, history = train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=config["epochs"],
                learning_rate=config["learning_rate"],
                model_name=config["model_name"],
                pretrained=config["pretrained"]
            )

            report["steps"].append({
                "name": "training",
                "status": "completed",
                "details": {
                    "epochs": config["epochs"],
                    "final_train_acc": history["train_acc"][-1] if history["train_acc"] else 0,
                    "final_val_acc": history["val_acc"][-1] if history["val_acc"] else 0,
                    "best_val_acc": max(history["val_acc"]) if history["val_acc"] else 0
                }
            })

            # Étape 3: Évaluation
            self.logger.info("Step 3: Model evaluation")
            from src.utils.metrics import evaluate_model

            metrics = evaluate_model(model, test_loader)
            report["steps"].append({
                "name": "evaluation",
                "status": "completed",
                "details": metrics
            })

            # Étape 4: Enregistrement
            self.logger.info("Step 4: Model registration")
            version = self.model_registry.register_model(
                model_path="models/best_model.pth",
                model_name="cifar10-resnet18",
                metrics=metrics,
                tags={
                    "retraining_run": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "trigger": "automatic"
                },
                description="Retrained model"
            )

            report["steps"].append({
                "name": "registration",
                "status": "completed",
                "details": {"version": version}
            })

            # Étape 5: Promotion conditionnelle
            if self.config["auto_promote"]:
                self.logger.info("Step 5: Auto-promotion check")
                promotion_result = self._check_and_promote(
                    version,
                    metrics,
                    "cifar10-resnet18"
                )
                report["steps"].append({
                    "name": "promotion",
                    "status": "completed" if promotion_result["promoted"] else "skipped",
                    "details": promotion_result
                })

            report["status"] = "completed"
            report["completed_at"] = datetime.now().isoformat()
            report["new_version"] = version

            self.metrics.counter_inc("retraining_success_total")

        except Exception as e:
            report["status"] = "failed"
            report["error"] = str(e)
            report["completed_at"] = datetime.now().isoformat()

            self.logger.error("Retraining failed", error=str(e))
            self.metrics.counter_inc("retraining_failures_total")

        self.logger.info(
            "Retraining pipeline completed",
            status=report["status"],
            new_version=report.get("new_version")
        )

        return report

    def _check_and_promote(
        self,
        version: str,
        metrics: Dict[str, float],
        model_name: str
    ) -> Dict[str, Any]:
        """Vérifie et promeut le modèle si les critères sont remplis."""
        # Récupérer les métriques de production
        try:
            prod_model = self.model_registry.load_model(model_name, stage="production")
            prod_metrics = prod_model["metadata"]["metrics"]
            prod_accuracy = prod_metrics.get("accuracy", 0)
        except (ValueError, KeyError):
            prod_accuracy = 0

        new_accuracy = metrics.get("accuracy", 0)
        improvement = new_accuracy - prod_accuracy

        if improvement >= self.config["min_accuracy_improvement"]:
            # Promouvoir en staging d'abord
            self.model_registry.promote_model(model_name, version, "staging")

            return {
                "promoted": True,
                "stage": "staging",
                "improvement": improvement,
                "message": f"Model promoted to staging (improvement: {improvement:.4f})"
            }

        return {
            "promoted": False,
            "improvement": improvement,
            "min_required": self.config["min_accuracy_improvement"],
            "message": "Improvement insufficient for promotion"
        }

    def schedule_retraining(
        self,
        trigger: str = "manual",
        schedule: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Planifie un retraining.

        Args:
            trigger: Type de déclenchement (manual, scheduled, drift)
            schedule: Expression cron (optionnel)

        Returns:
            Informations de planification
        """
        job_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return {
            "job_id": job_id,
            "trigger": trigger,
            "schedule": schedule,
            "status": "scheduled",
            "created_at": datetime.now().isoformat()
        }

    def generate_retraining_report(
        self,
        retraining_result: Dict[str, Any]
    ) -> str:
        """Génère un rapport de retraining au format Markdown."""
        report = f"""# Retraining Report

## Summary
- **Started:** {retraining_result.get('started_at', 'N/A')}
- **Completed:** {retraining_result.get('completed_at', 'N/A')}
- **Status:** {'✅ ' + retraining_result['status'].upper() if retraining_result['status'] == 'completed' else '❌ ' + retraining_result['status'].upper()}
- **New Version:** {retraining_result.get('new_version', 'N/A')}

## Pipeline Steps
"""
        for step in retraining_result.get("steps", []):
            status_icon = "✅" if step["status"] == "completed" else "⚠️" if step["status"] == "warning" else "❌"
            report += f"\n### {step['name'].replace('_', ' ').title()}\n"
            report += f"**Status:** {status_icon} {step['status']}\n\n"

            if step.get("details"):
                for key, value in step["details"].items():
                    if isinstance(value, float):
                        report += f"- {key}: {value:.4f}\n"
                    elif isinstance(value, dict):
                        report += f"- {key}:\n"
                        for k, v in value.items():
                            report += f"  - {k}: {v}\n"
                    else:
                        report += f"- {key}: {value}\n"

        if retraining_result.get("error"):
            report += f"\n## Error\n```\n{retraining_result['error']}\n```\n"

        return report


class RetrainingTrigger:
    """
    Gestionnaire de déclencheurs de retraining.
    """

    def __init__(self):
        self.triggers: List[Dict[str, Any]] = []
        self.callbacks: List[Callable] = []

    def register_trigger(
        self,
        name: str,
        condition: Callable[[], bool],
        priority: int = 0
    ):
        """Enregistre un nouveau déclencheur."""
        self.triggers.append({
            "name": name,
            "condition": condition,
            "priority": priority
        })
        self.triggers.sort(key=lambda x: x["priority"], reverse=True)

    def register_callback(self, callback: Callable):
        """Enregistre un callback à appeler lors du déclenchement."""
        self.callbacks.append(callback)

    def check_triggers(self) -> Optional[str]:
        """Vérifie tous les déclencheurs."""
        for trigger in self.triggers:
            try:
                if trigger["condition"]():
                    return trigger["name"]
            except Exception:
                pass
        return None

    def fire(self, trigger_name: str):
        """Déclenche les callbacks."""
        for callback in self.callbacks:
            try:
                callback(trigger_name)
            except Exception:
                pass
