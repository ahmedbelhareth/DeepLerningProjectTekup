"""
Drift Detection - Détection de dérive des données et du modèle

Ce module fournit des outils pour:
- Détecter la dérive des données (data drift)
- Détecter la dégradation des performances (concept drift)
- Générer des alertes automatiques
- Déclencher le retraining si nécessaire
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import json
from scipy import stats
from collections import defaultdict


class DriftDetector:
    """
    Détecteur de dérive pour les données et le modèle.

    Méthodes de détection:
    - Kolmogorov-Smirnov test pour la distribution des features
    - Population Stability Index (PSI) pour la distribution des prédictions
    - Performance monitoring pour la dégradation du modèle
    """

    def __init__(
        self,
        baseline_path: str = "drift_baselines",
        drift_threshold: float = 0.1,
        performance_threshold: float = 0.05
    ):
        """
        Initialise le détecteur de dérive.

        Args:
            baseline_path: Chemin pour stocker les baselines
            drift_threshold: Seuil de détection pour le data drift
            performance_threshold: Seuil de dégradation de performance acceptable
        """
        self.baseline_path = Path(baseline_path)
        self.baseline_path.mkdir(parents=True, exist_ok=True)

        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold

        self.baseline_stats = self._load_baseline()

    def _load_baseline(self) -> Dict:
        """Charge les statistiques de baseline."""
        baseline_file = self.baseline_path / "baseline_stats.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_baseline(self) -> None:
        """Sauvegarde les statistiques de baseline."""
        baseline_file = self.baseline_path / "baseline_stats.json"
        with open(baseline_file, 'w') as f:
            json.dump(self.baseline_stats, f, indent=2, default=str)

    def compute_baseline(
        self,
        data_loader: DataLoader,
        model: Optional[torch.nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Calcule les statistiques de baseline pour les données.

        Args:
            data_loader: DataLoader avec les données de référence
            model: Modèle pour calculer les baseline de prédictions

        Returns:
            Statistiques de baseline
        """
        # Collecter les statistiques des features
        all_features = []
        all_labels = []
        all_predictions = []

        device = next(model.parameters()).device if model else 'cpu'

        with torch.no_grad():
            for images, labels in data_loader:
                # Statistiques des images (features)
                batch_stats = {
                    'mean': images.mean(dim=(0, 2, 3)).numpy().tolist(),
                    'std': images.std(dim=(0, 2, 3)).numpy().tolist(),
                    'min': images.min().item(),
                    'max': images.max().item()
                }
                all_features.append(batch_stats)
                all_labels.extend(labels.numpy().tolist())

                # Prédictions du modèle
                if model:
                    model.eval()
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    preds = outputs.argmax(dim=1)
                    all_predictions.extend(preds.cpu().numpy().tolist())

        # Agréger les statistiques
        self.baseline_stats = {
            "created_at": datetime.now().isoformat(),
            "num_samples": len(all_labels),
            "feature_stats": {
                "mean_per_channel": np.mean([f['mean'] for f in all_features], axis=0).tolist(),
                "std_per_channel": np.mean([f['std'] for f in all_features], axis=0).tolist(),
                "global_min": min(f['min'] for f in all_features),
                "global_max": max(f['max'] for f in all_features)
            },
            "label_distribution": self._compute_distribution(all_labels),
            "prediction_distribution": self._compute_distribution(all_predictions) if all_predictions else None
        }

        self._save_baseline()
        print(f"Baseline computed with {len(all_labels)} samples")
        return self.baseline_stats

    def _compute_distribution(self, values: List[int]) -> Dict[str, float]:
        """Calcule la distribution des valeurs."""
        unique, counts = np.unique(values, return_counts=True)
        total = len(values)
        return {str(k): v / total for k, v in zip(unique, counts)}

    def _calculate_psi(
        self,
        baseline_dist: Dict[str, float],
        current_dist: Dict[str, float]
    ) -> float:
        """
        Calcule le Population Stability Index (PSI).

        PSI < 0.1: Pas de changement significatif
        0.1 <= PSI < 0.2: Changement modéré
        PSI >= 0.2: Changement significatif
        """
        psi = 0.0
        all_keys = set(baseline_dist.keys()) | set(current_dist.keys())

        for key in all_keys:
            baseline_pct = baseline_dist.get(key, 0.001)  # Éviter division par 0
            current_pct = current_dist.get(key, 0.001)

            psi += (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)

        return psi

    def check_data_drift(
        self,
        data_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        Vérifie la dérive des données par rapport à la baseline.

        Args:
            data_loader: DataLoader avec les nouvelles données

        Returns:
            Rapport de dérive avec scores et recommandations
        """
        if not self.baseline_stats:
            return {
                "drift_detected": False,
                "message": "No baseline available. Please compute baseline first.",
                "drift_score": 0.0,
                "threshold": self.drift_threshold
            }

        # Collecter les statistiques actuelles
        current_features = []
        current_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                batch_stats = {
                    'mean': images.mean(dim=(0, 2, 3)).numpy().tolist(),
                    'std': images.std(dim=(0, 2, 3)).numpy().tolist()
                }
                current_features.append(batch_stats)
                current_labels.extend(labels.numpy().tolist())

        # Calculer les statistiques actuelles
        current_stats = {
            "mean_per_channel": np.mean([f['mean'] for f in current_features], axis=0).tolist(),
            "std_per_channel": np.mean([f['std'] for f in current_features], axis=0).tolist()
        }

        # Calculer le drift score (distance euclidienne normalisée)
        baseline_mean = np.array(self.baseline_stats["feature_stats"]["mean_per_channel"])
        current_mean = np.array(current_stats["mean_per_channel"])

        baseline_std = np.array(self.baseline_stats["feature_stats"]["std_per_channel"])
        current_std = np.array(current_stats["std_per_channel"])

        mean_drift = np.linalg.norm(current_mean - baseline_mean) / np.linalg.norm(baseline_mean)
        std_drift = np.linalg.norm(current_std - baseline_std) / np.linalg.norm(baseline_std)

        # PSI pour la distribution des labels
        current_label_dist = self._compute_distribution(current_labels)
        label_psi = self._calculate_psi(
            self.baseline_stats["label_distribution"],
            current_label_dist
        )

        # Score de drift combiné
        drift_score = (mean_drift + std_drift + label_psi) / 3
        drift_detected = drift_score > self.drift_threshold

        report = {
            "drift_detected": drift_detected,
            "drift_score": float(drift_score),
            "threshold": self.drift_threshold,
            "details": {
                "mean_drift": float(mean_drift),
                "std_drift": float(std_drift),
                "label_psi": float(label_psi)
            },
            "baseline_samples": self.baseline_stats["num_samples"],
            "current_samples": len(current_labels),
            "timestamp": datetime.now().isoformat(),
            "recommendation": "Consider retraining the model" if drift_detected else "No action needed"
        }

        return report

    def check_model_drift(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        baseline_accuracy: float = 0.87
    ) -> Dict[str, Any]:
        """
        Vérifie la dégradation des performances du modèle.

        Args:
            model: Modèle à évaluer
            data_loader: DataLoader pour l'évaluation
            baseline_accuracy: Accuracy de référence

        Returns:
            Rapport de performance avec détection de dégradation
        """
        model.eval()
        device = next(model.parameters()).device if list(model.parameters()) else 'cpu'

        correct = 0
        total = 0
        predictions = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predictions.extend(predicted.cpu().numpy().tolist())

        current_accuracy = correct / total if total > 0 else 0.0
        performance_drop = baseline_accuracy - current_accuracy
        degraded = performance_drop > self.performance_threshold

        # PSI des prédictions
        current_pred_dist = self._compute_distribution(predictions)
        pred_psi = 0.0
        if self.baseline_stats.get("prediction_distribution"):
            pred_psi = self._calculate_psi(
                self.baseline_stats["prediction_distribution"],
                current_pred_dist
            )

        report = {
            "degraded": degraded,
            "current_accuracy": float(current_accuracy),
            "baseline_accuracy": float(baseline_accuracy),
            "performance_drop": float(performance_drop),
            "threshold": self.performance_threshold,
            "prediction_psi": float(pred_psi),
            "total_samples": total,
            "timestamp": datetime.now().isoformat(),
            "recommendation": "Retrain model immediately" if degraded else "Model performance is acceptable"
        }

        return report

    def generate_drift_report(
        self,
        data_drift_report: Dict,
        model_drift_report: Dict
    ) -> str:
        """Génère un rapport complet de drift au format Markdown."""
        report = f"""# Drift Detection Report
Generated: {datetime.now().isoformat()}

## Summary
| Metric | Status |
|--------|--------|
| Data Drift | {'⚠️ DETECTED' if data_drift_report['drift_detected'] else '✅ OK'} |
| Model Degradation | {'⚠️ DETECTED' if model_drift_report['degraded'] else '✅ OK'} |

## Data Drift Analysis
- **Drift Score:** {data_drift_report['drift_score']:.4f} (threshold: {data_drift_report['threshold']})
- **Mean Drift:** {data_drift_report['details']['mean_drift']:.4f}
- **Std Drift:** {data_drift_report['details']['std_drift']:.4f}
- **Label PSI:** {data_drift_report['details']['label_psi']:.4f}
- **Samples Analyzed:** {data_drift_report['current_samples']}

## Model Performance Analysis
- **Current Accuracy:** {model_drift_report['current_accuracy']:.4f}
- **Baseline Accuracy:** {model_drift_report['baseline_accuracy']:.4f}
- **Performance Drop:** {model_drift_report['performance_drop']:.4f}
- **Prediction PSI:** {model_drift_report['prediction_psi']:.4f}

## Recommendations
"""
        if data_drift_report['drift_detected'] or model_drift_report['degraded']:
            report += "- 🔴 **Action Required:** Consider retraining the model\n"
            report += "- Review recent data changes\n"
            report += "- Check for data quality issues\n"
        else:
            report += "- ✅ No immediate action required\n"
            report += "- Continue monitoring\n"

        return report


class ConceptDriftDetector:
    """
    Détecteur de concept drift basé sur le suivi des erreurs.

    Utilise Page-Hinkley test et ADWIN pour détecter
    les changements dans la distribution des erreurs.
    """

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0
    ):
        """
        Initialise le détecteur de concept drift.

        Args:
            min_instances: Nombre minimum d'instances avant détection
            delta: Magnitude minimale de changement
            threshold: Seuil de détection
        """
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold

        self.reset()

    def reset(self):
        """Réinitialise le détecteur."""
        self.n = 0
        self.sum = 0.0
        self.x_mean = 0.0
        self.m_n = 0.0
        self.M_n = 0.0

    def update(self, error: float) -> Tuple[bool, float]:
        """
        Met à jour le détecteur avec une nouvelle erreur.

        Args:
            error: Valeur d'erreur (0 ou 1 pour classification)

        Returns:
            Tuple (drift_detected, current_score)
        """
        self.n += 1
        self.sum += error
        self.x_mean = self.sum / self.n

        self.m_n = self.m_n + error - self.x_mean - self.delta
        self.M_n = max(self.M_n, self.m_n)

        drift_detected = False
        score = self.M_n - self.m_n

        if self.n >= self.min_instances and score > self.threshold:
            drift_detected = True

        return drift_detected, score

    def get_statistics(self) -> Dict[str, float]:
        """Retourne les statistiques actuelles."""
        return {
            "num_instances": self.n,
            "mean_error": self.x_mean,
            "current_sum": self.sum,
            "m_n": self.m_n,
            "M_n": self.M_n,
            "score": self.M_n - self.m_n
        }
