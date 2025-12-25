"""
A/B Testing - Infrastructure pour les tests A/B de modèles

Ce module fournit:
- Gestion des expériences A/B
- Routage du trafic entre modèles
- Analyse statistique des résultats
- Décision automatique de promotion
"""

import random
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
from scipy import stats as scipy_stats
import hashlib


class ABTestManager:
    """
    Gestionnaire de tests A/B pour les modèles ML.

    Fonctionnalités:
    - Création et gestion des expériences
    - Routage du trafic avec poids configurables
    - Collecte des métriques par variante
    - Analyse statistique (test t, chi-carré)
    - Détermination automatique du gagnant
    """

    def __init__(
        self,
        experiment_path: str = "ab_experiments",
        min_sample_size: int = 100,
        confidence_level: float = 0.95
    ):
        """
        Initialise le gestionnaire A/B.

        Args:
            experiment_path: Chemin pour stocker les expériences
            min_sample_size: Taille minimale d'échantillon par variante
            confidence_level: Niveau de confiance pour les tests statistiques
        """
        self.experiment_path = Path(experiment_path)
        self.experiment_path.mkdir(parents=True, exist_ok=True)

        self.min_sample_size = min_sample_size
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

        self._load_experiments()

    def _load_experiments(self):
        """Charge les expériences existantes."""
        experiments_file = self.experiment_path / "experiments.json"
        if experiments_file.exists():
            with open(experiments_file, 'r') as f:
                self.experiments = json.load(f)
        else:
            self.experiments = {}
            self._save_experiments()

    def _save_experiments(self):
        """Sauvegarde les expériences."""
        experiments_file = self.experiment_path / "experiments.json"
        with open(experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)

    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Dict[str, Any]],
        traffic_weights: Optional[List[float]] = None,
        metrics: Optional[List[str]] = None
    ) -> str:
        """
        Crée une nouvelle expérience A/B.

        Args:
            name: Nom de l'expérience
            description: Description de l'expérience
            variants: Liste des variantes (nom, model_path, etc.)
            traffic_weights: Poids de trafic pour chaque variante
            metrics: Métriques à suivre

        Returns:
            ID de l'expérience
        """
        experiment_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if traffic_weights is None:
            traffic_weights = [1.0 / len(variants)] * len(variants)

        if metrics is None:
            metrics = ["accuracy", "latency_ms", "confidence"]

        # Normaliser les poids
        total_weight = sum(traffic_weights)
        traffic_weights = [w / total_weight for w in traffic_weights]

        self.experiments[experiment_id] = {
            "name": name,
            "description": description,
            "variants": variants,
            "traffic_weights": traffic_weights,
            "metrics": metrics,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "results": {v["name"]: {"samples": 0, "metrics": defaultdict(list)} for v in variants}
        }

        self._save_experiments()
        return experiment_id

    def get_variant(
        self,
        experiment_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sélectionne une variante pour une requête.

        Args:
            experiment_id: ID de l'expérience
            user_id: ID utilisateur pour routage cohérent (optionnel)

        Returns:
            Variante sélectionnée
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        experiment = self.experiments[experiment_id]

        if experiment["status"] != "active":
            raise ValueError(f"Experiment '{experiment_id}' is not active")

        variants = experiment["variants"]
        weights = experiment["traffic_weights"]

        # Routage cohérent basé sur user_id
        if user_id:
            hash_value = int(hashlib.md5(f"{experiment_id}_{user_id}".encode()).hexdigest(), 16)
            random.seed(hash_value)

        # Sélection pondérée
        selected = random.choices(variants, weights=weights, k=1)[0]

        return selected

    def record_result(
        self,
        experiment_id: str,
        variant_name: str,
        metrics: Dict[str, float]
    ):
        """
        Enregistre un résultat pour une variante.

        Args:
            experiment_id: ID de l'expérience
            variant_name: Nom de la variante
            metrics: Métriques observées
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        results = self.experiments[experiment_id]["results"]

        if variant_name not in results:
            raise ValueError(f"Variant '{variant_name}' not found in experiment")

        results[variant_name]["samples"] += 1

        for metric, value in metrics.items():
            if metric not in results[variant_name]["metrics"]:
                results[variant_name]["metrics"][metric] = []
            results[variant_name]["metrics"][metric].append(value)

        self._save_experiments()

    def analyze_experiment(
        self,
        experiment_id: str
    ) -> Dict[str, Any]:
        """
        Analyse les résultats d'une expérience.

        Args:
            experiment_id: ID de l'expérience

        Returns:
            Analyse statistique complète
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        experiment = self.experiments[experiment_id]
        results = experiment["results"]
        metrics = experiment["metrics"]

        analysis = {
            "experiment_id": experiment_id,
            "name": experiment["name"],
            "status": experiment["status"],
            "created_at": experiment["created_at"],
            "variants": {},
            "statistical_tests": {},
            "recommendations": []
        }

        # Analyser chaque variante
        for variant_name, variant_results in results.items():
            variant_analysis = {
                "samples": variant_results["samples"],
                "metrics": {}
            }

            for metric in metrics:
                if metric in variant_results["metrics"] and variant_results["metrics"][metric]:
                    values = variant_results["metrics"][metric]
                    variant_analysis["metrics"][metric] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "median": np.median(values)
                    }

            analysis["variants"][variant_name] = variant_analysis

        # Tests statistiques entre variantes
        variant_names = list(results.keys())
        if len(variant_names) >= 2:
            for metric in metrics:
                values_a = results[variant_names[0]]["metrics"].get(metric, [])
                values_b = results[variant_names[1]]["metrics"].get(metric, [])

                if len(values_a) >= self.min_sample_size and len(values_b) >= self.min_sample_size:
                    # Test t de Welch (inégalité des variances)
                    t_stat, p_value = scipy_stats.ttest_ind(values_a, values_b, equal_var=False)

                    analysis["statistical_tests"][metric] = {
                        "test": "welch_t_test",
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < self.alpha,
                        "confidence_level": self.confidence_level
                    }

                    # Déterminer le gagnant
                    if p_value < self.alpha:
                        mean_a = np.mean(values_a)
                        mean_b = np.mean(values_b)

                        # Pour accuracy, plus haut est mieux; pour latency, plus bas est mieux
                        if metric in ["accuracy", "confidence"]:
                            winner = variant_names[0] if mean_a > mean_b else variant_names[1]
                        else:
                            winner = variant_names[0] if mean_a < mean_b else variant_names[1]

                        analysis["statistical_tests"][metric]["winner"] = winner
                        analysis["recommendations"].append(
                            f"For {metric}: {winner} performs significantly better"
                        )

        # Recommandation finale
        if analysis["recommendations"]:
            analysis["ready_for_decision"] = True
        else:
            total_samples = sum(r["samples"] for r in results.values())
            needed = self.min_sample_size * len(variant_names)
            analysis["ready_for_decision"] = False
            analysis["recommendations"].append(
                f"Need more data: {total_samples}/{needed} samples collected"
            )

        return analysis

    def conclude_experiment(
        self,
        experiment_id: str,
        winner: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Conclut une expérience et désigne un gagnant.

        Args:
            experiment_id: ID de l'expérience
            winner: Variante gagnante (auto-déterminée si None)

        Returns:
            Résumé de conclusion
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        experiment = self.experiments[experiment_id]

        if winner is None:
            # Auto-détermination basée sur l'analyse
            analysis = self.analyze_experiment(experiment_id)
            winners = []
            for metric, test in analysis.get("statistical_tests", {}).items():
                if test.get("winner"):
                    winners.append(test["winner"])

            if winners:
                # Variante qui gagne le plus de métriques
                from collections import Counter
                winner = Counter(winners).most_common(1)[0][0]
            else:
                winner = experiment["variants"][0]["name"]  # Par défaut

        experiment["status"] = "completed"
        experiment["completed_at"] = datetime.now().isoformat()
        experiment["winner"] = winner

        self._save_experiments()

        return {
            "experiment_id": experiment_id,
            "winner": winner,
            "completed_at": experiment["completed_at"],
            "message": f"Experiment concluded. Winner: {winner}"
        }

    def list_experiments(
        self,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Liste toutes les expériences.

        Args:
            status: Filtrer par statut (active, completed, archived)

        Returns:
            Liste des expériences
        """
        experiments = []
        for exp_id, exp in self.experiments.items():
            if status is None or exp["status"] == status:
                experiments.append({
                    "id": exp_id,
                    "name": exp["name"],
                    "status": exp["status"],
                    "variants": [v["name"] for v in exp["variants"]],
                    "total_samples": sum(r["samples"] for r in exp["results"].values()),
                    "created_at": exp["created_at"],
                    "winner": exp.get("winner")
                })
        return experiments

    def get_experiment_summary(
        self,
        experiment_id: str
    ) -> str:
        """Génère un résumé de l'expérience au format Markdown."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        exp = self.experiments[experiment_id]
        analysis = self.analyze_experiment(experiment_id)

        summary = f"""# A/B Test Report: {exp['name']}

## Overview
- **Experiment ID:** {experiment_id}
- **Status:** {exp['status']}
- **Created:** {exp['created_at']}
- **Description:** {exp['description']}

## Variants
| Variant | Traffic Weight | Samples |
|---------|---------------|---------|
"""
        for i, variant in enumerate(exp["variants"]):
            samples = exp["results"][variant["name"]]["samples"]
            weight = f"{exp['traffic_weights'][i]:.1%}"
            summary += f"| {variant['name']} | {weight} | {samples} |\n"

        summary += "\n## Metrics Summary\n"
        for variant_name, variant_data in analysis["variants"].items():
            summary += f"\n### {variant_name}\n"
            for metric, stats in variant_data.get("metrics", {}).items():
                summary += f"- **{metric}:** mean={stats['mean']:.4f}, std={stats['std']:.4f}\n"

        if analysis.get("statistical_tests"):
            summary += "\n## Statistical Analysis\n"
            for metric, test in analysis["statistical_tests"].items():
                summary += f"\n### {metric}\n"
                summary += f"- Test: {test['test']}\n"
                summary += f"- p-value: {test['p_value']:.6f}\n"
                summary += f"- Significant: {'Yes' if test['significant'] else 'No'}\n"
                if test.get("winner"):
                    summary += f"- Winner: **{test['winner']}**\n"

        if analysis.get("recommendations"):
            summary += "\n## Recommendations\n"
            for rec in analysis["recommendations"]:
                summary += f"- {rec}\n"

        return summary


class CanaryDeployment:
    """
    Gestionnaire de déploiement canary.

    Permet un rollout progressif avec monitoring.
    """

    def __init__(
        self,
        initial_traffic: float = 0.05,
        max_traffic: float = 1.0,
        increment: float = 0.10,
        rollback_threshold: float = 0.95
    ):
        """
        Initialise le gestionnaire canary.

        Args:
            initial_traffic: Trafic initial vers la nouvelle version
            max_traffic: Trafic maximum
            increment: Incrément de trafic
            rollback_threshold: Seuil de performance pour rollback
        """
        self.initial_traffic = initial_traffic
        self.max_traffic = max_traffic
        self.increment = increment
        self.rollback_threshold = rollback_threshold

        self.current_traffic = 0.0
        self.baseline_metrics: Dict[str, float] = {}
        self.canary_metrics: Dict[str, float] = {}
        self.status = "inactive"

    def start_deployment(
        self,
        baseline_metrics: Dict[str, float]
    ):
        """Démarre un déploiement canary."""
        self.baseline_metrics = baseline_metrics
        self.current_traffic = self.initial_traffic
        self.canary_metrics = {}
        self.status = "active"

        return {
            "status": "started",
            "initial_traffic": self.initial_traffic,
            "baseline_metrics": self.baseline_metrics
        }

    def update_metrics(
        self,
        canary_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Met à jour les métriques canary et décide de l'action."""
        self.canary_metrics = canary_metrics

        # Comparer avec baseline
        for metric, baseline_value in self.baseline_metrics.items():
            if metric in canary_metrics:
                canary_value = canary_metrics[metric]
                ratio = canary_value / baseline_value if baseline_value > 0 else 0

                if ratio < self.rollback_threshold:
                    return self._rollback(
                        f"Performance degradation on {metric}: {ratio:.2%} of baseline"
                    )

        # Si tout va bien, augmenter le trafic
        if self.current_traffic < self.max_traffic:
            self.current_traffic = min(self.current_traffic + self.increment, self.max_traffic)

            if self.current_traffic >= self.max_traffic:
                return self._complete()

        return {
            "action": "continue",
            "current_traffic": self.current_traffic,
            "next_increment_at": self.current_traffic + self.increment
        }

    def _rollback(self, reason: str) -> Dict[str, Any]:
        """Effectue un rollback."""
        self.status = "rolled_back"
        self.current_traffic = 0.0

        return {
            "action": "rollback",
            "reason": reason,
            "baseline_restored": True
        }

    def _complete(self) -> Dict[str, Any]:
        """Complète le déploiement."""
        self.status = "completed"

        return {
            "action": "complete",
            "message": "Canary deployment successful",
            "final_traffic": self.max_traffic
        }

    def get_routing_decision(self) -> str:
        """Retourne la décision de routage."""
        if self.status != "active":
            return "baseline"

        if random.random() < self.current_traffic:
            return "canary"
        return "baseline"
