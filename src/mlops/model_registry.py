"""
Model Registry - Gestion et versioning des modèles

Ce module fournit un système de registre pour:
- Enregistrer les modèles avec versioning sémantique
- Gérer les métadonnées et métriques
- Promouvoir les modèles entre environnements (staging, production)
- Charger les modèles par version
"""

import os
import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any, Union
import torch


class ModelRegistry:
    """
    Registre de modèles avec versioning sémantique.

    Fonctionnalités:
    - Versioning automatique (major.minor.patch)
    - Stockage des métadonnées et métriques
    - Gestion des stages (development, staging, production)
    - Comparaison de versions
    - Rollback facile
    """

    def __init__(self, registry_path: str = "model_registry"):
        """
        Initialise le registre de modèles.

        Args:
            registry_path: Chemin vers le répertoire du registre
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.models_path = self.registry_path / "models"
        self.models_path.mkdir(exist_ok=True)

        self.metadata_file = self.registry_path / "registry.json"
        self._load_registry()

    def _load_registry(self) -> None:
        """Charge le registre depuis le fichier JSON."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "models": {},
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }
            self._save_registry()

    def _save_registry(self) -> None:
        """Sauvegarde le registre dans le fichier JSON."""
        self.registry["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)

    def _compute_checksum(self, file_path: str) -> str:
        """Calcule le checksum MD5 d'un fichier."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_next_version(self, model_name: str, bump: str = "patch") -> str:
        """
        Calcule la prochaine version selon le versioning sémantique.

        Args:
            model_name: Nom du modèle
            bump: Type d'incrément (major, minor, patch)

        Returns:
            Nouvelle version sous forme de string
        """
        if model_name not in self.registry["models"]:
            return "1.0.0"

        versions = self.registry["models"][model_name]["versions"]
        if not versions:
            return "1.0.0"

        latest = sorted(versions.keys(), key=lambda v: [int(x) for x in v.split(".")])[-1]
        major, minor, patch = map(int, latest.split("."))

        if bump == "major":
            return f"{major + 1}.0.0"
        elif bump == "minor":
            return f"{major}.{minor + 1}.0"
        else:
            return f"{major}.{minor}.{patch + 1}"

    def register_model(
        self,
        model_path: str,
        model_name: str,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: str = "",
        bump: str = "patch"
    ) -> str:
        """
        Enregistre un nouveau modèle dans le registre.

        Args:
            model_path: Chemin vers le fichier du modèle (.pth)
            model_name: Nom unique du modèle
            metrics: Métriques de performance (accuracy, f1, etc.)
            tags: Tags additionnels
            description: Description du modèle
            bump: Type de version bump (major, minor, patch)

        Returns:
            Version du modèle enregistré
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        version = self._get_next_version(model_name, bump)

        # Créer le répertoire pour ce modèle/version
        version_path = self.models_path / model_name / version
        version_path.mkdir(parents=True, exist_ok=True)

        # Copier le modèle
        dest_path = version_path / "model.pth"
        shutil.copy2(model_path, dest_path)

        # Calculer le checksum
        checksum = self._compute_checksum(str(dest_path))

        # Métadonnées de version
        version_metadata = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "checksum": checksum,
            "file_size_mb": os.path.getsize(dest_path) / (1024 * 1024),
            "metrics": metrics or {},
            "tags": tags or {},
            "description": description,
            "stage": "development",
            "model_path": str(dest_path)
        }

        # Mettre à jour le registre
        if model_name not in self.registry["models"]:
            self.registry["models"][model_name] = {
                "created_at": datetime.now().isoformat(),
                "versions": {},
                "production_version": None,
                "staging_version": None
            }

        self.registry["models"][model_name]["versions"][version] = version_metadata
        self._save_registry()

        print(f"Model '{model_name}' registered with version {version}")
        return version

    def promote_model(
        self,
        model_name: str,
        version: str,
        stage: str
    ) -> bool:
        """
        Promeut un modèle vers un stage (staging ou production).

        Args:
            model_name: Nom du modèle
            version: Version à promouvoir
            stage: Stage cible (staging, production)

        Returns:
            True si la promotion a réussi
        """
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model '{model_name}' not found")

        if version not in self.registry["models"][model_name]["versions"]:
            raise ValueError(f"Version '{version}' not found for model '{model_name}'")

        if stage not in ["staging", "production", "development", "archived"]:
            raise ValueError(f"Invalid stage: {stage}")

        # Mettre à jour le stage de la version
        self.registry["models"][model_name]["versions"][version]["stage"] = stage

        # Mettre à jour le pointeur de stage
        if stage == "production":
            # Archiver l'ancienne version de production
            old_prod = self.registry["models"][model_name]["production_version"]
            if old_prod and old_prod != version:
                self.registry["models"][model_name]["versions"][old_prod]["stage"] = "archived"
            self.registry["models"][model_name]["production_version"] = version
        elif stage == "staging":
            self.registry["models"][model_name]["staging_version"] = version

        self._save_registry()
        print(f"Model '{model_name}' version {version} promoted to {stage}")
        return True

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Charge un modèle depuis le registre.

        Args:
            model_name: Nom du modèle
            version: Version spécifique (optionnel)
            stage: Stage à charger (production, staging) si pas de version

        Returns:
            Dictionnaire avec le modèle et ses métadonnées
        """
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model '{model_name}' not found")

        model_info = self.registry["models"][model_name]

        # Déterminer la version à charger
        if version is None:
            if stage == "production":
                version = model_info["production_version"]
            elif stage == "staging":
                version = model_info["staging_version"]
            else:
                # Charger la dernière version
                versions = list(model_info["versions"].keys())
                version = sorted(versions, key=lambda v: [int(x) for x in v.split(".")])[-1]

        if version is None:
            raise ValueError(f"No version available for model '{model_name}'")

        version_info = model_info["versions"][version]
        model_path = version_info["model_path"]

        # Charger le modèle
        checkpoint = torch.load(model_path, map_location='cpu')

        return {
            "checkpoint": checkpoint,
            "version": version,
            "metadata": version_info
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """Liste tous les modèles enregistrés."""
        models = []
        for name, info in self.registry["models"].items():
            models.append({
                "name": name,
                "num_versions": len(info["versions"]),
                "production_version": info["production_version"],
                "staging_version": info["staging_version"],
                "created_at": info["created_at"]
            })
        return models

    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Liste toutes les versions d'un modèle."""
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model '{model_name}' not found")

        versions = []
        for version, info in self.registry["models"][model_name]["versions"].items():
            versions.append({
                "version": version,
                "stage": info["stage"],
                "created_at": info["created_at"],
                "metrics": info["metrics"],
                "file_size_mb": info["file_size_mb"]
            })

        return sorted(versions, key=lambda v: [int(x) for x in v["version"].split(".")])

    def compare_versions(
        self,
        model_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare deux versions d'un modèle."""
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model '{model_name}' not found")

        v1_info = self.registry["models"][model_name]["versions"].get(version1)
        v2_info = self.registry["models"][model_name]["versions"].get(version2)

        if not v1_info or not v2_info:
            raise ValueError("One or both versions not found")

        comparison = {
            "version1": version1,
            "version2": version2,
            "metrics_comparison": {},
            "size_diff_mb": v2_info["file_size_mb"] - v1_info["file_size_mb"]
        }

        # Comparer les métriques
        all_metrics = set(v1_info["metrics"].keys()) | set(v2_info["metrics"].keys())
        for metric in all_metrics:
            m1 = v1_info["metrics"].get(metric)
            m2 = v2_info["metrics"].get(metric)
            if m1 is not None and m2 is not None:
                comparison["metrics_comparison"][metric] = {
                    "v1": m1,
                    "v2": m2,
                    "diff": m2 - m1,
                    "improved": m2 > m1
                }

        return comparison

    def delete_version(self, model_name: str, version: str) -> bool:
        """Supprime une version d'un modèle."""
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model '{model_name}' not found")

        if version not in self.registry["models"][model_name]["versions"]:
            raise ValueError(f"Version '{version}' not found")

        version_info = self.registry["models"][model_name]["versions"][version]

        # Vérifier que ce n'est pas la version de production
        if version == self.registry["models"][model_name]["production_version"]:
            raise ValueError("Cannot delete production version")

        # Supprimer le fichier
        model_path = Path(version_info["model_path"])
        if model_path.exists():
            shutil.rmtree(model_path.parent)

        # Supprimer du registre
        del self.registry["models"][model_name]["versions"][version]
        self._save_registry()

        print(f"Version {version} of '{model_name}' deleted")
        return True

    def export_model_card(self, model_name: str, version: str) -> str:
        """Génère une Model Card au format Markdown."""
        if model_name not in self.registry["models"]:
            raise ValueError(f"Model '{model_name}' not found")

        version_info = self.registry["models"][model_name]["versions"].get(version)
        if not version_info:
            raise ValueError(f"Version '{version}' not found")

        card = f"""# Model Card: {model_name}

## Model Details
- **Version:** {version}
- **Stage:** {version_info['stage']}
- **Created:** {version_info['created_at']}
- **File Size:** {version_info['file_size_mb']:.2f} MB
- **Checksum:** {version_info['checksum']}

## Description
{version_info['description'] or 'No description provided.'}

## Performance Metrics
"""
        for metric, value in version_info['metrics'].items():
            if isinstance(value, float):
                card += f"- **{metric}:** {value:.4f}\n"
            else:
                card += f"- **{metric}:** {value}\n"

        card += "\n## Tags\n"
        for tag, value in version_info['tags'].items():
            card += f"- **{tag}:** {value}\n"

        return card
