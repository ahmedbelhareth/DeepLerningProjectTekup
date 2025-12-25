"""
Feature Store - Gestion centralisée des features

Ce module fournit:
- Stockage et versioning des features
- Calcul et cache des transformations
- Gestion des features en temps réel et batch
- Statistiques et métadonnées des features
"""

import os
import json
import pickle
import hashlib
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
from collections import OrderedDict


class FeatureStore:
    """
    Feature Store simple pour la gestion des features ML.

    Fonctionnalités:
    - Stockage persistant des features
    - Versioning des transformations
    - Cache des features calculées
    - Statistiques des features
    """

    def __init__(
        self,
        store_path: str = "feature_store",
        cache_size: int = 1000
    ):
        """
        Initialise le Feature Store.

        Args:
            store_path: Chemin vers le répertoire de stockage
            cache_size: Taille maximale du cache en mémoire
        """
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.features_path = self.store_path / "features"
        self.features_path.mkdir(exist_ok=True)

        self.metadata_path = self.store_path / "metadata"
        self.metadata_path.mkdir(exist_ok=True)

        self.cache_size = cache_size
        self._cache: OrderedDict = OrderedDict()

        self._load_metadata()

    def _load_metadata(self):
        """Charge les métadonnées du store."""
        metadata_file = self.metadata_path / "store_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "created_at": datetime.now().isoformat(),
                "feature_groups": {},
                "transformations": {}
            }
            self._save_metadata()

    def _save_metadata(self):
        """Sauvegarde les métadonnées."""
        metadata_file = self.metadata_path / "store_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def _compute_hash(self, data: Any) -> str:
        """Calcule un hash unique pour les données."""
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        elif not isinstance(data, bytes):
            data = str(data).encode()
        return hashlib.md5(data).hexdigest()[:12]

    def _update_cache(self, key: str, value: Any):
        """Met à jour le cache LRU."""
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def register_feature_group(
        self,
        name: str,
        description: str,
        features: List[Dict[str, Any]],
        source: str = "computed"
    ) -> str:
        """
        Enregistre un groupe de features.

        Args:
            name: Nom du groupe
            description: Description du groupe
            features: Liste des définitions de features
            source: Source des features (computed, raw, external)

        Returns:
            ID du groupe de features
        """
        group_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.metadata["feature_groups"][name] = {
            "id": group_id,
            "description": description,
            "features": features,
            "source": source,
            "created_at": datetime.now().isoformat(),
            "version": 1
        }

        self._save_metadata()
        return group_id

    def store_features(
        self,
        feature_group: str,
        entity_id: str,
        features: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Stocke les features pour une entité.

        Args:
            feature_group: Nom du groupe de features
            entity_id: Identifiant de l'entité
            features: Dictionnaire des features
            timestamp: Timestamp des features

        Returns:
            ID de stockage
        """
        if timestamp is None:
            timestamp = datetime.now()

        storage_id = f"{feature_group}_{entity_id}_{self._compute_hash(features)}"

        # Convertir les tensors en numpy pour le stockage
        stored_features = {}
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                stored_features[key] = value.numpy().tolist()
            elif isinstance(value, np.ndarray):
                stored_features[key] = value.tolist()
            else:
                stored_features[key] = value

        feature_data = {
            "storage_id": storage_id,
            "feature_group": feature_group,
            "entity_id": entity_id,
            "features": stored_features,
            "timestamp": timestamp.isoformat(),
            "created_at": datetime.now().isoformat()
        }

        # Sauvegarder sur disque
        feature_file = self.features_path / f"{storage_id}.json"
        with open(feature_file, 'w') as f:
            json.dump(feature_data, f, indent=2)

        # Mettre en cache
        self._update_cache(storage_id, feature_data)

        return storage_id

    def get_features(
        self,
        feature_group: str,
        entity_id: str,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Récupère les features pour une entité.

        Args:
            feature_group: Nom du groupe de features
            entity_id: Identifiant de l'entité
            version: Version spécifique (optionnel)

        Returns:
            Dictionnaire des features ou None
        """
        # Chercher dans le cache
        cache_key = f"{feature_group}_{entity_id}"
        for key, value in reversed(list(self._cache.items())):
            if key.startswith(cache_key):
                return value["features"]

        # Chercher sur disque
        pattern = f"{feature_group}_{entity_id}_*.json"
        matching_files = list(self.features_path.glob(pattern))

        if not matching_files:
            return None

        # Prendre le plus récent
        latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
        with open(latest_file, 'r') as f:
            data = json.load(f)

        self._update_cache(data["storage_id"], data)
        return data["features"]

    def compute_image_features(
        self,
        image: Union[torch.Tensor, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Calcule les features d'une image CIFAR-10.

        Args:
            image: Image sous forme de tensor ou array

        Returns:
            Dictionnaire des features calculées
        """
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # Ajouter dimension batch si nécessaire
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        features = {}

        # Statistiques de base par canal
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = image[:, i, :, :]
            features[f'{channel}_mean'] = float(channel_data.mean())
            features[f'{channel}_std'] = float(channel_data.std())
            features[f'{channel}_min'] = float(channel_data.min())
            features[f'{channel}_max'] = float(channel_data.max())

        # Statistiques globales
        features['global_mean'] = float(image.mean())
        features['global_std'] = float(image.std())

        # Histogramme simplifié
        hist = torch.histc(image.flatten(), bins=10, min=-3, max=3)
        for i, count in enumerate(hist):
            features[f'hist_bin_{i}'] = float(count)

        # Statistiques spatiales
        features['brightness'] = float(image.mean(dim=1).mean())
        features['contrast'] = float(image.std(dim=(2, 3)).mean())

        return features

    def register_transformation(
        self,
        name: str,
        transform_fn: str,
        input_features: List[str],
        output_features: List[str],
        description: str = ""
    ) -> str:
        """
        Enregistre une transformation de features.

        Args:
            name: Nom de la transformation
            transform_fn: Code de la fonction (pour documentation)
            input_features: Features d'entrée
            output_features: Features de sortie
            description: Description de la transformation

        Returns:
            ID de la transformation
        """
        transform_id = f"{name}_v{len(self.metadata['transformations']) + 1}"

        self.metadata["transformations"][name] = {
            "id": transform_id,
            "transform_fn": transform_fn,
            "input_features": input_features,
            "output_features": output_features,
            "description": description,
            "created_at": datetime.now().isoformat()
        }

        self._save_metadata()
        return transform_id

    def get_feature_statistics(
        self,
        feature_group: str
    ) -> Dict[str, Any]:
        """
        Calcule les statistiques d'un groupe de features.

        Args:
            feature_group: Nom du groupe de features

        Returns:
            Statistiques des features
        """
        pattern = f"{feature_group}_*.json"
        matching_files = list(self.features_path.glob(pattern))

        if not matching_files:
            return {"error": "No features found for this group"}

        all_features = []
        for file in matching_files:
            with open(file, 'r') as f:
                data = json.load(f)
                all_features.append(data["features"])

        # Agréger les statistiques
        stats = {}
        if all_features:
            feature_names = all_features[0].keys()
            for name in feature_names:
                values = [f[name] for f in all_features if name in f and isinstance(f[name], (int, float))]
                if values:
                    stats[name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "count": len(values)
                    }

        return {
            "feature_group": feature_group,
            "num_entities": len(matching_files),
            "feature_statistics": stats
        }

    def list_feature_groups(self) -> List[Dict[str, Any]]:
        """Liste tous les groupes de features."""
        return [
            {
                "name": name,
                "description": info["description"],
                "num_features": len(info["features"]),
                "source": info["source"],
                "created_at": info["created_at"]
            }
            for name, info in self.metadata["feature_groups"].items()
        ]

    def export_features(
        self,
        feature_group: str,
        format: str = "csv"
    ) -> str:
        """
        Exporte les features au format spécifié.

        Args:
            feature_group: Nom du groupe de features
            format: Format d'export (csv, parquet, json)

        Returns:
            Chemin du fichier exporté
        """
        pattern = f"{feature_group}_*.json"
        matching_files = list(self.features_path.glob(pattern))

        if not matching_files:
            raise ValueError(f"No features found for group: {feature_group}")

        # Collecter toutes les features
        records = []
        for file in matching_files:
            with open(file, 'r') as f:
                data = json.load(f)
                record = {"entity_id": data["entity_id"], **data["features"]}
                records.append(record)

        export_path = self.store_path / "exports"
        export_path.mkdir(exist_ok=True)

        if format == "csv":
            import csv
            file_path = export_path / f"{feature_group}_export.csv"
            if records:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)
        elif format == "json":
            file_path = export_path / f"{feature_group}_export.json"
            with open(file_path, 'w') as f:
                json.dump(records, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return str(file_path)

    def clear_cache(self):
        """Vide le cache en mémoire."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        return {
            "cache_size": len(self._cache),
            "max_size": self.cache_size,
            "utilization": len(self._cache) / self.cache_size
        }
