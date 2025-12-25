"""
Data Validation - Validation de la qualité des données

Ce module fournit des outils pour:
- Valider l'intégrité des datasets
- Vérifier l'équilibre des classes
- Détecter les anomalies dans les données
- Générer des rapports de qualité
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from collections import Counter
import warnings


class DataValidator:
    """
    Validateur de données pour les pipelines ML.

    Vérifie:
    - Intégrité des données (NaN, valeurs extrêmes)
    - Équilibre des classes
    - Distribution des features
    - Qualité des images
    """

    def __init__(
        self,
        num_classes: int = 10,
        expected_image_size: Tuple[int, int, int] = (3, 32, 32),
        class_balance_threshold: float = 0.15
    ):
        """
        Initialise le validateur.

        Args:
            num_classes: Nombre de classes attendu
            expected_image_size: Taille attendue des images (C, H, W)
            class_balance_threshold: Seuil d'écart acceptable pour l'équilibre des classes
        """
        self.num_classes = num_classes
        self.expected_image_size = expected_image_size
        self.class_balance_threshold = class_balance_threshold

    def validate_dataset(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        Valide l'ensemble du dataset.

        Args:
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
            test_loader: DataLoader de test

        Returns:
            Rapport de validation complet
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "train_samples": 0,
            "val_samples": 0,
            "test_samples": 0,
            "data_integrity_ok": True,
            "class_balance_ok": True,
            "image_quality_ok": True,
            "issues": []
        }

        # Valider chaque split
        train_report = self._validate_split(train_loader, "train")
        val_report = self._validate_split(val_loader, "validation")
        test_report = self._validate_split(test_loader, "test")

        report["train_samples"] = train_report["num_samples"]
        report["val_samples"] = val_report["num_samples"]
        report["test_samples"] = test_report["num_samples"]

        # Agréger les résultats
        all_reports = [train_report, val_report, test_report]

        for r in all_reports:
            if not r["integrity_ok"]:
                report["data_integrity_ok"] = False
                report["issues"].extend(r["issues"])

            if not r["class_balance_ok"]:
                report["class_balance_ok"] = False
                if f"Class imbalance in {r['split_name']}" not in report["issues"]:
                    report["issues"].append(f"Class imbalance in {r['split_name']}")

            if not r["image_quality_ok"]:
                report["image_quality_ok"] = False

        # Vérifier les ratios des splits
        total = report["train_samples"] + report["val_samples"] + report["test_samples"]
        if total > 0:
            train_ratio = report["train_samples"] / total
            if train_ratio < 0.6:
                report["issues"].append(f"Training set too small ({train_ratio:.1%})")

        report["validation_passed"] = (
            report["data_integrity_ok"] and
            report["class_balance_ok"] and
            report["image_quality_ok"]
        )

        return report

    def _validate_split(
        self,
        data_loader: DataLoader,
        split_name: str
    ) -> Dict[str, Any]:
        """Valide un split de données."""
        report = {
            "split_name": split_name,
            "num_samples": 0,
            "integrity_ok": True,
            "class_balance_ok": True,
            "image_quality_ok": True,
            "issues": [],
            "class_distribution": {},
            "image_stats": {}
        }

        all_labels = []
        all_means = []
        all_stds = []
        nan_count = 0
        inf_count = 0
        wrong_size_count = 0

        for images, labels in data_loader:
            batch_size = images.size(0)
            report["num_samples"] += batch_size

            # Vérifier l'intégrité
            nan_mask = torch.isnan(images)
            inf_mask = torch.isinf(images)

            nan_count += nan_mask.sum().item()
            inf_count += inf_mask.sum().item()

            # Vérifier la taille
            expected = (batch_size,) + self.expected_image_size
            if images.shape != expected:
                wrong_size_count += batch_size

            # Statistiques
            all_means.append(images.mean().item())
            all_stds.append(images.std().item())

            # Labels
            all_labels.extend(labels.numpy().tolist())

        # Analyser l'intégrité
        if nan_count > 0:
            report["integrity_ok"] = False
            report["issues"].append(f"{nan_count} NaN values in {split_name}")

        if inf_count > 0:
            report["integrity_ok"] = False
            report["issues"].append(f"{inf_count} Inf values in {split_name}")

        if wrong_size_count > 0:
            report["image_quality_ok"] = False
            report["issues"].append(f"{wrong_size_count} wrong-sized images in {split_name}")

        # Analyser l'équilibre des classes
        class_counts = Counter(all_labels)
        report["class_distribution"] = dict(class_counts)

        if len(class_counts) > 0:
            mean_count = np.mean(list(class_counts.values()))
            for cls, count in class_counts.items():
                deviation = abs(count - mean_count) / mean_count
                if deviation > self.class_balance_threshold:
                    report["class_balance_ok"] = False
                    break

        # Statistiques des images
        if all_means:
            report["image_stats"] = {
                "mean": float(np.mean(all_means)),
                "std": float(np.mean(all_stds)),
                "min_mean": float(min(all_means)),
                "max_mean": float(max(all_means))
            }

        return report

    def validate_single_image(
        self,
        image: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Valide une seule image.

        Args:
            image: Tensor de l'image

        Returns:
            Résultat de validation
        """
        result = {
            "valid": True,
            "issues": []
        }

        # Vérifier les dimensions
        if len(image.shape) not in [3, 4]:
            result["valid"] = False
            result["issues"].append(f"Invalid dimensions: {image.shape}")
            return result

        # Ajouter dimension batch si nécessaire
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # Vérifier NaN/Inf
        if torch.isnan(image).any():
            result["valid"] = False
            result["issues"].append("Image contains NaN values")

        if torch.isinf(image).any():
            result["valid"] = False
            result["issues"].append("Image contains Inf values")

        # Vérifier les valeurs
        if image.min() < -10 or image.max() > 10:
            result["issues"].append("Image values outside expected range")

        return result

    def check_data_leakage(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Vérifie s'il y a des fuites de données entre train et test.

        Args:
            train_loader: DataLoader d'entraînement
            test_loader: DataLoader de test
            sample_size: Nombre d'échantillons à comparer

        Returns:
            Rapport de détection de fuites
        """
        train_hashes = set()
        test_hashes = set()

        # Collecter les hash des images d'entraînement
        count = 0
        for images, _ in train_loader:
            for img in images:
                if count >= sample_size:
                    break
                img_hash = hash(img.numpy().tobytes())
                train_hashes.add(img_hash)
                count += 1
            if count >= sample_size:
                break

        # Vérifier les images de test
        count = 0
        duplicates = 0
        for images, _ in test_loader:
            for img in images:
                if count >= sample_size:
                    break
                img_hash = hash(img.numpy().tobytes())
                if img_hash in train_hashes:
                    duplicates += 1
                test_hashes.add(img_hash)
                count += 1
            if count >= sample_size:
                break

        leakage_ratio = duplicates / len(test_hashes) if test_hashes else 0

        return {
            "leakage_detected": duplicates > 0,
            "duplicate_count": duplicates,
            "leakage_ratio": leakage_ratio,
            "train_samples_checked": len(train_hashes),
            "test_samples_checked": len(test_hashes),
            "recommendation": "Data leakage found! Review data splits." if duplicates > 0 else "No leakage detected."
        }

    def generate_validation_report(
        self,
        validation_result: Dict[str, Any]
    ) -> str:
        """Génère un rapport de validation au format Markdown."""
        report = f"""# Data Validation Report
Generated: {validation_result.get('timestamp', datetime.now().isoformat())}

## Dataset Overview
| Split | Samples |
|-------|---------|
| Training | {validation_result.get('train_samples', 'N/A')} |
| Validation | {validation_result.get('val_samples', 'N/A')} |
| Test | {validation_result.get('test_samples', 'N/A')} |

## Validation Results
| Check | Status |
|-------|--------|
| Data Integrity | {'✅ PASSED' if validation_result.get('data_integrity_ok', False) else '❌ FAILED'} |
| Class Balance | {'✅ PASSED' if validation_result.get('class_balance_ok', False) else '⚠️ WARNING'} |
| Image Quality | {'✅ PASSED' if validation_result.get('image_quality_ok', False) else '❌ FAILED'} |

## Overall Status: {'✅ PASSED' if validation_result.get('validation_passed', False) else '❌ FAILED'}

"""
        if validation_result.get('issues'):
            report += "## Issues Found\n"
            for issue in validation_result['issues']:
                report += f"- ⚠️ {issue}\n"
        else:
            report += "## No Issues Found\n"

        return report


class SchemaValidator:
    """
    Validateur de schéma pour les données d'entrée de l'API.
    """

    def __init__(self):
        self.expected_schema = {
            "image": {
                "type": "tensor",
                "shape": [3, 32, 32],
                "dtype": "float32",
                "range": [-3.0, 3.0]  # Après normalisation
            }
        }

    def validate_input(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valide les données d'entrée selon le schéma."""
        errors = []

        if "image" not in data:
            errors.append("Missing 'image' field")
            return False, errors

        image = data["image"]

        if not isinstance(image, (torch.Tensor, np.ndarray)):
            errors.append("Image must be a tensor or numpy array")
            return False, errors

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        # Vérifier la forme
        if len(image.shape) == 3:
            if list(image.shape) != self.expected_schema["image"]["shape"]:
                errors.append(f"Expected shape {self.expected_schema['image']['shape']}, got {list(image.shape)}")
        elif len(image.shape) == 4:
            if list(image.shape[1:]) != self.expected_schema["image"]["shape"]:
                errors.append(f"Expected shape [batch, 3, 32, 32], got {list(image.shape)}")
        else:
            errors.append(f"Invalid number of dimensions: {len(image.shape)}")

        # Vérifier le type
        if image.dtype != torch.float32:
            errors.append(f"Expected dtype float32, got {image.dtype}")

        # Vérifier la plage
        min_val, max_val = self.expected_schema["image"]["range"]
        if image.min() < min_val or image.max() > max_val:
            errors.append(f"Values outside expected range [{min_val}, {max_val}]")

        return len(errors) == 0, errors
