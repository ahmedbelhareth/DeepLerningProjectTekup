"""
Tests unitaires pour le module de données.

Ce module teste le chargement et le prétraitement des données CIFAR-10.
"""

import pytest
import torch
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, '..')

from src.data.dataset import (
    load_cifar10_dataset,
    create_data_loaders,
    get_train_transforms,
    get_test_transforms,
    get_inference_transforms
)
from src.data.preprocessing import (
    preprocess_single_image,
    mixup_data,
    denormalize
)
from src.utils.config import BATCH_SIZE, NUM_CLASSES, IMAGE_SIZE


class TestDataset:
    """Tests pour le chargement du dataset."""
    
    def test_load_cifar10_dataset(self):
        """Teste le chargement du dataset CIFAR-10."""
        train_ds, test_ds = load_cifar10_dataset(download=True)
        
        assert len(train_ds) == 50000, "Le dataset d'entraînement doit avoir 50000 images"
        assert len(test_ds) == 10000, "Le dataset de test doit avoir 10000 images"
    
    def test_dataset_shape(self):
        """Teste la forme des données."""
        train_ds, _ = load_cifar10_dataset(download=True)
        
        image, label = train_ds[0]
        
        assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE), \
            f"Forme attendue: (3, {IMAGE_SIZE}, {IMAGE_SIZE}), obtenue: {image.shape}"
        assert 0 <= label < NUM_CLASSES, \
            f"Label doit être entre 0 et {NUM_CLASSES-1}"
    
    def test_create_data_loaders(self):
        """Teste la création des DataLoaders."""
        train_ds, test_ds = load_cifar10_dataset(download=True)
        train_loader, val_loader, test_loader = create_data_loaders(
            train_ds, test_ds, batch_size=32
        )
        
        # Vérifier qu'on peut itérer
        batch = next(iter(train_loader))
        images, labels = batch
        
        assert images.shape[0] == 32, "Batch size doit être 32"
        assert labels.shape[0] == 32, "Nombre de labels doit correspondre"


class TestTransforms:
    """Tests pour les transformations."""
    
    def test_train_transforms(self):
        """Teste les transformations d'entraînement."""
        transform = get_train_transforms()
        
        # Créer une image de test
        image = Image.fromarray(
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        )
        
        transformed = transform(image)
        
        assert transformed.shape == (3, 32, 32), \
            "La transformation doit produire une image 3x32x32"
        assert isinstance(transformed, torch.Tensor), \
            "La sortie doit être un tensor PyTorch"
    
    def test_inference_transforms(self):
        """Teste les transformations d'inférence."""
        transform = get_inference_transforms()
        
        # Image de taille différente
        image = Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        
        transformed = transform(image)
        
        assert transformed.shape == (3, 32, 32), \
            "L'image doit être redimensionnée en 32x32"


class TestPreprocessing:
    """Tests pour le prétraitement."""
    
    def test_preprocess_single_image(self):
        """Teste le prétraitement d'une image unique."""
        image = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        
        tensor = preprocess_single_image(image)
        
        assert tensor.shape == (1, 3, 32, 32), \
            "La sortie doit avoir une dimension batch"
    
    def test_mixup_data(self):
        """Teste l'augmentation MixUp."""
        x = torch.randn(8, 3, 32, 32)
        y = torch.randint(0, 10, (8,))
        
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
        
        assert mixed_x.shape == x.shape, "La forme doit être préservée"
        assert 0 <= lam <= 1, "Lambda doit être entre 0 et 1"
    
    def test_denormalize(self):
        """Teste la dénormalisation."""
        # Créer un tensor normalisé
        tensor = torch.randn(3, 32, 32)
        
        denorm = denormalize(tensor)
        
        assert denorm.shape == tensor.shape, "La forme doit être préservée"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
