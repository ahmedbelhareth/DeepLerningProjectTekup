"""
Module de gestion du dataset CIFAR-10.

Ce module gère le téléchargement, le chargement et la préparation
des données CIFAR-10 pour l'entraînement et l'évaluation.
"""

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
import numpy as np

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.utils.config import (
    BATCH_SIZE,
    CIFAR10_MEAN,
    CIFAR10_STD,
    RANDOM_SEED,
    DATA_DIR
)


def get_train_transforms() -> transforms.Compose:
    """
    Retourne les transformations pour l'ensemble d'entraînement.
    
    Inclut des augmentations de données pour améliorer la généralisation.
    
    Returns:
        transforms.Compose: Pipeline de transformations
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        transforms.RandomErasing(p=0.1)  # Cutout simplifié
    ])


def get_test_transforms() -> transforms.Compose:
    """
    Retourne les transformations pour l'ensemble de test/validation.
    
    Pas d'augmentation, seulement normalisation.
    
    Returns:
        transforms.Compose: Pipeline de transformations
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])


def get_inference_transforms() -> transforms.Compose:
    """
    Retourne les transformations pour l'inférence.
    
    Redimensionne l'image si nécessaire et normalise.
    
    Returns:
        transforms.Compose: Pipeline de transformations
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])


def load_cifar10_dataset(
    download: bool = True
) -> Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
    """
    Télécharge et charge le dataset CIFAR-10.
    
    Args:
        download: Si True, télécharge le dataset s'il n'existe pas
        
    Returns:
        Tuple contenant les datasets d'entraînement et de test
    """
    # Dataset d'entraînement avec augmentations
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=download,
        transform=get_train_transforms()
    )
    
    # Dataset de test sans augmentations
    test_dataset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=download,
        transform=get_test_transforms()
    )
    
    return train_dataset, test_dataset


def create_data_loaders(
    train_dataset: torchvision.datasets.CIFAR10,
    test_dataset: torchvision.datasets.CIFAR10,
    batch_size: int = BATCH_SIZE,
    val_split: float = 0.1,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crée les DataLoaders pour l'entraînement, la validation et le test.
    
    Args:
        train_dataset: Dataset d'entraînement
        test_dataset: Dataset de test
        batch_size: Taille des batches
        val_split: Proportion pour la validation
        num_workers: Nombre de workers pour le chargement
        
    Returns:
        Tuple de DataLoaders (train, val, test)
    """
    # Diviser le dataset d'entraînement en train/val
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    # Fixer le seed pour la reproductibilité
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    
    train_subset, val_subset = random_split(
        train_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_class_distribution(dataset: torchvision.datasets.CIFAR10) -> dict:
    """
    Calcule la distribution des classes dans le dataset.
    
    Args:
        dataset: Dataset CIFAR-10
        
    Returns:
        Dictionnaire avec le nombre d'échantillons par classe
    """
    from utils.config import CIFAR10_CLASSES
    
    targets = np.array(dataset.targets)
    distribution = {}
    
    for idx, class_name in enumerate(CIFAR10_CLASSES):
        count = np.sum(targets == idx)
        distribution[class_name] = int(count)
    
    return distribution


def get_sample_images(
    dataset: torchvision.datasets.CIFAR10, 
    num_samples: int = 10
) -> Tuple[torch.Tensor, list]:
    """
    Récupère des échantillons d'images du dataset.
    
    Args:
        dataset: Dataset CIFAR-10
        num_samples: Nombre d'échantillons à récupérer
        
    Returns:
        Tuple (images, labels)
    """
    from utils.config import CIFAR10_CLASSES
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    images = []
    labels = []
    
    for idx in indices:
        img, label = dataset[idx]
        images.append(img)
        labels.append(CIFAR10_CLASSES[label])
    
    return torch.stack(images), labels


if __name__ == "__main__":
    # Test du module
    print("Chargement du dataset CIFAR-10...")
    train_ds, test_ds = load_cifar10_dataset()
    
    print(f"Taille du dataset d'entraînement: {len(train_ds)}")
    print(f"Taille du dataset de test: {len(test_ds)}")
    
    print("\nCréation des DataLoaders...")
    train_loader, val_loader, test_loader = create_data_loaders(train_ds, test_ds)
    
    print(f"Batches d'entraînement: {len(train_loader)}")
    print(f"Batches de validation: {len(val_loader)}")
    print(f"Batches de test: {len(test_loader)}")
    
    print("\nDistribution des classes (entraînement):")
    distribution = get_class_distribution(train_ds)
    for class_name, count in distribution.items():
        print(f"  {class_name}: {count}")
