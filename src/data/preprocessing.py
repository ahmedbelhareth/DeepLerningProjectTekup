"""
Module de prétraitement et augmentation des données.

Ce module contient les fonctions et classes pour le prétraitement
des images et les techniques d'augmentation avancées.
"""

import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional, Tuple, Callable
import torchvision.transforms as transforms

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.utils.config import CIFAR10_MEAN, CIFAR10_STD, IMAGE_SIZE


class AlbumentationsTransform:
    """
    Wrapper pour utiliser Albumentations avec PyTorch datasets.
    
    Albumentations offre des augmentations plus rapides et plus variées
    que torchvision.transforms.
    """
    
    def __init__(self, transform: A.Compose):
        """
        Initialise le wrapper.
        
        Args:
            transform: Pipeline de transformations Albumentations
        """
        self.transform = transform
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Applique les transformations à une image PIL.
        
        Args:
            image: Image PIL
            
        Returns:
            Tensor PyTorch transformé
        """
        # Convertir PIL en numpy
        image_np = np.array(image)
        
        # Appliquer les transformations
        augmented = self.transform(image=image_np)
        
        return augmented['image']


def get_albumentations_train_transform() -> AlbumentationsTransform:
    """
    Retourne les augmentations d'entraînement avec Albumentations.
    
    Returns:
        AlbumentationsTransform: Pipeline d'augmentation
    """
    transform = A.Compose([
        # Augmentations géométriques
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        
        # Augmentations de couleur
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        
        # Cutout / CoarseDropout
        A.CoarseDropout(
            max_holes=1,
            max_height=8,
            max_width=8,
            min_holes=1,
            min_height=4,
            min_width=4,
            fill_value=0,
            p=0.3
        ),
        
        # Normalisation et conversion en tensor
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2()
    ])
    
    return AlbumentationsTransform(transform)


def get_albumentations_test_transform() -> AlbumentationsTransform:
    """
    Retourne les transformations de test avec Albumentations.
    
    Returns:
        AlbumentationsTransform: Pipeline de transformation
    """
    transform = A.Compose([
        A.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ToTensorV2()
    ])
    
    return AlbumentationsTransform(transform)


def mixup_data(
    x: torch.Tensor, 
    y: torch.Tensor, 
    alpha: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applique l'augmentation MixUp aux données.
    
    MixUp mélange deux images et leurs labels de manière linéaire.
    
    Args:
        x: Batch d'images
        y: Batch de labels
        alpha: Paramètre de la distribution Beta
        
    Returns:
        Tuple (images_mixées, labels_a, labels_b, lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(
    x: torch.Tensor, 
    y: torch.Tensor, 
    alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applique l'augmentation CutMix aux données.
    
    CutMix découpe une partie d'une image et la remplace par
    une partie d'une autre image.
    
    Args:
        x: Batch d'images
        y: Batch de labels
        alpha: Paramètre de la distribution Beta
        
    Returns:
        Tuple (images_mixées, labels_a, labels_b, lambda)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    # Calculer les coordonnées du rectangle à découper
    W = x.size(2)
    H = x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Position centrale du rectangle
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Coordonnées bornées
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Appliquer CutMix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Ajuster lambda en fonction de la surface réelle
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam


def denormalize(
    tensor: torch.Tensor,
    mean: list = CIFAR10_MEAN,
    std: list = CIFAR10_STD
) -> torch.Tensor:
    """
    Dénormalise un tensor pour l'affichage.
    
    Args:
        tensor: Tensor normalisé
        mean: Moyenne utilisée pour la normalisation
        std: Écart-type utilisé pour la normalisation
        
    Returns:
        Tensor dénormalisé
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    return tensor * std + mean


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convertit un tensor PyTorch en image PIL.
    
    Args:
        tensor: Tensor de forme (C, H, W)
        
    Returns:
        Image PIL
    """
    # Dénormaliser si nécessaire
    if tensor.min() < 0:
        tensor = denormalize(tensor)
    
    # Convertir en numpy et transposer
    numpy_image = tensor.numpy().transpose((1, 2, 0))
    
    # Clipper et convertir en uint8
    numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)
    
    return Image.fromarray(numpy_image)


def preprocess_single_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)
) -> torch.Tensor:
    """
    Prétraite une image unique pour l'inférence.
    
    Args:
        image: Image PIL
        target_size: Taille cible
        
    Returns:
        Tensor prêt pour l'inférence
    """
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])
    
    return transform(image).unsqueeze(0)  # Ajouter dimension batch


if __name__ == "__main__":
    # Test du module
    print("Test des transformations Albumentations...")
    
    # Créer une image de test
    test_image = Image.fromarray(
        np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    )
    
    # Tester les transformations
    train_transform = get_albumentations_train_transform()
    test_transform = get_albumentations_test_transform()
    
    train_tensor = train_transform(test_image)
    test_tensor = test_transform(test_image)
    
    print(f"Forme après transformation train: {train_tensor.shape}")
    print(f"Forme après transformation test: {test_tensor.shape}")
    
    # Tester MixUp
    print("\nTest MixUp...")
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    mixed_x, y_a, y_b, lam = mixup_data(x, y)
    print(f"Lambda MixUp: {lam:.3f}")
    
    print("\nTous les tests passés avec succès!")
