"""
Module d'architecture du modèle.

Ce module définit l'architecture ResNet adaptée pour CIFAR-10
avec support pour le transfer learning.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple
import timm

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.utils.config import (
    NUM_CLASSES,
    MODEL_NAME,
    PRETRAINED,
    DROPOUT_RATE,
    FREEZE_BACKBONE
)


class CIFAR10ResNet(nn.Module):
    """
    Modèle ResNet adapté pour la classification CIFAR-10.
    
    Utilise un ResNet pré-entraîné sur ImageNet avec une tête
    de classification personnalisée pour 10 classes.
    """
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = PRETRAINED,
        dropout_rate: float = DROPOUT_RATE,
        freeze_backbone: bool = FREEZE_BACKBONE
    ):
        """
        Initialise le modèle.
        
        Args:
            model_name: Nom de l'architecture ("resnet18", "resnet34", etc.)
            num_classes: Nombre de classes de sortie
            pretrained: Utiliser les poids pré-entraînés
            dropout_rate: Taux de dropout
            freeze_backbone: Geler les couches du backbone
        """
        super(CIFAR10ResNet, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Charger le modèle de base
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
        elif model_name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            num_features = self.backbone.fc.in_features
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
        elif model_name == "efficientnet_b0":
            self.backbone = timm.create_model(
                'efficientnet_b0', 
                pretrained=pretrained
            )
            num_features = self.backbone.classifier.in_features
        else:
            raise ValueError(f"Modèle non supporté: {model_name}")
        
        # Geler le backbone si demandé
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Remplacer la tête de classification
        if "efficientnet" in model_name:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, num_classes)
            )
        else:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, num_classes)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant du modèle.
        
        Args:
            x: Tensor d'entrée de forme (B, 3, H, W)
            
        Returns:
            Logits de forme (B, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrait les features avant la couche de classification.
        
        Utile pour la visualisation et l'analyse.
        
        Args:
            x: Tensor d'entrée de forme (B, 3, H, W)
            
        Returns:
            Features de forme (B, num_features)
        """
        # Pour ResNet
        if "resnet" in self.model_name:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            
            return x
        else:
            # Pour EfficientNet, utiliser forward_features de timm
            return self.backbone.forward_features(x)
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        Compte le nombre de paramètres du modèle.
        
        Returns:
            Tuple (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def unfreeze_backbone(self):
        """Dégèle toutes les couches du backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_backbone_layers(self, num_layers: int = 2):
        """
        Gèle les premières couches du backbone.
        
        Args:
            num_layers: Nombre de couches à geler (pour ResNet: layer1, layer2, etc.)
        """
        if "resnet" not in self.model_name:
            print("Avertissement: freeze_backbone_layers optimisé pour ResNet")
            return
        
        # Geler conv1 et bn1
        for param in self.backbone.conv1.parameters():
            param.requires_grad = False
        for param in self.backbone.bn1.parameters():
            param.requires_grad = False
        
        # Geler les couches spécifiées
        layers = [
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        ]
        
        for i in range(min(num_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = False


def create_model(
    model_name: str = MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = PRETRAINED,
    dropout_rate: float = DROPOUT_RATE,
    freeze_backbone: bool = FREEZE_BACKBONE
) -> CIFAR10ResNet:
    """
    Fonction factory pour créer un modèle.
    
    Args:
        model_name: Nom de l'architecture
        num_classes: Nombre de classes
        pretrained: Utiliser les poids pré-entraînés
        dropout_rate: Taux de dropout
        freeze_backbone: Geler le backbone
        
    Returns:
        Instance du modèle
    """
    return CIFAR10ResNet(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone
    )


def load_model(
    checkpoint_path: str,
    model_name: str = MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    device: str = "cpu"
) -> CIFAR10ResNet:
    """
    Charge un modèle depuis un checkpoint.
    
    Args:
        checkpoint_path: Chemin vers le fichier checkpoint
        model_name: Nom de l'architecture
        num_classes: Nombre de classes
        device: Device cible
        
    Returns:
        Modèle avec les poids chargés
    """
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Gérer différents formats de checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test du module
    print("Test de l'architecture du modèle...")
    
    # Créer un modèle
    model = create_model()
    
    # Afficher les informations
    total, trainable = model.count_parameters()
    print(f"Modèle: {MODEL_NAME}")
    print(f"Paramètres totaux: {total:,}")
    print(f"Paramètres entraînables: {trainable:,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    output = model(x)
    print(f"\nForme d'entrée: {x.shape}")
    print(f"Forme de sortie: {output.shape}")
    
    # Test extraction de features
    features = model.get_features(x)
    print(f"Forme des features: {features.shape}")
    
    print("\nTous les tests passés avec succès!")
