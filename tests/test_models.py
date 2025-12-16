"""
Tests unitaires pour le module de modèles.

Ce module teste l'architecture et l'entraînement du modèle.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.insert(0, '..')

from src.models.architecture import (
    CIFAR10ResNet,
    create_model,
    load_model
)
from src.utils.config import NUM_CLASSES, MODEL_NAME


class TestModelArchitecture:
    """Tests pour l'architecture du modèle."""
    
    def test_create_model(self):
        """Teste la création du modèle."""
        model = create_model()
        
        assert isinstance(model, CIFAR10ResNet), \
            "Le modèle doit être une instance de CIFAR10ResNet"
        assert model.num_classes == NUM_CLASSES, \
            f"Le modèle doit avoir {NUM_CLASSES} classes"
    
    def test_forward_pass(self):
        """Teste la passe avant du modèle."""
        model = create_model()
        model.eval()
        
        # Batch de test
        x = torch.randn(4, 3, 32, 32)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (4, NUM_CLASSES), \
            f"Sortie attendue: (4, {NUM_CLASSES}), obtenue: {output.shape}"
    
    def test_different_architectures(self):
        """Teste différentes architectures."""
        architectures = ["resnet18", "resnet34"]
        
        for arch in architectures:
            model = create_model(model_name=arch)
            x = torch.randn(2, 3, 32, 32)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (2, NUM_CLASSES), \
                f"Architecture {arch} échoue"
    
    def test_count_parameters(self):
        """Teste le comptage des paramètres."""
        model = create_model()
        total, trainable = model.count_parameters()
        
        assert total > 0, "Le modèle doit avoir des paramètres"
        assert trainable > 0, "Le modèle doit avoir des paramètres entraînables"
        assert trainable <= total, \
            "Les paramètres entraînables ne peuvent dépasser le total"
    
    def test_get_features(self):
        """Teste l'extraction de features."""
        model = create_model()
        model.eval()
        
        x = torch.randn(2, 3, 32, 32)
        
        with torch.no_grad():
            features = model.get_features(x)
        
        assert len(features.shape) == 2, \
            "Les features doivent être un vecteur 2D"
        assert features.shape[0] == 2, \
            "La dimension batch doit être préservée"
    
    def test_freeze_backbone(self):
        """Teste le gel du backbone."""
        model = create_model(freeze_backbone=True)
        
        # Vérifier que certains paramètres sont gelés
        frozen_count = sum(
            1 for p in model.parameters() if not p.requires_grad
        )
        
        assert frozen_count > 0, \
            "Au moins certains paramètres doivent être gelés"
    
    def test_unfreeze_backbone(self):
        """Teste le dégel du backbone."""
        model = create_model(freeze_backbone=True)
        model.unfreeze_backbone()
        
        # Vérifier que tous les paramètres sont entraînables
        for param in model.parameters():
            assert param.requires_grad, \
                "Tous les paramètres doivent être entraînables après dégel"


class TestModelPrediction:
    """Tests pour les prédictions du modèle."""
    
    def test_output_probabilities(self):
        """Teste que la sortie peut être convertie en probabilités."""
        model = create_model()
        model.eval()
        
        x = torch.randn(2, 3, 32, 32)
        
        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)
        
        # Vérifier que les probabilités somment à 1
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(2), atol=1e-5), \
            "Les probabilités doivent sommer à 1"
    
    def test_deterministic_prediction(self):
        """Teste que les prédictions sont déterministes en mode eval."""
        model = create_model()
        model.eval()
        
        x = torch.randn(2, 3, 32, 32)
        
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        assert torch.allclose(output1, output2), \
            "Les prédictions doivent être identiques en mode eval"


class TestModelTraining:
    """Tests pour l'entraînement du modèle."""
    
    def test_backward_pass(self):
        """Teste la rétropropagation."""
        model = create_model()
        model.train()
        
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, NUM_CLASSES, (4,))
        
        criterion = nn.CrossEntropyLoss()
        
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # Vérifier que les gradients existent
        has_grad = any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        assert has_grad, "Les gradients doivent être calculés"
    
    def test_optimizer_step(self):
        """Teste une étape d'optimisation."""
        model = create_model()
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, NUM_CLASSES, (4,))
        
        criterion = nn.CrossEntropyLoss()
        
        # Sauvegarder les poids initiaux
        initial_params = [p.clone() for p in model.parameters()]
        
        # Faire une étape
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # Vérifier que les poids ont changé
        params_changed = any(
            not torch.equal(p1, p2) 
            for p1, p2 in zip(initial_params, model.parameters())
        )
        assert params_changed, "Les poids doivent être mis à jour"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
