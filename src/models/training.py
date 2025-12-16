"""
Module d'entraînement du modèle.

Ce module contient la boucle d'entraînement, la validation,
et l'intégration avec MLflow pour le tracking des expériences.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from pathlib import Path
import time

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.utils.config import (
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_SEED
)
from src.utils.metrics import compute_metrics


class EarlyStopping:
    """
    Implémentation de l'early stopping pour éviter le surapprentissage.
    """
    
    def __init__(
        self,
        patience: int = EARLY_STOPPING_PATIENCE,
        min_delta: float = EARLY_STOPPING_MIN_DELTA,
        mode: str = 'min'
    ):
        """
        Initialise l'early stopping.
        
        Args:
            patience: Nombre d'époques sans amélioration avant arrêt
            min_delta: Amélioration minimale considérée comme significative
            mode: 'min' pour loss, 'max' pour accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Vérifie si l'entraînement doit s'arrêter.
        
        Args:
            score: Score actuel (loss ou accuracy)
            epoch: Époque actuelle
            
        Returns:
            True si l'entraînement doit s'arrêter
        """
        if self.mode == 'min':
            score = -score
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        
        return self.early_stop


def set_seed(seed: int = RANDOM_SEED):
    """Fixe les seeds pour la reproductibilité."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Entraîne le modèle pour une époque.
    
    Args:
        model: Modèle à entraîner
        train_loader: DataLoader d'entraînement
        criterion: Fonction de perte
        optimizer: Optimiseur
        device: Device (CPU/GPU)
        epoch: Numéro de l'époque
        
    Returns:
        Tuple (loss moyenne, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Époque {epoch+1} [Train]")
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Mise à jour de la barre de progression
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Tuple[float, float, Dict]:
    """
    Évalue le modèle sur l'ensemble de validation.
    
    Args:
        model: Modèle à évaluer
        val_loader: DataLoader de validation
        criterion: Fonction de perte
        device: Device (CPU/GPU)
        epoch: Numéro de l'époque
        
    Returns:
        Tuple (loss moyenne, accuracy, métriques détaillées)
    """
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Époque {epoch+1} [Valid]")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_loss = running_loss / len(val_loader)
    
    # Calcul des métriques détaillées
    metrics = compute_metrics(
        np.array(all_targets),
        np.array(all_predictions)
    )
    
    return avg_loss, metrics['accuracy'], metrics


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    use_mlflow: bool = True
) -> Dict:
    """
    Fonction principale d'entraînement.
    
    Args:
        model: Modèle à entraîner
        train_loader: DataLoader d'entraînement
        val_loader: DataLoader de validation
        device: Device (CPU/GPU)
        num_epochs: Nombre d'époques
        learning_rate: Taux d'apprentissage
        weight_decay: Régularisation L2
        use_mlflow: Activer le tracking MLflow
        
    Returns:
        Dictionnaire avec l'historique d'entraînement
    """
    # Fixer le seed
    set_seed()
    
    # Déplacer le modèle sur le device
    model = model.to(device)
    
    # Critère et optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode='min')
    
    # Historique
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    best_model_path = MODELS_DIR / "best_model.pth"
    
    # Configuration MLflow
    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        mlflow.start_run()
        
        # Logger les hyperparamètres
        mlflow.log_params({
            'model_name': model.model_name,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': train_loader.batch_size,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR'
        })
    
    print(f"\n{'='*60}")
    print(f"Début de l'entraînement sur {device}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        for epoch in range(num_epochs):
            # Entraînement
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # Validation
            val_loss, val_acc, val_metrics = validate(
                model, val_loader, criterion, device, epoch
            )
            
            # Mise à jour du scheduler
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Sauvegarder l'historique
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rates'].append(current_lr)
            
            # Afficher les résultats
            print(f"\nÉpoque {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"  F1-Score: {val_metrics['f1_macro']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Logger dans MLflow
            if use_mlflow:
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'f1_macro': val_metrics['f1_macro'],
                    'learning_rate': current_lr
                }, step=epoch)
            
            # Sauvegarder le meilleur modèle
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, best_model_path)
                print(f"  ✓ Meilleur modèle sauvegardé (acc: {val_acc:.2f}%)")
            
            # Vérifier l'early stopping
            if early_stopping(val_loss, epoch):
                print(f"\n⚠ Early stopping à l'époque {epoch+1}")
                break
    
    finally:
        total_time = time.time() - start_time
        
        if use_mlflow:
            # Logger le modèle final
            mlflow.pytorch.log_model(model, "model")
            mlflow.log_metric('best_val_acc', best_val_acc)
            mlflow.log_metric('training_time_seconds', total_time)
            mlflow.end_run()
    
    print(f"\n{'='*60}")
    print(f"Entraînement terminé en {total_time/60:.2f} minutes")
    print(f"Meilleure accuracy de validation: {best_val_acc:.2f}%")
    print(f"{'='*60}\n")
    
    return history


if __name__ == "__main__":
    # Test du module
    from data.dataset import load_cifar10_dataset, create_data_loaders
    from models.architecture import create_model
    from utils.config import get_device
    
    print("Test du module d'entraînement...")
    
    # Charger les données
    train_ds, test_ds = load_cifar10_dataset()
    train_loader, val_loader, test_loader = create_data_loaders(
        train_ds, test_ds, batch_size=64
    )
    
    # Créer le modèle
    device = get_device()
    model = create_model()
    
    # Entraîner (quelques époques pour le test)
    history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=2,
        use_mlflow=False  # Désactiver MLflow pour le test
    )
    
    print("Test terminé avec succès!")
