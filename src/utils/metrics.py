"""
Module de calcul des métriques d'évaluation.

Ce module contient les fonctions pour calculer et visualiser
les métriques de performance du modèle.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.utils.config import CIFAR10_CLASSES, NUM_CLASSES


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict:
    """
    Calcule toutes les métriques de classification.
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        y_proba: Probabilités prédites (optionnel, pour Top-5)
        
    Returns:
        Dictionnaire contenant toutes les métriques
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred) * 100
    
    # Precision, Recall, F1 (macro et micro)
    metrics['precision_macro'] = precision_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['precision_micro'] = precision_score(
        y_true, y_pred, average='micro', zero_division=0
    )
    metrics['recall_macro'] = recall_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['recall_micro'] = recall_score(
        y_true, y_pred, average='micro', zero_division=0
    )
    metrics['f1_macro'] = f1_score(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['f1_micro'] = f1_score(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    # Top-5 accuracy si probabilités disponibles
    if y_proba is not None and y_proba.shape[1] >= 5:
        metrics['top5_accuracy'] = top_k_accuracy_score(
            y_true, y_proba, k=5
        ) * 100
    
    # Métriques par classe
    metrics['precision_per_class'] = precision_score(
        y_true, y_pred, average=None, zero_division=0
    ).tolist()
    metrics['recall_per_class'] = recall_score(
        y_true, y_pred, average=None, zero_division=0
    ).tolist()
    metrics['f1_per_class'] = f1_score(
        y_true, y_pred, average=None, zero_division=0
    ).tolist()
    
    # Matrice de confusion
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CIFAR10_CLASSES
) -> str:
    """
    Génère un rapport de classification détaillé.
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        class_names: Noms des classes
        
    Returns:
        Rapport de classification formaté
    """
    return classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        zero_division=0
    )


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CIFAR10_CLASSES,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualise la matrice de confusion.
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        class_names: Noms des classes
        figsize: Taille de la figure
        save_path: Chemin pour sauvegarder (optionnel)
        
    Returns:
        Figure matplotlib
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normaliser pour avoir des pourcentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Matrice de confusion brute
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0]
    )
    axes[0].set_title('Matrice de Confusion (Nombres)')
    axes[0].set_xlabel('Prédiction')
    axes[0].set_ylabel('Réalité')
    
    # Matrice de confusion normalisée
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1]
    )
    axes[1].set_title('Matrice de Confusion (Normalisée)')
    axes[1].set_xlabel('Prédiction')
    axes[1].set_ylabel('Réalité')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualise l'historique d'entraînement.
    
    Args:
        history: Dictionnaire d'historique
        figsize: Taille de la figure
        save_path: Chemin pour sauvegarder (optionnel)
        
    Returns:
        Figure matplotlib
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0].set_title('Évolution de la Loss')
    axes[0].set_xlabel('Époque')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation')
    axes[1].set_title('Évolution de l\'Accuracy')
    axes[1].set_xlabel('Époque')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2].plot(epochs, history['learning_rates'], 'g-')
    axes[2].set_title('Évolution du Learning Rate')
    axes[2].set_xlabel('Époque')
    axes[2].set_ylabel('Learning Rate')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_per_class_metrics(
    metrics: Dict,
    class_names: List[str] = CIFAR10_CLASSES,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualise les métriques par classe.
    
    Args:
        metrics: Dictionnaire de métriques
        class_names: Noms des classes
        figsize: Taille de la figure
        save_path: Chemin pour sauvegarder (optionnel)
        
    Returns:
        Figure matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    precision = metrics['precision_per_class']
    recall = metrics['recall_per_class']
    f1 = metrics['f1_per_class']
    
    bars1 = ax.bar(x - width, precision, width, label='Précision', color='steelblue')
    bars2 = ax.bar(x, recall, width, label='Rappel', color='darkorange')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='forestgreen')
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Métriques par Classe')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: List[str] = CIFAR10_CLASSES
) -> Tuple[Dict, str]:
    """
    Évalue complètement le modèle sur le dataset de test.
    
    Args:
        model: Modèle à évaluer
        test_loader: DataLoader de test
        device: Device (CPU/GPU)
        class_names: Noms des classes
        
    Returns:
        Tuple (métriques, rapport de classification)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probas = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            probas = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probas.extend(probas.cpu().numpy())
    
    y_true = np.array(all_targets)
    y_pred = np.array(all_predictions)
    y_proba = np.array(all_probas)
    
    # Calculer les métriques
    metrics = compute_metrics(y_true, y_pred, y_proba)
    
    # Générer le rapport
    report = get_classification_report(y_true, y_pred, class_names)
    
    return metrics, report


if __name__ == "__main__":
    # Test du module
    print("Test du module de métriques...")
    
    # Données synthétiques
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.randint(0, NUM_CLASSES, n_samples)
    y_pred = np.random.randint(0, NUM_CLASSES, n_samples)
    y_proba = np.random.rand(n_samples, NUM_CLASSES)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    # Calculer les métriques
    metrics = compute_metrics(y_true, y_pred, y_proba)
    
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
    if 'top5_accuracy' in metrics:
        print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
    
    # Afficher le rapport
    print("\nRapport de classification:")
    print(get_classification_report(y_true, y_pred))
    
    print("\nTest terminé avec succès!")
