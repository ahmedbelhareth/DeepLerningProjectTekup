"""
Configuration globale du projet de classification CIFAR-10.

Ce module centralise tous les hyperparamètres et chemins
pour faciliter la reproductibilité des expériences.
"""

import os
from pathlib import Path

# =============================================================================
# CHEMINS DU PROJET
# =============================================================================

# Répertoire racine du projet
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Chemins des données
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Chemins des modèles
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Chemins MLflow
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")

# =============================================================================
# CONFIGURATION DU DATASET
# =============================================================================

# Classes CIFAR-10 (en français)
CIFAR10_CLASSES = [
    "Avion",
    "Automobile", 
    "Oiseau",
    "Chat",
    "Cerf",
    "Chien",
    "Grenouille",
    "Cheval",
    "Navire",
    "Camion"
]

# Classes CIFAR-10 (en anglais - pour correspondance)
CIFAR10_CLASSES_EN = [
    "airplane",
    "automobile",
    "bird", 
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

NUM_CLASSES = 10

# Dimensions des images
IMAGE_SIZE = 32
IMAGE_CHANNELS = 3

# =============================================================================
# HYPERPARAMÈTRES D'ENTRAÎNEMENT
# =============================================================================

# Paramètres de base
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Optimisation
OPTIMIZER = "adam"  # Options: "adam", "sgd", "adamw"
SCHEDULER = "cosine"  # Options: "cosine", "step", "plateau"

# Early stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

# =============================================================================
# CONFIGURATION DU MODÈLE
# =============================================================================

# Architecture
MODEL_NAME = "resnet18"  # Options: "resnet18", "resnet34", "efficientnet_b0"
PRETRAINED = True
FREEZE_BACKBONE = False

# Dropout
DROPOUT_RATE = 0.2

# =============================================================================
# AUGMENTATION DES DONNÉES
# =============================================================================

# Augmentations d'entraînement
TRAIN_AUGMENTATION = {
    "horizontal_flip": True,
    "random_crop": True,
    "color_jitter": True,
    "cutout": True,
    "normalize": True
}

# Normalisation (statistiques ImageNet)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Normalisation CIFAR-10 spécifique
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# =============================================================================
# CONFIGURATION DE L'API
# =============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "API Classification CIFAR-10"
API_DESCRIPTION = "API de classification d'images utilisant ResNet sur CIFAR-10"
API_VERSION = "1.0.0"

# =============================================================================
# CONFIGURATION STREAMLIT
# =============================================================================

STREAMLIT_TITLE = "🖼️ Classification d'Images CIFAR-10"
STREAMLIT_ICON = "🖼️"

# =============================================================================
# CONFIGURATION MLFLOW
# =============================================================================

EXPERIMENT_NAME = "CIFAR10_Classification"
RUN_NAME_PREFIX = "resnet18_run"

# =============================================================================
# SEEDS POUR REPRODUCTIBILITÉ
# =============================================================================

RANDOM_SEED = 42

# =============================================================================
# CONFIGURATION GPU
# =============================================================================

# Utiliser GPU si disponible
USE_GPU = True
DEVICE = "cuda" if USE_GPU else "cpu"

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def create_directories():
    """Crée les répertoires nécessaires s'ils n'existent pas."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        CHECKPOINTS_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_device():
    """Retourne le device approprié (GPU ou CPU)."""
    import torch
    if USE_GPU and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Créer les répertoires au chargement du module
create_directories()
