"""
Endpoints de l'API de classification.

Ce module définit les endpoints REST pour la prédiction
et les informations sur le modèle.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
from PIL import Image
import io
import base64
import numpy as np
from pathlib import Path

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.utils.config import (
    CIFAR10_CLASSES,
    MODELS_DIR,
    MODEL_NAME,
    NUM_CLASSES,
    get_device
)
from src.models.architecture import load_model, create_model
from src.data.preprocessing import preprocess_single_image

router = APIRouter()

# Variable globale pour le modèle (chargé au démarrage)
model = None
device = None


class PredictionResponse(BaseModel):
    """Schéma de réponse pour une prédiction."""
    class_id: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]


class BatchPredictionResponse(BaseModel):
    """Schéma de réponse pour les prédictions par lot."""
    predictions: List[PredictionResponse]


class ModelInfoResponse(BaseModel):
    """Schéma de réponse pour les informations du modèle."""
    model_name: str
    num_classes: int
    input_size: str
    total_parameters: int
    trainable_parameters: int
    device: str


def load_model_if_needed():
    """Charge le modèle si ce n'est pas déjà fait."""
    global model, device
    
    if model is None:
        device = get_device()
        model_path = MODELS_DIR / "best_model.pth"
        
        if model_path.exists():
            print(f"Chargement du modèle depuis {model_path}")
            model = load_model(
                str(model_path),
                model_name=MODEL_NAME,
                num_classes=NUM_CLASSES,
                device=str(device)
            )
        else:
            print("Modèle pré-entraîné non trouvé, création d'un nouveau modèle")
            model = create_model()
            model = model.to(device)
            model.eval()
        
        print(f"Modèle chargé sur {device}")


def process_image(image_bytes: bytes) -> torch.Tensor:
    """
    Prétraite une image pour l'inférence.
    
    Args:
        image_bytes: Bytes de l'image
        
    Returns:
        Tensor prêt pour l'inférence
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return preprocess_single_image(image)


@router.on_event("startup")
async def startup_event():
    """Événement de démarrage - charge le modèle."""
    load_model_if_needed()


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Prédit la classe d'une image.

    Args:
        file: Fichier image uploadé

    Returns:
        Prédiction avec classe et confiance
    """
    load_model_if_needed()

    # Vérifier le type de fichier
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Le fichier doit être une image"
        )
    
    try:
        # Lire et prétraiter l'image
        image_bytes = await file.read()
        input_tensor = process_image(image_bytes)
        input_tensor = input_tensor.to(device)
        
        # Inférence
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        # Préparer la réponse
        class_id = predicted.item()
        class_name = CIFAR10_CLASSES[class_id]
        confidence_value = confidence.item()
        
        # Probabilités pour toutes les classes
        probs_dict = {
            CIFAR10_CLASSES[i]: float(probabilities[0][i])
            for i in range(len(CIFAR10_CLASSES))
        }
        
        return PredictionResponse(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence_value,
            probabilities=probs_dict
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


@router.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(files: List[UploadFile] = File(...)):
    """
    Prédit les classes de plusieurs images.

    Args:
        files: Liste de fichiers images

    Returns:
        Liste de prédictions
    """
    load_model_if_needed()

    if len(files) > 32:
        raise HTTPException(
            status_code=400,
            detail="Maximum 32 images par requête"
        )
    
    predictions = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            continue
        
        try:
            image_bytes = await file.read()
            input_tensor = process_image(image_bytes)
            input_tensor = input_tensor.to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = probabilities.max(1)
            
            class_id = predicted.item()
            probs_dict = {
                CIFAR10_CLASSES[i]: float(probabilities[0][i])
                for i in range(len(CIFAR10_CLASSES))
            }
            
            predictions.append(PredictionResponse(
                class_id=class_id,
                class_name=CIFAR10_CLASSES[class_id],
                confidence=confidence.item(),
                probabilities=probs_dict
            ))
        
        except Exception as e:
            predictions.append(PredictionResponse(
                class_id=-1,
                class_name="Erreur",
                confidence=0.0,
                probabilities={}
            ))
    
    return BatchPredictionResponse(predictions=predictions)


@router.post("/predict_base64", response_model=PredictionResponse)
async def predict_base64(image_data: Dict):
    """
    Prédit la classe d'une image encodée en base64.

    Args:
        image_data: Dictionnaire avec clé 'image' contenant le base64

    Returns:
        Prédiction avec classe et confiance
    """
    load_model_if_needed()

    if 'image' not in image_data:
        raise HTTPException(
            status_code=400,
            detail="Clé 'image' manquante dans le corps de la requête"
        )
    
    try:
        # Décoder le base64
        image_bytes = base64.b64decode(image_data['image'])
        input_tensor = process_image(image_bytes)
        input_tensor = input_tensor.to(device)
        
        # Inférence
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        class_id = predicted.item()
        probs_dict = {
            CIFAR10_CLASSES[i]: float(probabilities[0][i])
            for i in range(len(CIFAR10_CLASSES))
        }
        
        return PredictionResponse(
            class_id=class_id,
            class_name=CIFAR10_CLASSES[class_id],
            confidence=confidence.item(),
            probabilities=probs_dict
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Retourne les informations sur le modèle.
    
    Returns:
        Informations détaillées sur le modèle
    """
    load_model_if_needed()
    
    total_params, trainable_params = model.count_parameters()
    
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        input_size="32x32x3",
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        device=str(device)
    )
