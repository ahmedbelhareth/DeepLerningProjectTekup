"""
API FastAPI pour la classification d'images CIFAR-10.

Ce module expose les endpoints REST pour l'inférence
et les informations sur le modèle.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict
import io

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import (
    API_TITLE,
    API_DESCRIPTION,
    API_VERSION,
    API_HOST,
    API_PORT,
    CIFAR10_CLASSES
)
from src.api.endpoints import router as api_router

# Créer l'application FastAPI
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS pour permettre les requêtes cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les origines autorisées
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure les routes
app.include_router(api_router, prefix="/api/v1", tags=["Classification"])


@app.get("/", tags=["Root"])
async def root():
    """
    Point d'entrée racine de l'API.
    
    Returns:
        Message de bienvenue et liens utiles
    """
    return {
        "message": "Bienvenue sur l'API de Classification CIFAR-10",
        "version": API_VERSION,
        "documentation": "/docs",
        "endpoints": {
            "predict": "/api/v1/predict",
            "batch_predict": "/api/v1/batch_predict",
            "model_info": "/api/v1/model/info",
            "classes": "/api/v1/classes",
            "health": "/api/v1/health"
        }
    }


@app.get("/api/v1/health", tags=["Système"])
async def health_check():
    """
    Vérifie l'état de santé de l'API.
    
    Returns:
        Statut de l'API et informations système
    """
    import torch
    
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "api_version": API_VERSION
    }


@app.get("/api/v1/classes", tags=["Informations"])
async def get_classes():
    """
    Retourne la liste des classes CIFAR-10.
    
    Returns:
        Liste des classes avec leurs indices
    """
    return {
        "num_classes": len(CIFAR10_CLASSES),
        "classes": {i: name for i, name in enumerate(CIFAR10_CLASSES)}
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Gestionnaire global des exceptions.
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erreur interne du serveur",
            "detail": str(exc)
        }
    )


def start_server():
    """Lance le serveur FastAPI."""
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )


if __name__ == "__main__":
    start_server()
