"""
Tests unitaires pour l'API FastAPI.

Ce module teste les endpoints de l'API de classification.
"""

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import numpy as np
import base64

import sys
sys.path.insert(0, '..')

from src.api.main import app


# Client de test
client = TestClient(app)


class TestRootEndpoints:
    """Tests pour les endpoints racine."""
    
    def test_root(self):
        """Teste le point d'entrée racine."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_health_check(self):
        """Teste le endpoint de santé."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "gpu_available" in data
    
    def test_get_classes(self):
        """Teste la récupération des classes."""
        response = client.get("/api/v1/classes")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["num_classes"] == 10
        assert len(data["classes"]) == 10


class TestPredictionEndpoints:
    """Tests pour les endpoints de prédiction."""
    
    @staticmethod
    def create_test_image():
        """Crée une image de test."""
        img = Image.fromarray(
            np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        )
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf
    
    def test_predict_valid_image(self):
        """Teste la prédiction avec une image valide."""
        image_buf = self.create_test_image()
        
        response = client.post(
            "/api/v1/predict",
            files={"file": ("test.png", image_buf, "image/png")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "class_id" in data
        assert "class_name" in data
        assert "confidence" in data
        assert "probabilities" in data
        
        assert 0 <= data["class_id"] < 10
        assert 0 <= data["confidence"] <= 1
    
    def test_predict_invalid_file_type(self):
        """Teste la prédiction avec un type de fichier invalide."""
        response = client.post(
            "/api/v1/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        
        assert response.status_code == 400
    
    def test_batch_predict(self):
        """Teste la prédiction par lot."""
        images = [self.create_test_image() for _ in range(3)]
        
        files = [
            ("files", (f"test{i}.png", img, "image/png"))
            for i, img in enumerate(images)
        ]
        
        response = client.post("/api/v1/batch_predict", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "predictions" in data
        assert len(data["predictions"]) == 3
    
    def test_predict_base64(self):
        """Teste la prédiction avec image base64."""
        image_buf = self.create_test_image()
        image_b64 = base64.b64encode(image_buf.read()).decode('utf-8')
        
        response = client.post(
            "/api/v1/predict_base64",
            json={"image": image_b64}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "class_id" in data
        assert "class_name" in data


class TestModelInfoEndpoint:
    """Tests pour les informations du modèle."""
    
    def test_model_info(self):
        """Teste la récupération des informations du modèle."""
        response = client.get("/api/v1/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model_name" in data
        assert "num_classes" in data
        assert "total_parameters" in data
        assert "device" in data
        
        assert data["num_classes"] == 10
        assert data["total_parameters"] > 0


class TestErrorHandling:
    """Tests pour la gestion des erreurs."""
    
    def test_missing_file(self):
        """Teste l'erreur quand le fichier est manquant."""
        response = client.post("/api/v1/predict")
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_endpoint(self):
        """Teste un endpoint inexistant."""
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
