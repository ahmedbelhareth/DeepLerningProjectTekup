# Classification CIFAR-10 - Deep Learning

**Etudiant:** Ahmed Belhareth
**Module:** Deep Learning
**Enseignant:** Haythem Ghazouani

---

## Probleme

Classifier automatiquement des images en 10 categories (avion, automobile, oiseau, chat, cerf, chien, grenouille, cheval, navire, camion) a partir du dataset CIFAR-10 (60k images 32x32).

## Solution

ResNet18 pre-entraine (ImageNet) avec fine-tuning sur CIFAR-10.

| Metrique | Resultat |
|----------|----------|
| Accuracy | **87.09%** |
| F1-Score | **0.8705** |

---

## Livrables

| Exigence | Implementation |
|----------|---------------|
| Modele entraine | `models/best_model.pth` (PyTorch) |
| API REST | FastAPI avec Swagger (`/docs`) |
| Interface UI | Streamlit |
| Tracking MLOps | MLflow |
| Conteneurisation | Docker + docker-compose |
| Tests | pytest (29/29 passing) |
| Notebooks | 3 notebooks avec outputs |

---

## Quick Start

```bash
# Installation
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Entrainement
python -m src.models.training

# API (http://localhost:8000/docs)
uvicorn src.api.main:app --reload --port 8000

# Interface (http://localhost:8501)
streamlit run streamlit_app/app.py

# MLflow (http://localhost:5000)
mlflow ui --port 5000

# Tests
pytest tests/ -v

# Docker
cd docker && docker-compose up --build
```

---

## Structure

```
src/
  models/       # ResNet architecture + training
  api/          # FastAPI endpoints
  data/         # Dataset loading + augmentation
  utils/        # Config + metrics
streamlit_app/  # Interface utilisateur
notebooks/      # EDA, training, evaluation
tests/          # Tests unitaires
docker/         # Containerisation
mlruns/         # Tracking experiments
```

---

## Technologies

- **DL:** PyTorch, torchvision, timm
- **API:** FastAPI, Uvicorn
- **UI:** Streamlit
- **MLOps:** MLflow, Docker
- **Tests:** pytest
