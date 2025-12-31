# Classification CIFAR-10 - Deep Learning

**Etudiant:** Ahmed Belhareth
**Module:** Deep Learning
**Enseignant:** Haythem Ghazouani

---

## Evolutions du Projet

| Version | Description | Fonctionnalites |
|---------|-------------|-----------------|
| **v1.0** | Base | ResNet-18, API FastAPI, Interface Streamlit |
| **v2.0** | MLOps | Pipeline CI/CD (11 jobs), MLflow, Docker, Tests |
| **v3.0** | Transfer Learning | STL-10 + CIFAR-10, MixUp, CutMix, Label Smoothing |

---

## Probleme

Classifier automatiquement des images en 10 categories (avion, automobile, oiseau, chat, cerf, chien, grenouille, cheval, navire, camion) a partir du dataset CIFAR-10 (60k images 32x32).

## Solution

### Approche Standard
ResNet18 pre-entraine (ImageNet) avec fine-tuning sur CIFAR-10.

| Metrique | Resultat |
|----------|----------|
| Accuracy | **87.09%** |
| F1-Score | **0.8705** |

### Approche Transfer Learning (STL-10 + CIFAR-10)

| Phase | Dataset | Images | Epochs | Accuracy |
|-------|---------|--------|--------|----------|
| Pre-entrainement | STL-10 | 4,500 (96x96 -> 32x32) | 5 | 40.57% |
| Fine-tuning | CIFAR-10 | 50,000 (32x32) | 15 | 64.52% |
| **Test Final** | CIFAR-10 | 10,000 | - | **68.18%** |

**Techniques avancees:**
- MixUp (alpha=0.2)
- CutMix (alpha=1.0)
- Label Smoothing (0.1)
- OneCycleLR Scheduler

---

## Livrables

| Exigence | Implementation |
|----------|---------------|
| Modele standard | `models/best_model.pth` |
| Modele Transfer Learning | `models/best_model_transfer_learning.pth` |
| API REST | FastAPI avec Swagger (`/docs`) |
| Interface UI | Streamlit avec visualisation Transfer Learning |
| Tracking MLOps | MLflow |
| Conteneurisation | Docker + docker-compose |
| CI/CD | GitHub Actions (11 jobs) |
| Tests | pytest (29/29 passing) |
| Notebooks | 3 notebooks avec outputs |

---

## Quick Start

```bash
# Installation
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Entrainement Standard
python -m src.models.training

# Entrainement Transfer Learning
python -c "from src.models.transfer_learning import transfer_learning_training; transfer_learning_training()"

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
  models/       # ResNet architecture + training + transfer_learning
  api/          # FastAPI endpoints
  data/         # Dataset loading + augmentation
  utils/        # Config + metrics
streamlit_app/  # Interface utilisateur avec evolutions
notebooks/      # EDA, training, evaluation
tests/          # Tests unitaires
docker/         # Containerisation
mlruns/         # Tracking experiments
.github/        # CI/CD Pipeline (11 jobs)
```

---

## Pipeline CI/CD (11 Jobs)

| Job | Description |
|-----|-------------|
| 1. Code Quality | Black, Flake8, Bandit, Safety |
| 2. Unit Tests | pytest avec coverage |
| 3. Integration Tests | Tests API |
| 4. Model Validation | Architecture validation |
| 5. Build Docker | API + Streamlit images |
| 6. Transfer Learning Validation | MixUp, CutMix tests |
| 7. Train Model | Entrainement standard (manuel) |
| 8. Train Transfer Learning | Entrainement TL (manuel) |
| 9. Deploy Staging | Environnement de test |
| 10. Deploy Production | Environnement de prod |
| 11. Notify Status | Notification finale |

---

## Technologies

- **DL:** PyTorch, torchvision, timm
- **API:** FastAPI, Uvicorn
- **UI:** Streamlit, Plotly
- **MLOps:** MLflow, Docker, GitHub Actions
- **Tests:** pytest, pytest-cov

---

## Transfer Learning - Details Techniques

### Architecture
```
ResNet-18 (ImageNet pretrained)
    |
    v
Phase 1: Pre-entrainement STL-10
    - 4,500 images (96x96 -> 32x32)
    - 9 classes communes
    - AdamW optimizer
    |
    v
Phase 2: Fine-tuning CIFAR-10
    - 50,000 images
    - MixUp + CutMix augmentation
    - Label Smoothing
    - OneCycleLR scheduler
```

### Hyperparametres
```python
batch_size = 128
learning_rate = 0.001
mixup_alpha = 0.2
cutmix_alpha = 1.0
label_smoothing = 0.1
weight_decay = 0.05
```

---

## Auteur

**Ahmed Belhareth**
Module Deep Learning - Prof. Haythem Ghazouani
