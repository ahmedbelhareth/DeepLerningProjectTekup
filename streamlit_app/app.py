"""
Application Streamlit pour la classification CIFAR-10.

Interface utilisateur interactive pour tester le modele
de classification d'images avec support Transfer Learning.
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import io

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import (
    CIFAR10_CLASSES,
    MODELS_DIR,
    MODEL_NAME,
    NUM_CLASSES,
    STREAMLIT_TITLE,
    STREAMLIT_ICON,
    get_device
)
from src.models.architecture import load_model, create_model
from src.data.preprocessing import preprocess_single_image

# Configuration de la page
st.set_page_config(
    page_title=STREAMLIT_TITLE,
    page_icon=STREAMLIT_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalise
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .transfer-learning-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
    }
    .improvement-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .evolution-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    .version-badge {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
        margin: 0.5rem;
    }
    .timeline-item {
        border-left: 3px solid #667eea;
        padding-left: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classification_model(model_type="standard"):
    """
    Charge le modele de classification.

    Args:
        model_type: "standard" ou "transfer_learning"
    """
    device = get_device()

    if model_type == "transfer_learning":
        model_path = MODELS_DIR / "best_model_transfer_learning.pth"
        model_name = "Transfer Learning (STL-10 + CIFAR-10)"
    else:
        model_path = MODELS_DIR / "best_model.pth"
        model_name = "Standard (CIFAR-10)"

    if model_path.exists():
        model = load_model(
            str(model_path),
            model_name=MODEL_NAME,
            num_classes=NUM_CLASSES,
            device=str(device)
        )
        return model, device, model_name, True
    else:
        model = create_model()
        model = model.to(device)
        model.eval()
        return model, device, model_name, False


def predict_image(model, image, device):
    """
    Effectue une prediction sur une image.
    """
    input_tensor = preprocess_single_image(image)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    class_id = predicted.item()
    class_name = CIFAR10_CLASSES[class_id]
    confidence_value = confidence.item() * 100

    probs = {
        CIFAR10_CLASSES[i]: probabilities[0][i].item() * 100
        for i in range(len(CIFAR10_CLASSES))
    }

    return class_name, confidence_value, probs


def create_probability_chart(probabilities):
    """Cree un graphique des probabilites."""
    sorted_probs = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))

    fig = go.Figure(go.Bar(
        x=list(sorted_probs.values()),
        y=list(sorted_probs.keys()),
        orientation='h',
        marker_color=['#667eea' if i == 0 else '#b8c1ec'
                      for i in range(len(sorted_probs))],
        text=[f'{v:.1f}%' for v in sorted_probs.values()],
        textposition='outside'
    ))

    fig.update_layout(
        title="Probabilites par Classe",
        xaxis_title="Probabilite (%)",
        yaxis_title="Classe",
        height=400,
        margin=dict(l=100, r=20, t=50, b=50),
        xaxis=dict(range=[0, 105])
    )

    return fig


def create_comparison_chart():
    """Cree un graphique de comparaison des modeles."""
    models = ['Standard\n(CIFAR-10)', 'Transfer Learning\n(STL-10 + CIFAR-10)']
    accuracy = [85.5, 68.18]  # Valeurs reelles du training

    colors = ['#b8c1ec', '#667eea']

    fig = go.Figure(go.Bar(
        x=models,
        y=accuracy,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in accuracy],
        textposition='outside'
    ))

    fig.update_layout(
        title="Comparaison des Performances",
        yaxis_title="Accuracy (%)",
        height=350,
        yaxis=dict(range=[0, 100]),
        showlegend=False
    )

    return fig


def main():
    """Fonction principale de l'application."""

    # En-tete
    st.markdown('<h1 class="main-header">Classification d\'Images CIFAR-10</h1>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Section Evolutions du Projet
    st.header("Evolutions du Projet")

    st.markdown("""
    <div style="text-align: center;">
        <span class="version-badge">v1.0 - Base</span>
        <span style="font-size: 1.5rem;"> → </span>
        <span class="version-badge">v2.0 - MLOps</span>
        <span style="font-size: 1.5rem;"> → </span>
        <span class="version-badge">v3.0 - Transfer Learning</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    evo_col1, evo_col2, evo_col3 = st.columns(3)

    with evo_col1:
        st.markdown("""
        <div class="evolution-card">
            <h3>v1.0 - Base</h3>
            <p><strong>ResNet-18 CIFAR-10</strong></p>
            <p>Accuracy: 87.09%</p>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <p>✓ Entrainement standard</p>
            <p>✓ API FastAPI</p>
            <p>✓ Interface Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

    with evo_col2:
        st.markdown("""
        <div class="evolution-card">
            <h3>v2.0 - MLOps</h3>
            <p><strong>Pipeline CI/CD</strong></p>
            <p>9 Jobs GitHub Actions</p>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <p>✓ MLflow Tracking</p>
            <p>✓ Docker Multi-Stage</p>
            <p>✓ Tests Automatises</p>
        </div>
        """, unsafe_allow_html=True)

    with evo_col3:
        st.markdown("""
        <div class="evolution-card">
            <h3>v3.0 - Transfer Learning</h3>
            <p><strong>STL-10 + CIFAR-10</strong></p>
            <p>Accuracy: 68.18%</p>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <p>✓ Pre-entrainement STL-10</p>
            <p>✓ MixUp + CutMix</p>
            <p>✓ Label Smoothing</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Barre laterale
    with st.sidebar:
        st.header("Informations")
        st.markdown("""
        **Projet Deep Learning**
        - **Etudiant**: Ahmed Belhareth
        - **Module**: Deep Learning
        - **Dataset**: CIFAR-10 + STL-10
        - **Modele**: ResNet-18
        """)

        st.header("Selection du Modele")
        model_choice = st.radio(
            "Choisir le modele:",
            ["Transfer Learning (STL-10 + CIFAR-10)", "Standard (CIFAR-10 uniquement)"],
            index=0
        )

        st.header("Classes CIFAR-10")
        for i, class_name in enumerate(CIFAR10_CLASSES):
            st.write(f"{i+1}. {class_name}")

        st.header("Instructions")
        st.markdown("""
        1. Selectionnez le modele
        2. Uploadez une image (JPG, PNG)
        3. Cliquez sur Predire
        4. Consultez les resultats
        """)

    # Determiner le type de modele
    model_type = "transfer_learning" if "Transfer" in model_choice else "standard"

    # Section Transfer Learning
    st.header("Transfer Learning: STL-10 + CIFAR-10")

    tl_col1, tl_col2, tl_col3 = st.columns(3)

    with tl_col1:
        st.markdown("""
        <div class="transfer-learning-box">
            <h3>Phase 1: Pre-entrainement</h3>
            <p><strong>Dataset:</strong> STL-10</p>
            <p><strong>Images:</strong> 4,500 (96x96 -> 32x32)</p>
            <p><strong>Epochs:</strong> 5</p>
            <p><strong>Accuracy:</strong> 40.57%</p>
        </div>
        """, unsafe_allow_html=True)

    with tl_col2:
        st.markdown("""
        <div class="transfer-learning-box">
            <h3>Phase 2: Fine-tuning</h3>
            <p><strong>Dataset:</strong> CIFAR-10</p>
            <p><strong>Images:</strong> 50,000 (32x32)</p>
            <p><strong>Epochs:</strong> 15</p>
            <p><strong>Accuracy:</strong> 64.52%</p>
        </div>
        """, unsafe_allow_html=True)

    with tl_col3:
        st.markdown("""
        <div class="transfer-learning-box">
            <h3>Resultats Finaux</h3>
            <p><strong>Test Accuracy:</strong> 68.18%</p>
            <p><strong>Techniques:</strong></p>
            <p>MixUp, CutMix, Label Smoothing</p>
            <p><strong>Temps:</strong> 104 min</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Graphique de comparaison
    st.subheader("Comparaison des Approches")

    comp_col1, comp_col2 = st.columns([1, 1])

    with comp_col1:
        fig_comp = create_comparison_chart()
        st.plotly_chart(fig_comp, use_container_width=True)

    with comp_col2:
        st.markdown("""
        ### Avantages du Transfer Learning

        **Pre-entrainement sur STL-10:**
        - Images plus grandes (96x96) redimensionnees
        - 9 classes communes avec CIFAR-10
        - Apprentissage de features generales

        **Techniques Avancees:**
        - **MixUp** (alpha=0.2): Melange d'images
        - **CutMix** (alpha=1.0): Decoupage/collage
        - **Label Smoothing** (0.1): Regularisation
        - **OneCycleLR**: Learning rate adaptatif

        **Architecture:**
        - ResNet-18 pre-entraine ImageNet
        - Fine-tuning complet du backbone
        - AdamW avec weight decay
        """)

    st.markdown("---")

    # Charger le modele selectionne
    model, device, model_name, model_loaded = load_classification_model(model_type)

    if model_loaded:
        st.success(f"Modele charge: {model_name}")
    else:
        st.warning(f"Modele {model_name} non trouve. Utilisation d'un modele non entraine.")

    # Zone de prediction
    st.header("Prediction d'Image")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload d'Image")

        uploaded_file = st.file_uploader(
            "Choisissez une image...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Formats acceptes: JPG, JPEG, PNG, WEBP"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Image uploadee", use_container_width=True)

            if st.button("Predire", type="primary", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    class_name, confidence, probabilities = predict_image(
                        model, image, device
                    )

                    st.session_state['prediction'] = {
                        'class_name': class_name,
                        'confidence': confidence,
                        'probabilities': probabilities,
                        'model_type': model_name
                    }

    with col2:
        st.subheader("Resultats")

        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']

            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

            if pred['confidence'] >= 80:
                conf_class = "confidence-high"
            elif pred['confidence'] >= 50:
                conf_class = "confidence-medium"
            else:
                conf_class = "confidence-low"

            st.markdown(f"""
            ### Prediction: **{pred['class_name']}**

            <p class="{conf_class}">
                Confiance: {pred['confidence']:.1f}%
            </p>
            <p><small>Modele: {pred['model_type']}</small></p>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            fig = create_probability_chart(pred['probabilities'])
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Top 3 Predictions")
            sorted_probs = sorted(
                pred['probabilities'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            for i, (class_name, prob) in enumerate(sorted_probs):
                medal = ["1.", "2.", "3."][i]
                st.write(f"{medal} **{class_name}**: {prob:.1f}%")

        else:
            st.info("Uploadez une image et cliquez sur 'Predire' pour voir les resultats.")

    # Section informations techniques
    st.markdown("---")
    st.header("Informations Techniques")

    tech_col1, tech_col2, tech_col3, tech_col4, tech_col5 = st.columns(5)

    with tech_col1:
        st.metric("Modele", MODEL_NAME.upper())

    with tech_col2:
        st.metric("Classes", NUM_CLASSES)

    with tech_col3:
        st.metric("Resolution", "32x32 px")

    with tech_col4:
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
        st.metric("Device", device_name)

    with tech_col5:
        st.metric("Transfer Learning", "STL-10")

    # Section MLflow
    st.markdown("---")
    st.header("MLflow Tracking")

    mlflow_col1, mlflow_col2 = st.columns([1, 1])

    with mlflow_col1:
        st.markdown("""
        ### Metriques Enregistrees
        - `phase1_train_loss`, `phase1_train_acc`
        - `phase1_val_loss`, `phase1_val_acc`
        - `phase2_train_loss`, `phase2_train_acc`
        - `phase2_val_loss`, `phase2_val_acc`
        - `test_loss`, `test_accuracy`
        - `total_training_time_seconds`
        """)

    with mlflow_col2:
        st.markdown("""
        ### Parametres Enregistres
        - `model_name`: resnet18
        - `num_epochs_stl10`: 5
        - `num_epochs_cifar10`: 15
        - `batch_size`: 128
        - `learning_rate`: 0.001
        - `mixup_alpha`: 0.2
        - `cutmix_alpha`: 1.0
        - `label_smoothing`: 0.1
        """)

    st.info("Accedez au dashboard MLflow: http://localhost:5000")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
        <p>Projet Deep Learning - Classification CIFAR-10 avec Transfer Learning</p>
        <p>Ahmed Belhareth | Module Deep Learning | Prof. Haythem Ghazouani</p>
        <p><strong>Datasets:</strong> CIFAR-10 + STL-10 | <strong>Framework:</strong> PyTorch + MLflow</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
