"""
Application Streamlit pour la classification CIFAR-10.

Interface utilisateur interactive pour tester le modèle
de classification d'images.
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

# CSS personnalisé
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
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classification_model():
    """
    Charge le modèle de classification.
    Utilise le cache Streamlit pour éviter de recharger à chaque interaction.
    """
    device = get_device()
    model_path = MODELS_DIR / "best_model.pth"
    
    if model_path.exists():
        model = load_model(
            str(model_path),
            model_name=MODEL_NAME,
            num_classes=NUM_CLASSES,
            device=str(device)
        )
        st.success(f"✅ Modèle chargé depuis {model_path}")
    else:
        st.warning("⚠️ Modèle pré-entraîné non trouvé. Utilisation d'un modèle non entraîné.")
        model = create_model()
        model = model.to(device)
        model.eval()
    
    return model, device


def predict_image(model, image, device):
    """
    Effectue une prédiction sur une image.
    
    Args:
        model: Modèle de classification
        image: Image PIL
        device: Device (CPU/GPU)
        
    Returns:
        Tuple (classe prédite, confiance, probabilités)
    """
    # Prétraitement
    input_tensor = preprocess_single_image(image)
    input_tensor = input_tensor.to(device)
    
    # Inférence
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    
    class_id = predicted.item()
    class_name = CIFAR10_CLASSES[class_id]
    confidence_value = confidence.item() * 100
    
    # Probabilités pour toutes les classes
    probs = {
        CIFAR10_CLASSES[i]: probabilities[0][i].item() * 100
        for i in range(len(CIFAR10_CLASSES))
    }
    
    return class_name, confidence_value, probs


def create_probability_chart(probabilities):
    """
    Crée un graphique des probabilités.
    
    Args:
        probabilities: Dictionnaire {classe: probabilité}
        
    Returns:
        Figure Plotly
    """
    # Trier par probabilité
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
        title="Probabilités par Classe",
        xaxis_title="Probabilité (%)",
        yaxis_title="Classe",
        height=400,
        margin=dict(l=100, r=20, t=50, b=50),
        xaxis=dict(range=[0, 105])
    )
    
    return fig


def main():
    """Fonction principale de l'application."""
    
    # En-tête
    st.markdown('<h1 class="main-header">🖼️ Classification d\'Images CIFAR-10</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Barre latérale
    with st.sidebar:
        st.header("📋 Informations")
        st.markdown("""
        **Projet Deep Learning**
        - **Étudiant**: Ahmed Belhareth
        - **Module**: Deep Learning
        - **Dataset**: CIFAR-10
        - **Modèle**: ResNet-18
        """)
        
        st.header("📊 Classes CIFAR-10")
        for i, class_name in enumerate(CIFAR10_CLASSES):
            st.write(f"{i+1}. {class_name}")
        
        st.header("ℹ️ Instructions")
        st.markdown("""
        1. Uploadez une image (JPG, PNG)
        2. Le modèle prédit la classe
        3. Consultez les probabilités
        
        **Note**: Les images sont redimensionnées
        automatiquement en 32×32 pixels.
        """)
    
    # Charger le modèle
    model, device = load_classification_model()
    
    # Zone principale
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload d'Image")
        
        uploaded_file = st.file_uploader(
            "Choisissez une image...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Formats acceptés: JPG, JPEG, PNG, WEBP"
        )
        
        # Images d'exemple
        st.subheader("Ou essayez avec des exemples")
        example_cols = st.columns(5)
        example_images = [
            ("✈️ Avion", "airplane"),
            ("🚗 Auto", "automobile"),
            ("🐱 Chat", "cat"),
            ("🐕 Chien", "dog"),
            ("🚢 Navire", "ship")
        ]
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Image uploadée", use_container_width=True)
            
            # Bouton de prédiction
            if st.button("🔮 Prédire", type="primary", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    class_name, confidence, probabilities = predict_image(
                        model, image, device
                    )
                    
                    # Stocker dans session state
                    st.session_state['prediction'] = {
                        'class_name': class_name,
                        'confidence': confidence,
                        'probabilities': probabilities
                    }
    
    with col2:
        st.header("📊 Résultats")
        
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            
            # Afficher la prédiction principale
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            
            # Couleur selon la confiance
            if pred['confidence'] >= 80:
                conf_class = "confidence-high"
            elif pred['confidence'] >= 50:
                conf_class = "confidence-medium"
            else:
                conf_class = "confidence-low"
            
            st.markdown(f"""
            ### 🎯 Prédiction: **{pred['class_name']}**
            
            <p class="{conf_class}">
                Confiance: {pred['confidence']:.1f}%
            </p>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Graphique des probabilités
            fig = create_probability_chart(pred['probabilities'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Top 3 prédictions
            st.subheader("🏆 Top 3 Prédictions")
            sorted_probs = sorted(
                pred['probabilities'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            for i, (class_name, prob) in enumerate(sorted_probs):
                medal = ["🥇", "🥈", "🥉"][i]
                st.write(f"{medal} **{class_name}**: {prob:.1f}%")
        
        else:
            st.info("👆 Uploadez une image et cliquez sur 'Prédire' pour voir les résultats.")
    
    # Section informations techniques
    st.markdown("---")
    st.header("🔧 Informations Techniques")
    
    tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)
    
    with tech_col1:
        st.metric("Modèle", MODEL_NAME.upper())
    
    with tech_col2:
        st.metric("Classes", NUM_CLASSES)
    
    with tech_col3:
        st.metric("Résolution", "32×32 px")
    
    with tech_col4:
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
        st.metric("Device", device_name)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
        <p>Projet Deep Learning - Classification CIFAR-10</p>
        <p>Ahmed Belhareth | Module Deep Learning | Prof. Haythem Ghazouani</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
