"""
Application Streamlit pour la démo du modèle de segmentation Cityscapes.

Permet de :
1. Sélectionner une image parmi les 10 images de démo embarquées
2. Appeler l'API FastAPI pour faire la prédiction
3. Afficher côte à côte : image originale / masque vérité / masque prédit

Les images et masques vérité sont pré-calculés (voir prepare_demo.py) et
embarqués dans app/data_demo/ pour fonctionner en déploiement sans dépendre
du dataset complet ni de src/labels.py.
"""

import base64
import csv
import io
import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

URL_API = os.environ.get("URL_API", "http://localhost:8000")

# Dossier de démo (relatif à l'emplacement de ce script)
DOSSIER_APP = Path(__file__).parent
DOSSIER_DEMO = DOSSIER_APP / "data_demo"
CHEMIN_DEMO_CSV = DOSSIER_DEMO / "demo.csv"

# Palette officielle Cityscapes (pour la légende)
PALETTE_CITYSCAPES = {
    0: (0,   0,   0),       # void
    1: (128, 64,  128),     # flat
    2: (70,  70,  70),      # construction
    3: (153, 153, 153),     # object
    4: (107, 142, 35),      # nature
    5: (70,  130, 180),     # sky
    6: (220, 20,  60),      # human
    7: (0,   0,   142),     # vehicle
}

NOMS_CATEGORIES = ["void", "flat", "construction", "object",
                   "nature", "sky", "human", "vehicle"]


# -----------------------------------------------------------------------------
# Fonctions utilitaires
# -----------------------------------------------------------------------------

@st.cache_data
def charger_liste_demo():
    """Charge la liste des images de démo depuis demo.csv."""
    entrees = []
    with open(CHEMIN_DEMO_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            entrees.append((row["nom"], row["image"], row["masque_verite"]))
    return entrees


def appeler_api_predict(chemin_image):
    """Envoie l'image à l'API et récupère le masque coloré prédit."""
    with open(chemin_image, "rb") as f:
        fichiers = {"image": (Path(chemin_image).name, f, "image/png")}
        reponse = requests.post(f"{URL_API}/predict", files=fichiers)

    if reponse.status_code != 200:
        return None, f"Erreur API : {reponse.status_code} — {reponse.text}"

    data = reponse.json()
    masque_b64 = data["masque_couleurs_png_base64"]
    masque_bytes = base64.b64decode(masque_b64)
    masque_image = Image.open(io.BytesIO(masque_bytes))
    return masque_image, None


def verifier_api():
    """Vérifie que l'API est joignable."""
    try:
        reponse = requests.get(f"{URL_API}/healthcheck", timeout=5)
        return reponse.status_code == 200
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Interface utilisateur Streamlit
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Segmentation Cityscapes — Démo",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Démo de segmentation sémantique — Cityscapes 8 catégories")
st.markdown(
    "Cette application présente le modèle **VGG16 U-Net** entraîné pour la "
    "segmentation de scènes urbaines pour véhicules autonomes. "
    "Sélectionnez une image du jeu de test pour comparer la prédiction du modèle "
    "avec la vérité terrain."
)

# Vérification que l'API est disponible
if not verifier_api():
    st.error(
        "❌ **L'API n'est pas accessible**. "
        "Vérifiez que l'API est bien déployée et que la variable "
        "d'environnement `URL_API` pointe vers la bonne adresse."
    )
    st.stop()

st.success("✅ API connectée et opérationnelle")

# Chargement de la liste des images de démo
entrees = charger_liste_demo()

# Sélecteur d'image
st.subheader("Sélection de l'image")
indice_choisi = st.selectbox(
    f"Choisissez une image parmi les {len(entrees)} du jeu de test :",
    options=range(len(entrees)),
    format_func=lambda i: f"{i+1:02d}. {entrees[i][0]}",
)

nom, image_rel, masque_rel = entrees[indice_choisi]
chemin_image = DOSSIER_DEMO / image_rel
chemin_masque = DOSSIER_DEMO / masque_rel

# Bouton pour lancer la prédiction
if st.button("🔍 Lancer la prédiction", type="primary"):
    with st.spinner("Prédiction en cours..."):
        masque_predit, erreur = appeler_api_predict(chemin_image)

    if erreur:
        st.error(erreur)
    else:
        image_originale = Image.open(chemin_image)
        masque_verite = Image.open(chemin_masque)

        # Affichage 3 colonnes
        st.subheader("Résultats")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image_originale, caption="Image originale", use_container_width=True)

        with col2:
            st.image(masque_verite, caption="Masque vérité terrain (8 catégories)", use_container_width=True)

        with col3:
            st.image(masque_predit, caption="Masque prédit par VGG16 U-Net", use_container_width=True)

        # Légende des couleurs
        st.subheader("Légende des catégories")
        cols_legende = st.columns(8)
        for i, (nom_cat, couleur) in enumerate(zip(NOMS_CATEGORIES, PALETTE_CITYSCAPES.values())):
            with cols_legende[i]:
                couleur_hex = "#{:02x}{:02x}{:02x}".format(*couleur)
                st.markdown(
                    f"<div style='background-color:{couleur_hex}; "
                    f"width:30px; height:30px; border-radius:5px; "
                    f"margin-bottom:5px;'></div>"
                    f"<small>{nom_cat}</small>",
                    unsafe_allow_html=True
                )

st.markdown("---")
st.caption(
    "Modèle : VGG16 U-Net (transfer learning ImageNet) | "
    "Score test : IoU 0.665, Dice 0.772 | "
    "Projet OpenClassrooms - Future Vision Transport"
)
