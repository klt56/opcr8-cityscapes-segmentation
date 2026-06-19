"""
Application Streamlit pour la démo du modèle de segmentation Cityscapes.

Permet de :
1. Sélectionner une image parmi les 53 du test set
2. Appeler l'API FastAPI pour faire la prédiction
3. Afficher côte à côte : image originale / masque vérité / masque prédit
"""

import base64
import csv
import io
from pathlib import Path

import requests
import streamlit as st
from PIL import Image
import numpy as np


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

import os
URL_API = os.environ.get("URL_API", "http://localhost:8000")
DOSSIER_PROJET    = Path.home() / "opcr8"
CHEMIN_TEST_CSV   = DOSSIER_PROJET / "data" / "splits" / "test.csv"
DOSSIER_DATA      = DOSSIER_PROJET / "data"

# Palette officielle Cityscapes (mêmes couleurs que dans l'API)
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
def charger_liste_paires():
    """
    Charge la liste des paires (image, masque) du test set depuis test.csv.
    Le décorateur @st.cache_data met le résultat en cache pour ne pas
    relire le CSV à chaque interaction.
    """
    paires = []
    with open(CHEMIN_TEST_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            paires.append((row["image"], row["masque"]))
    return paires


def charger_masque_verite(chemin_masque):
    """
    Charge le masque vérité depuis disque et le colore avec la palette Cityscapes.
    Le masque sur disque est en indices 34 catégories Cityscapes officielles,
    on doit le mapper en 8 catégories puis colorer.
    """
    # Lecture du masque (image en niveaux de gris où chaque pixel = id de catégorie)
    masque_pil = Image.open(chemin_masque)
    masque_pil = masque_pil.resize((512, 256), Image.NEAREST)
    masque_34 = np.array(masque_pil)

    # Mapping 34 -> 8 (table de correspondance)
    # On reconstruit la même LUT que dans data_generator.py
    import sys
    sys.path.insert(0, str(DOSSIER_PROJET / "src"))
    import labels as cs

    lut = np.zeros(256, dtype=np.uint8)
    for label in cs.labels:
        if label.id >= 0:
            lut[label.id] = label.categoryId
    masque_8 = lut[masque_34]

    # Coloration via la palette
    h, w = masque_8.shape
    image_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, couleur in PALETTE_CITYSCAPES.items():
        image_rgb[masque_8 == idx] = couleur

    return Image.fromarray(image_rgb, mode="RGB")


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
        reponse = requests.get(f"{URL_API}/healthcheck", timeout=2)
        return reponse.status_code == 200
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Interface utilisateur Streamlit
# -----------------------------------------------------------------------------

st.set_page_parameters = st.set_page_config(
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
        "❌ **L'API n'est pas accessible** sur `http://localhost:8000`. "
        "Vérifiez qu'elle est bien lancée dans un terminal séparé "
        "(`cd ~/opcr8/api && uvicorn main:app --reload`)."
    )
    st.stop()

st.success("✅ API connectée et opérationnelle")

# Chargement de la liste des images du test set
paires = charger_liste_paires()

# Sélecteur d'image
st.subheader("Sélection de l'image")
liste_noms_images = [Path(img).name for img, _ in paires]
indice_choisi = st.selectbox(
    f"Choisissez une image parmi les {len(paires)} du test set :",
    options=range(len(liste_noms_images)),
    format_func=lambda i: f"{i+1:02d}. {liste_noms_images[i]}",
)

chemin_image_rel, chemin_masque_rel = paires[indice_choisi]
chemin_image  = DOSSIER_DATA / chemin_image_rel
chemin_masque = DOSSIER_DATA / chemin_masque_rel

# Bouton pour lancer la prédiction
if st.button("🔍 Lancer la prédiction", type="primary"):
    with st.spinner("Prédiction en cours..."):
        # Appel API
        masque_predit, erreur = appeler_api_predict(chemin_image)

    if erreur:
        st.error(erreur)
    else:
        # Chargement des autres images pour l'affichage
        image_originale = Image.open(chemin_image)
        masque_verite   = charger_masque_verite(chemin_masque)

        # Affichage 3 colonnes
        st.subheader("Résultats")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image_originale, caption="Image originale", use_container_width=True)

        with col2:
            st.image(masque_verite, caption="Masque vérité terrain", use_container_width=True)

        with col3:
            st.image(masque_predit, caption="Masque prédit par VGG16 U-Net", use_container_width=True)

        # Légende des couleurs
        st.subheader("Légende des catégories")
        cols_legende = st.columns(8)
        for i, (nom, couleur) in enumerate(zip(NOMS_CATEGORIES, PALETTE_CITYSCAPES.values())):
            with cols_legende[i]:
                # Petit carré coloré + nom
                couleur_hex = "#{:02x}{:02x}{:02x}".format(*couleur)
                st.markdown(
                    f"<div style='background-color:{couleur_hex}; "
                    f"width:30px; height:30px; border-radius:5px; "
                    f"margin-bottom:5px;'></div>"
                    f"<small>{nom}</small>",
                    unsafe_allow_html=True
                )

st.markdown("---")
st.caption(
    "Modèle : VGG16 U-Net (transfer learning ImageNet) | "
    "Score test : IoU 0.665, Dice 0.772 | "
    "Projet OpenClassrooms - Future Vision Transport"
)
