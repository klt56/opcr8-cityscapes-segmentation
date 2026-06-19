"""
API FastAPI de segmentation d'images Cityscapes en 8 catégories.

Endpoints :
- GET  /              : page d'accueil avec liste des routes
- GET  /healthcheck   : vérifie que l'API et le modèle sont opérationnels
- POST /predict       : reçoit une image, renvoie le masque prédit
"""

import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from model_loader import charger_modele
from utils import (
    preparer_image,
    masque_vers_png_base64,
    masque_vers_image_couleurs_base64,
    NOMS_CATEGORIES,
)


# Création de l'application FastAPI
app = FastAPI(
    title="API Segmentation Cityscapes",
    description="API REST pour la segmentation sémantique de scènes urbaines "
                "via un modèle VGG16 U-Net entraîné sur Cityscapes (8 catégories).",
    version="1.0.0",
)


# Chargement du modèle au démarrage de l'API
# (s'exécute une seule fois, avant la première requête)
@app.on_event("startup")
async def evenement_demarrage():
    """Charge le modèle en mémoire dès le lancement de l'API."""
    print("=" * 60)
    print("Démarrage de l'API de segmentation Cityscapes")
    print("=" * 60)
    charger_modele()
    print("API prête à recevoir des requêtes.")
    print("=" * 60)


# -----------------------------------------------------------------------------
# Endpoint racine : page d'accueil
# -----------------------------------------------------------------------------
@app.get("/")
async def racine():
    """Page d'accueil : liste les endpoints disponibles."""
    return {
        "nom": "API Segmentation Cityscapes",
        "version": "1.0.0",
        "description": "Segmentation sémantique en 8 catégories",
        "endpoints": {
            "GET /":            "Cette page d'accueil",
            "GET /healthcheck": "Vérification de l'état de l'API",
            "POST /predict":    "Prédire le masque de segmentation d'une image",
            "GET /docs":        "Documentation interactive Swagger",
        },
        "categories": NOMS_CATEGORIES,
    }


# -----------------------------------------------------------------------------
# Endpoint healthcheck : vérification de l'état
# -----------------------------------------------------------------------------
@app.get("/healthcheck")
async def healthcheck():
    """
    Vérifie que l'API est opérationnelle et que le modèle est chargé.
    Utile pour les sondes de monitoring (Azure App Service par exemple).
    """
    try:
        modele = charger_modele()
        return {
            "status":           "ok",
            "modele_charge":    True,
            "nom_modele":       modele.name,
            "nb_parametres":    int(modele.count_params()),
            "nb_categories":    len(NOMS_CATEGORIES),
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Modèle non disponible : {str(e)}"
        )


# -----------------------------------------------------------------------------
# Endpoint predict : prédiction du masque
# -----------------------------------------------------------------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Reçoit une image et renvoie le masque de segmentation prédit.

    Paramètres :
        image : fichier image (JPG, PNG, etc.)

    Renvoie un JSON contenant :
        - masque_indices_png_base64    : masque (0-7) en PNG niveaux de gris, base64
        - masque_couleurs_png_base64   : masque coloré (palette Cityscapes), base64
        - taille                       : dimensions du masque (hauteur, largeur)
        - categories                   : noms des catégories (pour référence)
    """
    # Validation basique du type de fichier
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Le fichier doit être une image. Type reçu : {image.content_type}"
        )

    # Lecture des octets de l'image
    try:
        contenu = await image.read()
        image_pil = Image.open(io.BytesIO(contenu))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Impossible de décoder l'image : {str(e)}"
        )

    # Préparation pour le modèle
    image_array = preparer_image(image_pil)

    # Prédiction
    modele = charger_modele()
    y_pred = modele.predict(image_array, verbose=0)

    # Conversion softmax → indices (argmax sur l'axe des canaux)
    # y_pred shape : (1, 256, 512, 8) → masque shape : (256, 512)
    masque_indices = np.argmax(y_pred[0], axis=-1).astype(np.uint8)

    # Encodage des sorties en base64
    png_indices  = masque_vers_png_base64(masque_indices)
    png_couleurs = masque_vers_image_couleurs_base64(masque_indices)

    return JSONResponse({
        "masque_indices_png_base64":  png_indices,
        "masque_couleurs_png_base64": png_couleurs,
        "taille": {
            "hauteur": int(masque_indices.shape[0]),
            "largeur": int(masque_indices.shape[1]),
        },
        "categories": NOMS_CATEGORIES,
    })
