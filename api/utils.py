"""
Fonctions utilitaires de l'API de segmentation Cityscapes.
"""

import base64
import io
import numpy as np
from PIL import Image


# Palette officielle Cityscapes pour les 8 catégories regroupées
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

# Dimensions cibles pour le modèle (largeur, hauteur) au format PIL
TAILLE_IMAGE = (512, 256)


def preparer_image(image_pil):
    """
    Convertit une image PIL au format attendu par le modèle :
    (1, 256, 512, 3), float32 normalisé [0, 1].
    """
    image_pil = image_pil.convert("RGB")
    image_pil = image_pil.resize(TAILLE_IMAGE, Image.BILINEAR)
    image_array = np.array(image_pil, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def masque_vers_png_base64(masque_indices):
    """
    Convertit un masque (indices 0-7) en PNG niveaux de gris, base64.
    """
    image = Image.fromarray(masque_indices.astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def masque_vers_image_couleurs_base64(masque_indices):
    """
    Convertit un masque (indices 0-7) en PNG coloré (palette Cityscapes), base64.
    """
    h, w = masque_indices.shape
    image_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, couleur in PALETTE_CITYSCAPES.items():
        image_rgb[masque_indices == idx] = couleur

    image = Image.fromarray(image_rgb, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
