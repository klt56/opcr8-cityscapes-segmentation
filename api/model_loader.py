"""
Chargement du modèle VGG16 U-Net entraîné (singleton).
"""

import sys
from pathlib import Path
from tensorflow.keras.models import load_model

# Ajout du dossier 'src' au PYTHONPATH pour importer les métriques custom
DOSSIER_PROJET = Path(__file__).parent.parent
sys.path.insert(0, str(DOSSIER_PROJET / "src"))

from metrics import iou_score, dice_coef, dice_loss


CHEMIN_MODELE = DOSSIER_PROJET / "models" / "vgg16_unet_aug_10epochs_best.h5"

_modele = None


def charger_modele():
    """
    Charge le modèle depuis disque (une seule fois grâce au singleton).
    """
    global _modele
    if _modele is None:
        print(f"Chargement du modèle depuis {CHEMIN_MODELE}...")
        _modele = load_model(
            str(CHEMIN_MODELE),
            custom_objects={
                "iou_score" : iou_score,
                "dice_coef" : dice_coef,
                "dice_loss" : dice_loss,
            }
        )
        print(f"Modèle chargé : {_modele.name}")
        print(f"  Paramètres : {_modele.count_params():,}")
    return _modele
