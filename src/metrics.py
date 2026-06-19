"""
Métriques personnalisées pour la segmentation Cityscapes 8 catégories.

Toutes les métriques travaillent sur des tenseurs au format one-hot :
- y_true : (batch, H, W, 8) — vérité terrain (1 sur la bonne catégorie, 0 ailleurs)
- y_pred : (batch, H, W, 8) — prédiction softmax (probabilités qui somment à 1)

Implémentation alignée avec le standard Cityscapes :
- Une catégorie absente de la vérité terrain n'est PAS comptée dans la moyenne.
- On évite K.int_shape() qui peut renvoyer None pendant la construction du graphe.
  On utilise des sommes directes sur les axes spatiaux (0, 1, 2) pour rester
  compatible avec l'exécution TensorFlow en mode graph.
"""

import tensorflow as tf
from tensorflow.keras import backend as K


SMOOTH = 1e-6

# Axes spatiaux et de batch sur lesquels on agrège : (batch, H, W).
# Le dernier axe (-1) reste : c'est l'axe des 8 catégories.
AXES_SPATIAUX = [0, 1, 2]


def _iou_par_categorie(y_true, y_pred):
    """
    Calcule l'IoU pour chacune des 8 catégories séparément.
    Renvoie un tenseur de shape (8,).
    """
    intersection = K.sum(y_true * y_pred, axis=AXES_SPATIAUX)
    union = (K.sum(y_true, axis=AXES_SPATIAUX)
             + K.sum(y_pred, axis=AXES_SPATIAUX)
             - intersection)
    return (intersection + SMOOTH) / (union + SMOOTH)


def _dice_par_categorie(y_true, y_pred):
    """
    Calcule le Dice pour chacune des 8 catégories séparément.
    Renvoie un tenseur de shape (8,).
    """
    intersection = K.sum(y_true * y_pred, axis=AXES_SPATIAUX)
    cardinal_total = K.sum(y_true, axis=AXES_SPATIAUX) + K.sum(y_pred, axis=AXES_SPATIAUX)
    return (2.0 * intersection + SMOOTH) / (cardinal_total + SMOOTH)


def _masque_categories_presentes(y_true):
    """
    Renvoie un masque (8,) qui vaut 1 pour chaque catégorie présente
    dans la vérité, 0 sinon.
    """
    nb_pixels_par_cat = K.sum(y_true, axis=AXES_SPATIAUX)
    return K.cast(nb_pixels_par_cat > 0, dtype=tf.float32)


def iou_score(y_true, y_pred):
    """
    Mean IoU sur les catégories EFFECTIVEMENT PRÉSENTES dans la vérité.
    """
    iou_cats = _iou_par_categorie(y_true, y_pred)
    presence = _masque_categories_presentes(y_true)
    return K.sum(iou_cats * presence) / (K.sum(presence) + SMOOTH)


def dice_coef(y_true, y_pred):
    """
    Mean Dice sur les catégories EFFECTIVEMENT PRÉSENTES dans la vérité.
    """
    dice_cats = _dice_par_categorie(y_true, y_pred)
    presence = _masque_categories_presentes(y_true)
    return K.sum(dice_cats * presence) / (K.sum(presence) + SMOOTH)


def dice_loss(y_true, y_pred):
    """
    Dice loss = 1 - Dice coefficient. À MINIMISER pendant l'entraînement.
    """
    dice_cats = _dice_par_categorie(y_true, y_pred)
    return 1.0 - K.mean(dice_cats)


def total_loss(y_true, y_pred):
    """
    Loss combinée : Dice loss + Categorical Cross-Entropy.
    """
    return dice_loss(y_true, y_pred) + tf.keras.losses.categorical_crossentropy(y_true, y_pred)
