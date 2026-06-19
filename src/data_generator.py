"""
Générateur de données pour la segmentation Cityscapes (8 catégories).
Hérite de tf.keras.utils.Sequence pour permettre le chargement en parallèle
sur plusieurs cœurs CPU pendant l'entraînement.

Briques :
- Brique 1 : squelette + lecture CSV
- Brique 2 : chargement réel image + masque
- Brique 3 : redimensionnement + mapping 34->8
- Brique 4 : normalisation + one-hot
- Brique 5 : on_epoch_end + augmentation albumentations  ← actuelle
"""

import csv
import numpy as np
from pathlib import Path
from PIL import Image
from tensorflow.keras.utils import Sequence
import albumentations as A

import labels as cs


# Lookup table 34 -> 8 (construite une seule fois au chargement du module)
LUT_34_VERS_8 = np.zeros(256, dtype=np.uint8)
for label in cs.labels:
    if label.id >= 0:
        LUT_34_VERS_8[label.id] = label.categoryId


# Pipeline d'augmentation pour le jeu d'ENTRAÎNEMENT uniquement.
# albumentations garantit que la même transformation est appliquée à l'image
# ET à son masque (cohérence géométrique préservée).
#
# Les probabilités sont réglées de manière conservatrice : on veut diversifier
# sans dénaturer les scènes urbaines.
AUGMENTATION_TRAIN = A.Compose([
    A.HorizontalFlip(p=0.5),                # retournement horizontal une fois sur deux
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5,
    ),                                       # variations d'éclairage modérées
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=15,
        val_shift_limit=10,
        p=0.3,
    ),                                       # variations de teinte/saturation légères
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.10,
        rotate_limit=10,
        p=0.5,
    ),                                       # léger zoom + rotation + translation
])


class CityscapesDataGenerator(Sequence):
    """
    Générateur Keras pour les paires (image, masque) de Cityscapes 8 catégories.

    Paramètres :
        chemin_csv     : chemin vers train.csv / val.csv / test.csv
        dossier_data   : dossier racine "data/"
        batch_size     : nombre d'images par batch (défaut : 8)
        taille_image   : (hauteur, largeur) cible (défaut : 256 x 512)
        nb_categories  : nombre de catégories en sortie (défaut : 8)
        augmenter      : True pour appliquer l'augmentation (UNIQUEMENT pour le train),
                         False pour val/test (défaut : False)
        graine         : graine aléatoire pour la reproductibilité du mélange
    """

    def __init__(self, chemin_csv, dossier_data, batch_size=8,
                 taille_image=(256, 512), nb_categories=8,
                 augmenter=False, graine=42):
        self.dossier_data  = Path(dossier_data)
        self.batch_size    = batch_size
        self.taille_image  = taille_image
        self.nb_categories = nb_categories
        self.augmenter     = augmenter

        # Générateur aléatoire dédié à cette instance, pour ne pas perturber
        # le random global du notebook. Permet la reproductibilité.
        self.rng = np.random.default_rng(graine)

        with open(chemin_csv, encoding="utf-8") as f:
            self.paires = [(row["image"], row["masque"])
                           for row in csv.DictReader(f)]

        # Indices des paires ; on les mélangera à chaque fin d'epoch.
        # On garde un indice plutôt que de muter la liste self.paires elle-même,
        # ce qui rend le code plus clair.
        self.indices = np.arange(len(self.paires))

    def __len__(self):
        return len(self.paires) // self.batch_size

    def __getitem__(self, idx):
        """
        Renvoie le batch numéro 'idx' au format final attendu par Keras :
        - X : images normalisées en [0, 1], shape (batch, H, W, 3)
        - y : masques en one-hot, shape (batch, H, W, 8)
        Si self.augmenter=True, applique aussi les transformations aléatoires.
        """
        debut = idx * self.batch_size
        fin   = debut + self.batch_size
        indices_batch = self.indices[debut:fin]

        h, w = self.taille_image
        taille_pil = (w, h)

        images  = []
        masques = []

        for i in indices_batch:
            chemin_img_rel, chemin_msk_rel = self.paires[i]
            chemin_img = self.dossier_data / chemin_img_rel
            chemin_msk = self.dossier_data / chemin_msk_rel

            img_pil = Image.open(chemin_img).resize(taille_pil, Image.BILINEAR)
            image = np.array(img_pil)

            msk_pil = Image.open(chemin_msk).resize(taille_pil, Image.NEAREST)
            masque_34 = np.array(msk_pil)
            masque_8 = LUT_34_VERS_8[masque_34]

            # Application de l'augmentation SI activée pour ce générateur.
            # albumentations attend des numpy arrays en uint8 pour l'image
            # et le masque, et fait sa "magie" en interne pour garder
            # image+masque alignés après transformation.
            if self.augmenter:
                augmente = AUGMENTATION_TRAIN(image=image, mask=masque_8)
                image    = augmente["image"]
                masque_8 = augmente["mask"]

            images.append(image)
            masques.append(masque_8)

        X = np.stack(images,  axis=0).astype(np.float32)
        y = np.stack(masques, axis=0).astype(np.uint8)

        # Normalisation [0, 255] -> [0, 1]
        X = X / 255.0

        # One-hot encoding des masques
        y_one_hot = np.eye(self.nb_categories, dtype=np.float32)[y]

        return X, y_one_hot

    def on_epoch_end(self):
        """
        Appelée automatiquement par Keras à la fin de chaque epoch.
        Mélange l'ordre des paires pour que le modèle ne les voie pas
        toujours dans la même séquence d'une epoch à l'autre.

        Important : on mélange SEULEMENT les indices, pas la liste des paires
        elle-même. Ça simplifie le debug et évite les effets de bord.
        """
        self.rng.shuffle(self.indices)
