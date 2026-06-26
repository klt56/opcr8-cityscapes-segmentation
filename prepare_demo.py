"""
Script de préparation des données de démo pour le déploiement Streamlit.

Prend les 10 premières paires du test set, copie les images originales,
génère les masques vérité colorés (mapping 34->8 catégories), et crée
un CSV de démo. Tout est placé dans app/data_demo/ pour être versionné.

À lancer UNE FOIS en local :
    cd ~/opcr8
    python prepare_demo.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJET = Path.home() / "opcr8"
DATA = PROJET / "data"
TEST_CSV = DATA / "splits" / "test.csv"
DOSSIER_DEMO = PROJET / "app" / "data_demo"
NB_IMAGES = 10

# Palette officielle Cityscapes (identique à streamlit_app.py)
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

# -----------------------------------------------------------------------------
# Import du mapping labels Cityscapes
# -----------------------------------------------------------------------------
sys.path.insert(0, str(PROJET / "src"))
import labels as cs

# Construction de la LUT id -> categoryId (mapping 34 -> 8)
lut = np.zeros(256, dtype=np.uint8)
for label in cs.labels:
    if label.id >= 0:
        lut[label.id] = label.categoryId


def colorer_masque_verite(chemin_masque):
    """Charge le masque gtFine, mappe 34->8, colore avec la palette."""
    masque_pil = Image.open(chemin_masque)
    masque_pil = masque_pil.resize((512, 256), Image.NEAREST)
    masque_34 = np.array(masque_pil)

    masque_8 = lut[masque_34]

    h, w = masque_8.shape
    image_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, couleur in PALETTE_CITYSCAPES.items():
        image_rgb[masque_8 == idx] = couleur

    return Image.fromarray(image_rgb, mode="RGB")


def main():
    # Création des dossiers de sortie
    dossier_images = DOSSIER_DEMO / "images"
    dossier_masques = DOSSIER_DEMO / "masques"
    dossier_images.mkdir(parents=True, exist_ok=True)
    dossier_masques.mkdir(parents=True, exist_ok=True)

    # Lecture des 10 premières paires du test set
    with open(TEST_CSV, encoding="utf-8") as f:
        paires = [(row["image"], row["masque"]) for row in csv.DictReader(f)]

    paires = paires[:NB_IMAGES]
    print(f"Préparation de {len(paires)} paires de démo...\n")

    lignes_csv = []
    for i, (img_rel, masque_rel) in enumerate(paires):
        # Nom de l'image originale (on garde le vrai nom pour le sélecteur)
        nom_img = Path(img_rel).name

        # 1. Copier l'image originale (redimensionnée pour alléger)
        img_src = DATA / img_rel
        img = Image.open(img_src).convert("RGB").resize((512, 256), Image.BILINEAR)
        img_dest = dossier_images / f"img_{i:02d}.png"
        img.save(img_dest)

        # 2. Générer le masque vérité coloré
        masque_src = DATA / masque_rel
        masque_colore = colorer_masque_verite(masque_src)
        masque_dest = dossier_masques / f"mask_{i:02d}.png"
        masque_colore.save(masque_dest)

        # 3. Ligne CSV (chemins relatifs à app/data_demo/)
        lignes_csv.append({
            "nom": nom_img,
            "image": f"images/img_{i:02d}.png",
            "masque_verite": f"masques/mask_{i:02d}.png",
        })

        print(f"  [{i+1:02d}/{len(paires)}] {nom_img}")

    # Écriture du CSV de démo
    chemin_csv = DOSSIER_DEMO / "demo.csv"
    with open(chemin_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["nom", "image", "masque_verite"])
        writer.writeheader()
        writer.writerows(lignes_csv)

    print(f"\nTerminé. Données de démo dans : {DOSSIER_DEMO}")
    print(f"CSV : {chemin_csv}")


if __name__ == "__main__":
    main()
