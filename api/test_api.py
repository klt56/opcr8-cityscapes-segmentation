"""
Script de test de l'API : simule un client qui envoie une image
et affiche les résultats.

Usage :
    python test_api.py <chemin_vers_une_image>

Si aucun chemin n'est donné, utilise la première image du test set.
"""

import base64
import io
import sys
from pathlib import Path

import requests
from PIL import Image
import matplotlib.pyplot as plt


# URL de l'API locale
URL_API = "http://localhost:8000"


def tester_healthcheck():
    """Vérifie que l'API répond."""
    print("--- Test du healthcheck ---")
    reponse = requests.get(f"{URL_API}/healthcheck")
    print(f"Status code : {reponse.status_code}")
    print(f"Réponse     : {reponse.json()}\n")


def tester_prediction(chemin_image):
    """Envoie une image à l'API et affiche le résultat."""
    print(f"--- Test de la prédiction sur {chemin_image} ---")

    # 1. Ouverture du fichier image
    with open(chemin_image, "rb") as f:
        fichiers = {"image": (Path(chemin_image).name, f, "image/png")}

        # 2. Appel à l'API
        reponse = requests.post(f"{URL_API}/predict", files=fichiers)

    print(f"Status code : {reponse.status_code}")
    if reponse.status_code != 200:
        print(f"Erreur : {reponse.text}")
        return

    # 3. Récupération des données
    data = reponse.json()
    print(f"Taille du masque : {data['taille']}")

    # 4. Décodage du masque coloré (base64 -> PIL Image)
    masque_b64 = data["masque_couleurs_png_base64"]
    masque_bytes = base64.b64decode(masque_b64)
    masque_image = Image.open(io.BytesIO(masque_bytes))

    # 5. Affichage : image originale + masque prédit côte à côte
    image_originale = Image.open(chemin_image)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(image_originale)
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    axes[1].imshow(masque_image)
    axes[1].set_title("Masque prédit (palette Cityscapes)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Healthcheck d'abord
    tester_healthcheck()

    # Récupération de l'image à tester
    if len(sys.argv) > 1:
        chemin = sys.argv[1]
    else:
        # Par défaut : on prend la première image du test set
        dossier_test = Path.home() / "opcr8" / "data" / "leftImg8bit" / "test"
        images = list(dossier_test.rglob("*.png"))
        if not images:
            print("Aucune image trouvée dans le test set.")
            sys.exit(1)
        chemin = str(images[0])

    tester_prediction(chemin)
