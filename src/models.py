"""
Architectures de modèles de segmentation pour Cityscapes 8 catégories.

Modèles disponibles :
- unet_mini  : baseline U-Net léger (3 niveaux, 32-64-128 filtres)
- vgg16_unet : U-Net avec encodeur VGG16 pré-entraîné sur ImageNet (gelé)
"""

from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import VGG16


# -----------------------------------------------------------------------------
# Blocs de base (Brique A)
# -----------------------------------------------------------------------------

def conv_block(entree, nb_filtres):
    """
    Bloc de convolution standard : deux Conv2D 3x3 -> BatchNorm -> ReLU.
    'padding=same' garde la même taille spatiale en sortie qu'en entrée.
    """
    x = layers.Conv2D(nb_filtres, 3, padding="same", use_bias=False)(entree)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(nb_filtres, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    return x


def encoder_block(entree, nb_filtres):
    """Étage descendant : conv_block + MaxPooling."""
    skip   = conv_block(entree, nb_filtres)
    sortie = layers.MaxPooling2D(pool_size=2)(skip)
    return skip, sortie


def decoder_block(entree, skip, nb_filtres):
    """Étage montant : UpSampling + concat avec skip + conv_block."""
    x = layers.UpSampling2D(size=2)(entree)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, nb_filtres)
    return x


# -----------------------------------------------------------------------------
# Modèle 1 : U-Net mini (baseline)
# -----------------------------------------------------------------------------

def unet_mini(taille_image=(256, 512), nb_categories=8):
    """
    U-Net mini : 2 niveaux d'encodeur + bottleneck + 2 niveaux de décodeur.
    Filtres : 32 -> 64 -> 128 -> 64 -> 32.
    Léger (~470K paramètres), sert de baseline.
    """
    h, w = taille_image
    entree = Input(shape=(h, w, 3))

    skip1, x = encoder_block(entree, 32)
    skip2, x = encoder_block(x,     64)
    x = conv_block(x, 128)
    x = decoder_block(x, skip2, 64)
    x = decoder_block(x, skip1, 32)
    sortie = layers.Conv2D(nb_categories, 1, activation="softmax")(x)

    return models.Model(inputs=entree, outputs=sortie, name="unet_mini")


# -----------------------------------------------------------------------------
# Modèle 2 : VGG16 U-Net (transfer learning)
# -----------------------------------------------------------------------------

def vgg16_unet(taille_image=(256, 512), nb_categories=8, encoder_trainable=False):
    """
    U-Net avec encodeur VGG16 pré-entraîné sur ImageNet.

    L'encodeur (VGG16) est gelé par défaut pour préserver ses features
    pré-apprises et éviter l'overfitting sur un dataset réduit.
    Le décodeur est entièrement entraînable et spécifique à la tâche.

    Paramètres :
      taille_image      : (hauteur, largeur), doit être divisible par 32
                          (VGG16 fait 5 poolings successifs)
      nb_categories     : nombre de catégories en sortie
      encoder_trainable : si True, dégèle les poids de VGG16 (fine-tuning)

    Renvoie :
      Un objet keras.Model prêt à être compilé.
    """
    h, w = taille_image
    entree = Input(shape=(h, w, 3))

    # --- Encoder : VGG16 pré-entraîné ---
    # 'include_top=False' enlève les couches de classification finales (FC layers)
    # qu'on n'utilise pas pour de la segmentation.
    # 'weights="imagenet"' charge les poids pré-entraînés sur ImageNet.
    # 'input_tensor=entree' branche VGG16 sur notre couche d'entrée.
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=entree)

    # Gel des poids de VGG16 (selon le paramètre).
    # trainable=False indique à Keras de ne PAS mettre à jour ces poids
    # pendant l'entraînement. Les gradients ne sont pas calculés ici, ce qui
    # accélère aussi le pas d'optimisation.
    vgg16.trainable = encoder_trainable

    # Récupération des sorties intermédiaires de VGG16 pour les skip connections.
    # On utilise les couches "blockN_conv3" (la dernière convolution avant le pool
    # de chaque bloc), parce que c'est là que les features sont les plus riches
    # avant la réduction spatiale.
    skip1 = vgg16.get_layer("block1_conv2").output    # (256, 512, 64)
    skip2 = vgg16.get_layer("block2_conv2").output    # (128, 256, 128)
    skip3 = vgg16.get_layer("block3_conv3").output    # (64,  128, 256)
    skip4 = vgg16.get_layer("block4_conv3").output    # (32,   64, 512)
    # Bottleneck : la sortie la plus profonde de VGG16 (avant le dernier pool).
    bottleneck = vgg16.get_layer("block5_conv3").output  # (16, 32, 512)

    # --- Decoder : remontée avec skip connections ---
    # On remonte progressivement en utilisant les skips de VGG16.
    # Le nombre de filtres descend symétriquement : 512 -> 256 -> 128 -> 64.
    x = decoder_block(bottleneck, skip4, 512)   # 16x32  -> 32x64
    x = decoder_block(x,          skip3, 256)   # 32x64  -> 64x128
    x = decoder_block(x,          skip2, 128)   # 64x128 -> 128x256
    x = decoder_block(x,          skip1, 64)    # 128x256 -> 256x512

    # Couche de sortie : Conv 1x1 + softmax, comme pour U-Net mini.
    sortie = layers.Conv2D(nb_categories, 1, activation="softmax")(x)

    return models.Model(inputs=entree, outputs=sortie, name="vgg16_unet")
