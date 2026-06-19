"""
Module d'entraînement pour la segmentation Cityscapes.
"""

from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
)
from tensorflow.keras.optimizers import Adam
import mlflow

from metrics import iou_score, dice_coef, dice_loss


class MLflowLogger(Callback):
    """
    Logue les métriques de chaque epoch dans MLflow.
    """
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        for nom_metrique, valeur in logs.items():
            mlflow.log_metric(nom_metrique, float(valeur), step=epoch)


def compiler_et_entrainer(modele, gen_train, gen_val, nom_run,
                          nb_epochs=50, learning_rate=1e-3,
                          patience_earlystop=10, patience_reducelr=5,
                          dossier_models="/home/klt/opcr8/models"):
    """
    Compile un modèle Keras, lance l'entraînement avec callbacks et MLflow.
    """
    Path(dossier_models).mkdir(parents=True, exist_ok=True)
    # On utilise le format .h5 (universellement supporté par ModelCheckpoint).
    # Le format .keras récent a un bug connu avec l'option 'options' dans TF 2.13.
    chemin_best = Path(dossier_models) / f"{nom_run}_best.h5"

    modele.compile(
        optimizer = Adam(learning_rate=learning_rate),
        loss      = dice_loss,
        metrics   = [iou_score, dice_coef],
    )

    callbacks = [
        EarlyStopping(
            monitor              = "val_iou_score",
            mode                 = "max",
            patience             = patience_earlystop,
            restore_best_weights = True,
            verbose              = 1,
        ),
        ModelCheckpoint(
            filepath          = str(chemin_best),
            monitor           = "val_iou_score",
            mode              = "max",
            save_best_only    = True,
            verbose           = 1,
        ),
        ReduceLROnPlateau(
            monitor  = "val_iou_score",
            mode     = "max",
            factor   = 0.5,
            patience = patience_reducelr,
            min_lr   = 1e-6,
            verbose  = 1,
        ),
        MLflowLogger(),
    ]

    with mlflow.start_run(run_name=nom_run):
        mlflow.log_params({
            "nom_modele"       : modele.name,
            "nb_epochs_max"    : nb_epochs,
            "learning_rate"    : learning_rate,
            "batch_size"       : gen_train.batch_size,
            "taille_image"     : str(gen_train.taille_image),
            "augmentation"     : gen_train.augmenter,
            "nb_paires_train"  : len(gen_train.paires),
            "nb_paires_val"    : len(gen_val.paires),
            "loss"             : "dice_loss",
            "optimizer"        : "Adam",
        })

        historique = modele.fit(
            gen_train,
            validation_data       = gen_val,
            epochs                = nb_epochs,
            callbacks             = callbacks,
            workers               = 4,
            use_multiprocessing   = False,
            verbose               = 1,
        )

        mlflow.log_artifact(str(chemin_best), artifact_path="model")

    return historique
