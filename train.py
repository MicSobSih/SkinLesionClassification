from pathlib import Path

import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.losses import BinaryFocalCrossentropy
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC, SpecificityAtSensitivity
from keras.callbacks import EarlyStopping, ModelCheckpoint

from data import make_dataset_from_csv, LABEL_TO_ID, IMG_SIZE
from model import ViTSkinLesionModel
from plots import save_metric_plots


def compute_class_weight_from_df(train_df: pd.DataFrame) -> dict[int, float]:
    y = train_df["label"].map(LABEL_TO_ID).to_numpy()
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    total = n0 + n1
    # Inverse frequency weighting: higher weight for the minority class
    w0 = total / (2.0 * n0) if n0 else 1.0
    w1 = total / (2.0 * n1) if n1 else 1.0
    class_weights = {0: float(w0), 1: float(w1)}
    return class_weights

if __name__ == "__main__":
    train_ds, train_df = make_dataset_from_csv("train.csv", training=True)
    valid_ds, valid_df = make_dataset_from_csv("valid.csv", training=False)

    class_weight = compute_class_weight_from_df(train_df)
    print("class_weight:", class_weight)

    model = ViTSkinLesionModel(IMG_SIZE, dropout=0.2).build()

    model.compile(
        optimizer=Adam(), #1e-4
        loss=BinaryFocalCrossentropy(),
        metrics=[
            BinaryAccuracy(name="acc"),
            AUC(name="auc", curve="PR"),
            Precision(name="precision"),
            Recall(name="recall"),
            SpecificityAtSensitivity(name="specificity", sensitivity=0.9)
        ],
    )

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_checkpoint = ModelCheckpoint(
        filepath=models_dir / "ViT-val_auc.keras",
        monitor="val_auc",
        mode="max",
        verbose=1,
        save_best_only=True
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        verbose=1,
        restore_best_weights=True
    )

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=5,
        class_weight=class_weight,
        callbacks=[model_checkpoint, early_stopping]
    )

    save_metric_plots(history)