import pandas as pd
import numpy as np
from keras.layers import Dropout, Dense
from keras import Input, Model
from keras.optimizers import Adam
from keras.losses import BinaryFocalCrossentropy
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from vit_keras import vit

from data import make_dataset_from_csv, LABEL_TO_ID, IMG_SIZE


def compute_class_weight_from_df(train_df: pd.DataFrame) -> dict[int, float]:
    y = train_df["label"].map(LABEL_TO_ID).to_numpy()
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    total = n0 + n1
    # Inverse frequency weighting: higher weight for the minority class
    w0 = total / (2.0 * n0) if n0 else 1.0
    w1 = total / (2.0 * n1) if n1 else 1.0
    return {0: float(w0), 1: float(w1)}

if __name__ == "__main__":
    train_ds, train_df = make_dataset_from_csv("train.csv", training=True)
    valid_ds, valid_df = make_dataset_from_csv("valid.csv", training=False)

    class_weight = compute_class_weight_from_df(train_df)
    print("class_weight:", class_weight)

    backbone = vit.vit_b16(
        classes=2,
        include_top=False,
        pretrained_top=False
    )
    backbone.trainable = False

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=False)  # x shape: (None, 768)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(), #1e-4
        loss=BinaryFocalCrossentropy(),
        metrics=[
            BinaryAccuracy(name="acc"),
            AUC(name="auc"),
            Precision(name="precision"),
            Recall(name="recall")
        ],
    )

    model.summary()

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=10,
        class_weight=class_weight,
    )