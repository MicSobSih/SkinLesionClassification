import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

IMG_SIZE = 224
BATCH_SIZE = 32

LABEL_TO_ID = {"benign": 0, "malignant": 1}


def _decode_and_resize(path: tf.Tensor) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    return img


def _preprocess_for_vit(img: tf.Tensor) -> tf.Tensor:
    # Common for ViT: scale to [0, 1]. (You can change this if your ViT expects different normalization.)
    img = img / 255.0
    return img


def _load_example(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    img = _decode_and_resize(path)
    img = _preprocess_for_vit(img)
    # For binary classification with sigmoid output, label should be float32 shape (1,) or scalar.
    label = tf.cast(label, tf.float32)
    return img, label


def make_dataset_from_csv(csv_path: str, training: bool) -> tuple[tf.data.Dataset, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    # Expect columns: image_path, label (string)
    paths = df["image_path"].astype(str).to_numpy()
    labels = df["label"].map(LABEL_TO_ID).to_numpy()

    if pd.isna(labels).any():
        bad = df[pd.isna(df["label"].map(LABEL_TO_ID))]["label"].unique().tolist()
        raise ValueError(f"Unknown labels in CSV: {bad}")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df))

    ds = ds.map(_load_example, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds, df