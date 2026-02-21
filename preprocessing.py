from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


MALIGNANT_DX = {"mel", "bcc", "akiec"}

def _drop_inconsistent_lesions_binary(meta: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    n_unique = meta.groupby("lesion_id")[label_col].nunique()
    bad_lesion_ids = set(n_unique[n_unique > 1].index.tolist())
    return meta.loc[~meta["lesion_id"].isin(bad_lesion_ids)].copy()


def _find_image_path(image_id: str, folders: list[Path], ext: str = ".jpg") -> str | None:
    fname = f"{image_id}{ext}"
    for folder in folders:
        path = folder / fname
        if path.exists():
            return str(path)
    return None


# Helper class to return train/valid split paths
# Set attributes finally to avoid accidental mutation
@dataclass(frozen=True)
class SplitPaths:
    train_csv: Path
    valid_csv: Path


def make_grouped_stratified_split(
    meta_csv: Path,
    image_folders: list[Path],
    out_train_csv: Path,
    out_valid_csv: Path,
    split: float = 0.8,
    seed: int = 42,
) -> SplitPaths:
    meta = pd.read_csv(meta_csv)

    # Check that all necessary columns are present
    required = {"image_id", "lesion_id", "dx"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata CSV is missing columns: {sorted(missing)}")

    # Map each metadata row to an actual file path in either images_part_1 or images_part_2
    meta["image_path"] = meta["image_id"].astype(str).apply(
        lambda iid: _find_image_path(iid, image_folders)
    )
    meta = meta.dropna(subset=["image_path"]).copy()

    # Binary label
    meta["label"] = meta["dx"].astype(str).apply(
        lambda d: "malignant" if d in MALIGNANT_DX else "benign"
    )

    meta = meta.dropna(subset=["label"]).copy()
    meta = _drop_inconsistent_lesions_binary(meta, label_col="label")

    # Build lesion-level table (assumes one label per lesion; if not, raise an error)
    lesion = (
        meta.groupby("lesion_id", as_index=False)
        .agg(
            lesion_label=("label", lambda s: s.iloc[0] if s.nunique(dropna=False) == 1 else (_ for _ in ()).throw(
                ValueError("Inconsistent lesion label"))),
            n_images=("image_id", "count"),
        )
    )

    rng = np.random.default_rng(seed)

    train_lesions: list[str] = []
    valid_lesions: list[str] = []

    # Stratify at lesion level
    for lbl, sub in lesion.groupby("lesion_label"):
        lesion_ids = sub["lesion_id"].to_numpy()
        rng.shuffle(lesion_ids)

        n_train = int(round(split * len(lesion_ids)))
        train_lesions.extend(lesion_ids[:n_train].tolist())
        valid_lesions.extend(lesion_ids[n_train:].tolist())

    train_lesions = set(train_lesions)
    valid_lesions = set(valid_lesions)

    # Safety: ensure no overlap (should be none)
    overlap = train_lesions & valid_lesions
    if overlap:
        raise RuntimeError(f"Lesion overlap between train/valid: {len(overlap)} lesions")

    train_df = meta[meta["lesion_id"].isin(train_lesions)].copy()
    valid_df = meta[meta["lesion_id"].isin(valid_lesions)].copy()

    # Keep only what you need downstream (add more columns if useful)
    keep_cols = ["image_id", "lesion_id", "dx", "label", "image_path"]
    train_df[keep_cols].to_csv(out_train_csv, index=False)
    valid_df[keep_cols].to_csv(out_valid_csv, index=False)

    return SplitPaths(train_csv=out_train_csv, valid_csv=out_valid_csv)


if __name__ == "__main__":
    base = Path("dataverse_files")
    meta_csv = base / "HAM10000_metadata"

    image_folders = [
        base / "HAM10000_images_part_1",
        base / "HAM10000_images_part_2",
    ]

    paths = make_grouped_stratified_split(
        meta_csv=meta_csv,
        image_folders=image_folders,
        out_train_csv=Path("train.csv"),
        out_valid_csv=Path("valid.csv"),
        split=0.8,
        seed=42,
    )
    print(paths)