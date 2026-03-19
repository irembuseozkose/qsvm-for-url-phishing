import os
import numpy as np
import pandas as pd
from typing import Tuple, Iterator, Optional
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ================================================
# L2 NORMALIZATION
# ================================================
def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return X / norm


# ================================================
# LOAD RAW DATA
# ================================================
def load_raw_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")
    if df.empty:
        raise ValueError("CSV is empty")

    return df


# ================================================
# UNIVERSAL FEATURE + LABEL EXTRACTOR
# ================================================
def prepare_features_and_labels(
    df: pd.DataFrame,
    label_column: str,
    positive_label: Optional[str] = None,
    negative_label: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataframe.")

    # 0) ID kolonlarını otomatik sil
    id_candidates = ["id"]
    df = df.drop(columns=[c for c in id_candidates if c in df.columns], errors="ignore")

    # 1) Labels
    label_series = df[label_column].astype(str)

    # Manual mapping
    if positive_label is not None and negative_label is not None:
        mapping = {positive_label: 1, negative_label: 0}
        y = label_series.map(mapping)

        if y.isna().any():
            raise ValueError("Label mapping failed — check positive/negative labels.")
        y = y.values.astype(int)

    else:  # Auto mapping for multi-class datasets
        unique_vals = sorted(label_series.unique())
        auto_map = {v: i for i, v in enumerate(unique_vals)}
        y = label_series.map(auto_map).values

    # 2) Features
    feature_df = df.drop(columns=[label_column], errors="ignore")
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")

    # Remove all-NaN columns
    nan_cols = [c for c in feature_df.columns if feature_df[c].isna().all()]
    feature_df = feature_df.drop(columns=nan_cols, errors="ignore")

    # Fill remaining NaNs
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    X = feature_df.values
    return X, y


# ================================================
# PREPROCESSOR (Correct: PCA → Scaler → L2)
# ================================================
class BreastCancerPreprocessor:
    def __init__(
        self,
        apply_minmax: bool = True,
        apply_standard: bool = False,
        apply_l2: bool = True,
        n_pca_components: Optional[int] = None,
    ):
        self.apply_minmax = apply_minmax
        self.apply_standard = apply_standard
        self.apply_l2 = apply_l2
        self.n_pca_components = n_pca_components

        self.std_scaler = None
        self.mm_scaler = None
        self.pca = None

    def fit(self, X: np.ndarray):
        X_temp = X

        # 1️⃣ StandardScaler (ÖNCE)
        if self.apply_standard:
            self.std_scaler = StandardScaler()
            X_temp = self.std_scaler.fit_transform(X_temp)

        # 2️⃣ PCA
        if self.n_pca_components is not None:
            self.pca = PCA(
                n_components=self.n_pca_components,
                whiten=True,
                random_state=42
            )
            X_temp = self.pca.fit_transform(X_temp)

        # 3️⃣ MinMax (SONRA)
        if self.apply_minmax:
            self.mm_scaler = MinMaxScaler()
            X_temp = self.mm_scaler.fit_transform(X_temp)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_temp = X

        # Aynı sıra korunmalı
        if self.std_scaler is not None:
            X_temp = self.std_scaler.transform(X_temp)

        if self.pca is not None:
            X_temp = self.pca.transform(X_temp)

        if self.mm_scaler is not None:
            X_temp = self.mm_scaler.transform(X_temp)

        if self.apply_l2:
            X_temp = l2_normalize_rows(X_temp)

        return X_temp

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)



# ================================================
# TRAIN/TEST SPLIT
# ================================================
def get_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
):
    stratify_arg = y if stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )


# ================================================
# K-FOLD SPLITS
# ================================================
def get_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
):
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    for train_idx, test_idx in skf.split(X, y):
        yield (
            X[train_idx],
            X[test_idx],
            y[train_idx],
            y[test_idx],
        )
