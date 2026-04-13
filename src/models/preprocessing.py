import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


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
# URL LEXICAL FEATURE EXTRACTION
# ================================================
def _shannon_entropy(s: str) -> float:
    """Verilen string'in Shannon entropisini hesaplar (bit cinsinden)."""
    if not s:
        return 0.0
    from collections import Counter
    counts = Counter(s)
    total  = len(s)
    return -sum((c / total) * np.log2(c / total) for c in counts.values())


def extract_url_features(url: str) -> dict:
    url = str(url).strip()

    # ---- Temel binary bayraklar ----
    has_https         = int(url.startswith("https"))
    has_http          = int(url.startswith("http"))
    has_at            = int("@" in url)
    has_ip            = int(bool(re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url)))
    has_port          = int(bool(re.search(r":\d{2,5}(/|$)", url)))   # YENİ: :8080, :3000 vb.
    has_encoded_chars = int("%" in url)                                # YENİ: %20, %3D vb.

    # ---- Uzunluk ve karakter sayımı ----
    url_len      = len(url)
    dot_count    = url.count(".")
    slash_count  = url.count("/")
    dash_count   = url.count("-")
    digit_count  = sum(c.isdigit() for c in url)
    letter_count = sum(c.isalpha() for c in url)
    digit_ratio  = digit_count / max(url_len, 1)                      # YENİ: uzunluktan bağımsız oran

    # ---- Şüpheli kelime kontrolü (genişletilmiş) ----
    suspicious_words = [
        # Kimlik avı (phishing)
        "login", "signin", "sign-in", "logon", "verify", "validation",
        "confirm", "authenticate", "credential", "password", "passwd",
        # Finansal / aciliyet
        "bank", "secure", "security", "account", "update", "upgrade",
        "billing", "invoice", "payment", "paypal", "ebay", "amazon",
        # Tıklama tuzakları
        "free", "click", "prize", "winner", "lucky", "bonus", "offer",
        # Kötü amaçlı yazılım / phishing
        "malware", "virus", "hack", "crack", "keygen", "download",
        "exe", "zip", "rar",
    ]
    has_suspicious = int(any(w in url.lower() for w in suspicious_words))

    # ---- Alan adı ayrıştırma ----
    try:
        domain  = url.split("/")[2] if "//" in url else url.split("/")[0]
        # Port varsa temizle
        domain  = domain.split(":")[0]
        tld     = domain.split(".")[-1]
        tld_len = len(tld)
    except Exception:
        domain  = ""
        tld_len = 0

    try:
        domain_parts    = domain.split(".")
        subdomain_count = max(len(domain_parts) - 2, 0)
        domain_len      = len(domain)                                  # YENİ: alan adı uzunluğu
    except Exception:
        subdomain_count = 0
        domain_len      = 0

    # ---- Path ve sorgu ayrıştırma ----
    try:
        path_part   = url.split("?")[0]                                # sorgu öncesi kısım
        query_part  = url.split("?")[1] if "?" in url else ""
        path_depth  = max(path_part.count("/") - 2, 0)                # YENİ: path segmenti sayısı
        query_len   = len(query_part)                                  # YENİ: sorgu parametresi uzunluğu
    except Exception:
        path_depth = 0
        query_len  = 0

    # ---- Özel karakter oranı ----
    special_chars = sum(
        1 for c in url if not c.isalnum() and c not in [".", "/", "-", "_"]
    )
    special_ratio = special_chars / max(url_len, 1)

    # ---- Shannon entropisi ----
    entropy = _shannon_entropy(url)                                    # YENİ: karakter düzeyi rastgelelik

    return {
        # Uzunluk / sayım
        "url_len":         url_len,
        "dot_count":       dot_count,
        "slash_count":     slash_count,
        "dash_count":      dash_count,
        "digit_count":     digit_count,
        "letter_count":    letter_count,
        "digit_ratio":     digit_ratio,       # YENİ
        # Binary bayraklar
        "has_https":       has_https,
        "has_http":        has_http,
        "has_at":          has_at,
        "has_ip":          has_ip,
        "has_port":        has_port,          # YENİ
        "has_encoded_chars": has_encoded_chars,  # YENİ
        "has_suspicious":  has_suspicious,
        # Alan adı
        "tld_len":         tld_len,
        "subdomain_count": subdomain_count,
        "domain_len":      domain_len,        # YENİ
        # Path / sorgu
        "path_depth":      path_depth,        # YENİ
        "query_len":       query_len,         # YENİ
        # İstatistiksel
        "special_ratio":   special_ratio,
        "entropy":         entropy,           # YENİ
    }


def build_url_feature_dataframe(df: pd.DataFrame, url_column: str = "url") -> pd.DataFrame:
    if url_column not in df.columns:
        raise ValueError(f"URL sütunu '{url_column}' bulunamadı.")

    print("⏳ URL feature extraction çalışıyor...")

    # Feature'ları çıkar
    features_list = [extract_url_features(u) for u in df[url_column]]
    df_features   = pd.DataFrame(features_list, index=df.index)  # ← index eşitle

    # url_column dışındaki tüm sütunları doğrudan kopyala
    extra_cols = [c for c in df.columns if c != url_column]
    for col in extra_cols:
        df_features[col] = df[col].values  # ← .values ile index sorununu geç

    print(f"✅ Feature extraction tamamlandı: {df_features.shape}")
    print(f"   Sütunlar: {df_features.columns.tolist()}")
    return df_features


# ================================================
# STRATIFIED SAMPLER
# ================================================
def stratified_sample(
    df: pd.DataFrame,
    label_column: str,
    n_per_class: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Her sınıftan en fazla n_per_class örnek alır."""
    parts = []
    for cls in df[label_column].unique():
        group = df[df[label_column] == cls]
        parts.append(
            group.sample(n=min(n_per_class, len(group)), random_state=random_state)
        )
    return pd.concat(parts).reset_index(drop=True)


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

    id_candidates = ["id"]
    df = df.drop(columns=[c for c in id_candidates if c in df.columns], errors="ignore")

    label_series = df[label_column].astype(str)

    if positive_label is not None and negative_label is not None:
        mapping = {positive_label: 1, negative_label: 0}
        y = label_series.map(mapping)
        if y.isna().any():
            raise ValueError("Label mapping failed — check positive/negative labels.")
        y = y.values.astype(int)
    else:
        unique_vals = sorted(label_series.unique())
        auto_map    = {v: i for i, v in enumerate(unique_vals)}
        print(f"ℹ️  Otomatik etiket haritası: {auto_map}")
        y = label_series.map(auto_map).values

    feature_df = df.drop(columns=[label_column], errors="ignore")
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")

    nan_cols   = [c for c in feature_df.columns if feature_df[c].isna().all()]
    feature_df = feature_df.drop(columns=nan_cols, errors="ignore")
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    X = feature_df.values
    return X, y


# ================================================
# PREPROCESSOR  (PCA → MinMax → L2)
# ================================================
class Preprocessor:
    def __init__(
        self,
        apply_minmax: bool = True,
        apply_l2: bool = True,
        n_pca_components: Optional[int | float] = None,
    ):
        self.apply_minmax     = apply_minmax
        self.apply_l2         = apply_l2
        # int → sabit bileşen sayısı, float → hedef varyans oranı (ör. 0.95)
        self.n_pca_components = n_pca_components

        self.mm_scaler = None
        self.pca       = None

    def fit(self, X: np.ndarray):
        X_temp = X.copy()

        if self.n_pca_components is not None:
            self.pca = PCA(
                n_components=self.n_pca_components,
                whiten=True,
                random_state=42,
            )
            X_temp = self.pca.fit_transform(X_temp)

        if self.apply_minmax:
            self.mm_scaler = MinMaxScaler()
            self.mm_scaler.fit(X_temp)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_temp = X.copy()

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
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=(y if stratify else None),
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
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in skf.split(X, y):
        yield X[train_idx], X[test_idx], y[train_idx], y[test_idx]