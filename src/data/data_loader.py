import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.config import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TimeSeriesDataset(Dataset):
    """Custom PyTorch Dataset für die On-the-fly-Sequenzgenerierung (Sliding Window)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_length: int = 12):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_length]
        y_target = self.y[idx + self.seq_length]
        return x_seq, y_target


def process_and_split_data(config: Config):
    """Lädt Rohdaten, generiert Features, führt einen 3-Wege-Split ohne Shuffling durch,

    skaliert die Daten ohne Leakage und speichert die Ergebnisse ab.
    """
    logger.info("Starte Datenaufbereitung und 3-Wege-Split...")

    # 1. Rohdaten laden mit pathlib
    raw_path = Path(config.paths["raw_data_path"])
    if not raw_path.exists():
        # Pfadkorrektur für Aufrufe aus verschiedenen Ebenen (ersetzt "../../")
        raw_path = Path(__file__).resolve().parents[2] / raw_path

    df = pd.read_csv(raw_path, parse_dates=["timestamp"], index_col="timestamp")
    df_sorted = df.sort_index()

    # 2. Feature Expansion (identisch zu deinen EDA-Erkenntnissen)
    df_enriched = df_sorted[["value"]].copy()
    df_enriched["hour_sin"] = np.sin(2 * np.pi * df_enriched.index.hour / 24.0)
    df_enriched["hour_cos"] = np.cos(2 * np.pi * df_enriched.index.hour / 24.0)
    df_enriched["rolling_mean_1h"] = df_enriched["value"].rolling(window=12).mean()
    df_enriched["rolling_std_1h"] = df_enriched["value"].rolling(window=12).std()
    df_enriched["rolling_mean_6h"] = df_enriched["value"].rolling(window=72).mean()
    df_enriched["rolling_std_6h"] = df_enriched["value"].rolling(window=72).std()
    df_enriched.dropna(inplace=True)

    # Features und Target trennen
    target_col = config.data_split["target_column"]
    X_raw = df_enriched.values
    y_raw = df_enriched[target_col].values

    # 3. Zeitreihenkonformer 3-Wege-Split (Train / Val / Test)
    total_len = len(df_enriched)
    train_end = int(total_len * config.data_split["train_ratio"])
    val_end = train_end + int(total_len * config.data_split["val_ratio"])

    X_train_raw, y_train = X_raw[:train_end], y_raw[:train_end]
    X_val_raw, y_val = X_raw[train_end:val_end], y_raw[train_end:val_end]
    X_test_raw, y_test = X_raw[val_end:], y_raw[val_end:]

    logger.info(
        "Split-Verhältnis: "
        f"Train={len(X_train_raw)} | Val={len(X_val_raw)} | Test={len(X_test_raw)}"
    )

    # 4. Skalierung (Fit AUSSCHLIESSLICH auf Train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 5. Transformierte Daten auf die Festplatte speichern (Persistierung)
    processed_dir = Path(config.paths["processed_dir"])
    if not processed_dir.exists():
        processed_dir = Path(__file__).resolve().parents[2] / processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Speichern über den eleganten / Operator von pathlib
    np.savez(processed_dir / "train_data.npz", X=X_train_scaled, y=y_train)
    np.savez(processed_dir / "val_data.npz", X=X_val_scaled, y=y_val)
    np.savez(processed_dir / "test_data.npz", X=X_test_scaled, y=y_test)

    # Scaler via pickle sichern
    with open(processed_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Lokaler Import, um globalen Scope sauber zu halten
    import os

    logger.info(f"✅ Alle Daten erfolgreich in '{os.path.relpath(processed_dir)}' persistiert.")


def load_prepared_datasets(config: Config):
    """Lädt die fertig transformierten Daten von der Festplatte

    und gibt einsatzbereite TimeSeriesDataset-Instanzen zurück.
    """
    processed_dir = Path(config.paths["processed_dir"])
    if not processed_dir.exists():
        processed_dir = Path(__file__).resolve().parents[2] / processed_dir

    train_data = np.load(processed_dir / "train_data.npz")
    val_data = np.load(processed_dir / "val_data.npz")
    test_data = np.load(processed_dir / "test_data.npz")

    seq_len = config.data_split["sequence_length"]

    train_dataset = TimeSeriesDataset(train_data["X"], train_data["y"], seq_len)
    val_dataset = TimeSeriesDataset(val_data["X"], val_data["y"], seq_len)
    test_dataset = TimeSeriesDataset(test_data["X"], test_data["y"], seq_len)

    return train_dataset, val_dataset, test_dataset


def get_data_loaders(config: Config):
    """Erzeugt fertige, direkt einsatzbereite DataLoader für das Modelltraining

    gemäß den Parametern aus der config.yaml.
    """
    logger.info("Generiere PyTorch DataLoader für Train, Val und Test...")

    # 1. Fertig präparierte Datasets laden
    train_dataset, val_dataset, test_dataset = load_prepared_datasets(config)

    # 2. DataLoader instanziieren
    batch_size = config.training["batch_size"]

    # CRITICAL: shuffle=False bei ALLEN Loadern, da es sich um eine fortlaufende Zeitreihe handelt!
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    logger.info(
        f"✅ DataLoader erfolgreich erstellt. Batches pro Epoche: "
        f"Train={len(train_loader)} | Val={len(val_loader)} | Test={len(test_loader)}"
    )

    return train_loader, val_loader, test_loader
