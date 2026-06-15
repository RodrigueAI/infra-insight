import pickle
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
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
    """Liest die von Spark fertig partitionierten Parquet-Splits ein,

    extrahiert die dichten Vektoren, skaliert leckagefrei und speichert
    die finalen .npz-Dateien für PyTorch.
    """
    logger.info("⚡ Starte dateibasierten PySpark-to-PyTorch Tensor-Export Layer...")

    project_root = Path(__file__).resolve().parents[2]
    spark_dir = project_root / "data" / "processed_spark"
    processed_dir = project_root / config.paths["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Daten direkt aus den von Spark geschriebenen Splits laden
    df_train = pq.read_table(spark_dir / "train.parquet").to_pandas().sort_values("timestamp")
    df_val = pq.read_table(spark_dir / "val.parquet").to_pandas().sort_values("timestamp")
    df_test = pq.read_table(spark_dir / "test.parquet").to_pandas().sort_values("timestamp")

    # CRITICAL FIX: Da es sich nun um ein echtes Array handelt,
    # konvertieren wir es sauber in eine Matrix
    X_train_raw = np.array(df_train["feature_array"].tolist())
    X_val_raw = np.array(df_val["feature_array"].tolist())
    X_test_raw = np.array(df_test["feature_array"].tolist())

    y_train = df_train["cpu_utilization"].values
    y_val = df_val["cpu_utilization"].values
    y_test = df_test["cpu_utilization"].values

    if len(X_train_raw.shape) == 1:
        feature_dim = 1
    else:
        feature_dim = X_train_raw.shape[1]

    logger.info(f"Vektoren extrahiert. Feature-Dimension: {feature_dim}")
    logger.info(f"Vektoren extrahiert. Feature-Dimension: {X_train_raw.shape[1]}")

    # 3. Skalierung (Fit AUSSCHLIESSLICH auf Train, um Data Leakage zu verhindern)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 4. Endgültige Persistierung für das Training
    np.savez(processed_dir / "train_data.npz", X=X_train_scaled, y=y_train)
    np.savez(processed_dir / "val_data.npz", X=X_val_scaled, y=y_val)
    np.savez(processed_dir / "test_data.npz", X=X_test_scaled, y=y_test)

    with open(processed_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    logger.info("✅ End-to-End Export-Layer erfolgreich abgeschlossen.")


def load_prepared_datasets(config: Config):
    """Lädt die fertig transformierten multivariaten Daten von der Festplatte."""
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / config.paths["processed_dir"]

    train_data = np.load(processed_dir / "train_data.npz")
    val_data = np.load(processed_dir / "val_data.npz")
    test_data = np.load(processed_dir / "test_data.npz")

    seq_len = config.data_split["sequence_length"]

    return (
        TimeSeriesDataset(train_data["X"], train_data["y"], seq_len),
        TimeSeriesDataset(val_data["X"], val_data["y"], seq_len),
        TimeSeriesDataset(test_data["X"], test_data["y"], seq_len),
    )


def get_data_loaders(config: Config):
    """Erzeugt einsatzbereite DataLoader für das Modell-Training."""
    train_dataset, val_dataset, test_dataset = load_prepared_datasets(config)
    batch_size = config.training["batch_size"]

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )
