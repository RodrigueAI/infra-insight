import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, df, target_col='value', seq_length=12):
        self.seq_length = seq_length
        
        # Features und Target trennen
        X_raw = df.drop(columns=[target_col]).values
        y_raw = df[target_col].values
        
        # WICHTIG: Skalierung der Features für stabile Gradienten im Deep Learning
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw)
        
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.y = torch.tensor(y_raw, dtype=torch.float32)

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        # Extrahiere das historische Fenster der Länge `seq_length`
        x_seq = self.X[idx : idx + self.seq_length]
        # Das Target ist der Wert direkt NACH dem Fenster
        y_target = self.y[idx + self.seq_length]
        
        return x_seq, y_target

# Instanziieren mit einem 1-Stunden-Fenster (12 Schritte à 5 Min)
dataset = TimeSeriesDataset(features_df, seq_length=12)
print(f"Anzahl der verfügbaren Sequenzen: {len(dataset)}")

# Test-Check einer einzelnen Sequenz
sample_x, sample_y = dataset[0]
print(f"Tensor-Form der Eingangssequenz (Seq_Length, Features): {sample_x.shape}")
print(f"Target-Wert (Skalar): {sample_y.item()}")