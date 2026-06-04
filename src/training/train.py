import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import prepare_data
from src.training.models import CPUForecastGRU


def train_model():
    # 1. Hyperparameter definieren
    SEQ_LENGTH = 12  # 1 Stunde Historie (aus der EDA ermittelt)
    BATCH_SIZE = 64
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    LR = 0.001
    EPOCHS = 15

    # Device-Auswahl: Nutzt die GPU deines M3 Macs (MPS) oder CPU falls nicht verfügbar
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training läuft auf Device: {device}")

    # 2. Daten laden (Hier nutzen wir das im Notebook erstellte Feature-Set)
    # Für ein sauberes Skript laden wir die Rohdaten und berechnen die Features kurz
    df = pd.read_csv(
        "../data/external/aws_cpu_utilization.csv", parse_dates=["timestamp"], index_col="timestamp"
    )

    # Schnelles Feature Engineering (identisch zum Notebook)
    df_enriched = df[["value"]].copy()
    import numpy as np

    df_enriched["hour_sin"] = np.sin(2 * np.pi * df_enriched.index.hour / 24.0)
    df_enriched["hour_cos"] = np.cos(2 * np.pi * df_enriched.index.hour / 24.0)
    df_enriched["rolling_mean_1h"] = df_enriched["value"].rolling(window=12).mean()
    df_enriched["rolling_std_1h"] = df_enriched["value"].rolling(window=12).std()
    df_enriched["rolling_mean_6h"] = df_enriched["value"].rolling(window=72).mean()
    df_enriched["rolling_std_6h"] = df_enriched["value"].rolling(window=72).std()
    df_enriched.dropna(inplace=True)

    # Datasets erstellen mittels data_loader.py
    train_dataset, test_dataset, _ = prepare_data(
        df_enriched, target_col="value", seq_length=SEQ_LENGTH
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False
    )  # Keine Shuffles bei Zeitreihen!
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Modell instanziieren
    input_dim = df_enriched.shape[1] - 1  # Alle Spalten außer 'value'
    model = CPUForecastGRU(
        input_dim=input_dim, hidden_dim=HIDDEN_DIM, output_dim=1, num_layers=NUM_LAYERS
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 4. Trainings-Schleife
    print("\n--- Starte Training ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward Pass
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)

            # Backward Pass & Optimierung
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # Validierung pro Epoche
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                test_loss += loss.item() * batch_x.size(0)
        test_loss /= len(test_loader.dataset)

        print(
            f"Epoch [{epoch}/{EPOCHS}] -> Train Loss (MSE): {train_loss:.4f} | Test Loss (MSE): {test_loss:.4f}"
        )

    # 5. Finale Evaluation (Metriken berechnen)
    print("\n--- Abschließende Evaluation ---")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).squeeze()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets)) ** 2))

    print("🏆 Finale Test-Metriken:")
    print(f"   -> Mean Absolute Error (MAE): {mae:.2f}% CPU")
    print(f"   -> Root Mean Squared Error (RMSE): {rmse:.2f}% CPU")


if __name__ == "__main__":
    train_model()
