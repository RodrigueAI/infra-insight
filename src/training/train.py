import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path

from src.config import Config
from src.data import get_data_loaders
from src.training.models import CPUForecastGRU
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Nutze Hardware-Beschleunigung: {self.device}")

        # 1. DataLoader anfordern
        self.train_loader, self.val_loader, _ = get_data_loaders(config)

        # 2. Modell dynamisch instanziieren
        # INPUT_DIM extrahieren wir direkt aus dem ersten Batch des Loaders
        sample_x, _ = next(iter(self.train_loader))
        input_dim = sample_x.shape[2]

        self.model = CPUForecastGRU(
            input_dim=input_dim,
            hidden_dim=config.model["hidden_dim"],
            num_layers=config.model["num_layers"],
            dropout=config.model["dropout"],
        ).to(self.device)

        # 3. Loss & Optimizer aus Config ableiten
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.training["learning_rate"]
        )

        # Speicher für Loss-Historie (für spätere Plots)
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0

        for batch_x, batch_y in self.train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # Forward Pass
            outputs = self.model(batch_x).squeeze(1)
            loss = self.criterion(outputs, batch_y)

            # Backward Pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

        return running_loss / len(self.train_loader.dataset)

    def validate(self) -> float:
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x).squeeze()
                loss = self.criterion(outputs, batch_y)
                running_loss += loss.item() * batch_x.size(0)

        return running_loss / len(self.val_loader.dataset)

    def fit(self):
        epochs = self.config.training["epochs"]
        best_val_loss = float("inf")
        checkpoint_dir = Path("models")
        checkpoint_dir.mkdir(exist_ok=True)

        logger.info(f"Starte Training für {epochs} Epochen...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            # Checkpoint speichern, wenn sich das Modell auf den Validierungsdaten verbessert
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), checkpoint_dir / "best_model.pt")
                logger.info(f"💾 Neues bestes Modell gespeichert (Val Loss: {best_val_loss:.4f})")

        self.plot_losses()

    def plot_losses(self):
        """Generiert den Loss-Plot für das GitHub README mittels Pathlib."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss (MSE)", color="royalblue", linewidth=2)
        plt.plot(self.val_losses, label="Validation Loss (MSE)", color="orange", linewidth=2)
        plt.title("Model Training & Validation Loss Trend", fontsize=12, fontweight="bold")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        # Absolute Pfadbestimmung ausgehend von dieser Datei (train.py)
        base_dir = Path(__file__).resolve().parents[2]  # Wandert hoch zu 'infra-insight/'
        plot_path = base_dir / "notebooks" / "loss_curve.png"

        # Ordner erstellen, falls er gelöscht wurde
        plot_path.parent.mkdir(exist_ok=True)

        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        logger.info(f"📊 Loss-Kurve erfolgreich unter '{plot_path}' gespeichert.")


if __name__ == "__main__":
    # Ermöglicht das direkte Ausführen des Skripts per Terminal
    config = Config()
    trainer = Trainer(config)
    trainer.fit()
