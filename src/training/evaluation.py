import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import Config
from src.utils.logger import setup_logger
from src.data import get_data_loaders
from src.training.models import CPUForecastGRU

logger = setup_logger(__name__)


def evaluate_and_plot(config: Config):
    """Evaluiert das trainierte GRU-Modell auf dem Test-Set, berechnet die

    finalen Metriken und speichert das Vorhersage-Dashboard ab.
    """
    logger.info("Starte abschließende Test-Evaluation...")

    # 1. Hardware-Device bestimmen und Test-Loader anfordern
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _, _, test_loader = get_data_loaders(config)

    # 2. Input-Dimension dynamisch bestimmen
    sample_x, _ = next(iter(test_loader))
    input_dim = sample_x.shape[2]

    # 3. Modell instanziieren
    model = CPUForecastGRU(
        input_dim=input_dim,
        hidden_dim=config.model["hidden_dim"],
        num_layers=config.model["num_layers"],
    ).to(device)

    # 4. Pfadsicheres Laden des Modells relativ zu DIESER Datei
    # __file__ ist 'src/training/evaluation.py'. parents[2] wandert hoch zu 'infra-insight/'
    project_root = Path(__file__).resolve().parents[2]
    checkpoint_path = project_root / "notebooks" / "models" / "best_model.pt"

    if not checkpoint_path.exists():
        logger.error(f"Modell-Checkpoint nicht gefunden unter: {checkpoint_path}")
        raise FileNotFoundError(
            f"Bitte trainiere zuerst das Modell, um {checkpoint_path.name} zu erzeugen."
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    logger.info(f"🎯 Bestes Modell erfolgreich geladen aus: {checkpoint_path}")

    # 5. Inferenz auf dem Test-Set
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x).squeeze(1)

            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # 6. Metriken berechnen
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))

    logger.info(f"🏆 Finale Test-Metriken: MAE = {mae:.2f}% CPU | RMSE = {rmse:.2f}% CPU")

    # 7. Dashboard visualisieren und pfadsicher speichern
    plt.figure(figsize=(14, 6))
    plt.plot(targets, label="Actual CPU Utilization", color="royalblue", alpha=0.8, linewidth=2)
    plt.plot(
        preds,
        label="Predicted CPU (GRU Baseline)",
        color="crimson",
        linestyle="--",
        alpha=0.9,
        linewidth=2,
    )

    plt.title(
        "Infrastructure Forecast Dashboard: Actual vs. Predicted CPU",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Time Steps (5-Minute Intervals on Test Set)")
    plt.ylabel("CPU Utilization (%)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)

    output_plot_path = project_root / "notebooks" / "predictions_vs_actual.png"
    plt.savefig(output_plot_path, bbox_inches="tight")
    plt.show()

    logger.info(f"📊 Dashboard-Plot erfolgreich gesichert unter: {output_plot_path}")

    return mae, rmse
