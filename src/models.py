import torch
import torch.nn as nn


class CPUForecastGRU(nn.Module):
    """Gated Recurrent Unit (GRU) Network für Zeitreihen-Vorhersage.

    Erwartet Input der Form: (Batch_Size, Sequence_Length, Input_Dim)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, num_layers: int = 2, dropout: float = 0.2):
        super(CPUForecastGRU, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU Layer: Verarbeitet die zeitliche Sequenz
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # Garantiert die Shape (Batch, Seq, Features)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Fully Connected Layer: Mappt das Hidden State des letzten Zeitschritts auf den Vorhersagewert
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialisierung der Hidden States mit Nullen
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward Pass durch die GRU
        # out Shape: (Batch_Size, Sequence_Length, Hidden_Dim)
        out, _ = self.gru(x, h0)

        # Wir wollen nur den letzten Zeitschritt (-1) der Sequenz vorhersagen
        # out[:, -1, :] Shape: (Batch_Size, Hidden_Dim)
        out = self.fc(out[:, -1, :])

        return out