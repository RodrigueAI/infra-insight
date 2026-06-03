# InfraInsight: Predictive System Analytics with PyTorch

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end Machine Learning pipeline designed to forecast infrastructure resource utilization (e.g., CPU, RAM, Network I/O). By leveraging deep learning sequentially, this project aims to proactively detect potential system bottlenecks or over-provisioning before they affect production environments.

This repository serves as a professional showcase demonstrating strict software engineering principles applied to Data Science: explicit environment management, modular code design, and mathematically sound time-series validation.

---

## 🏗️ Project Architecture & Workflow

The project is structured to transition smoothly from exploratory notebooks to production-ready, modular Python scripts.

```text
infra-insight/
├── environment.yml        # Strict Conda environment specification
├── pyproject.toml         # Tooling configuration (Black, Ruff, etc.)
├── README.md              # Project documentation
├── data/                  # Local data storage (Git ignored)
├── notebooks/             # Exploratory phase
│   ├── 01_eda.ipynb       # Time-series analysis, ACF/PACF plots
│   └── 02_training.ipynb  # Interactive model prototyping
└── src/                   # Production-grade modular source code
    ├── __init__.py
    ├── data_loader.py     # Custom PyTorch SlidingWindowDataset
    ├── models.py          # PyTorch Recurrent Architectures (GRU/LSTM)
    └── train.py           # Training loop & validation execution
```

# 🛠️ Tech Stack & Rigorous Standards

## Deep Learning Framework
- **PyTorch** für den Aufbau und das Training benutzerdefinierter rekurrenter neuronaler Netze.

## Data Wrangling
- **Pandas** & **NumPy** für Matrixoperationen und performante Feature-Berechnung.

## Visualization
- **Matplotlib** & **Seaborn** zur Analyse statistischer Verteilungen und Fehlermetriken.

## Environment Management
- Kein loses `requirements.txt`.
- Abhängigkeiten werden strikt über ein natives Conda-`environment.yml` verwaltet, um deterministische Builds sicherzustellen (inkl. CUDA-/MPS-Konfigurationen).
- Ergänzt durch ein `pyproject.toml` für standardisierte Linter- und Tool-Konfigurationen.

---

# 📊 Machine Learning Pipeline

## 1. Exploratory Data Analysis (EDA)

### Trend & Seasonality
Zerlegung von Metriken zur Isolierung täglicher und wöchentlicher Infrastrukturzyklen.

### Autocorrelation
Verwendung von **ACF**- und **PACF**-Plots zur mathematischen Bestimmung optimaler Sequenzlängen (Lags), anstatt diese heuristisch festzulegen.

---

## 2. Time-Series Feature Engineering

Um komplexe Muster effizient zu erlernen, werden folgende Features dynamisch erzeugt:

### Lag Features
Historische Verschiebungen:

$$
t-1,\; t-2,\; \dots,\; t-n
$$

### Rolling Statistics
- Gleitende Mittelwerte (*Rolling Averages*)
- Rollierende Standardabweichungen

über dynamische Fenster zur Erfassung von Volatilität.

### Temporal Context
Zyklische Kodierung zeitlicher Variablen:

- Stunde des Tages
- Wochentag

---

## 3. PyTorch Dataset & Modeling

### SlidingWindowDataset

Eine benutzerdefinierte Implementierung von `torch.utils.data.Dataset`, die 2D-tabellarische Infrastruktur-Logs in strukturierte 3D-Tensoren transformiert:

$$
\text{Shape: } (\text{Batch Size}, \text{Sequence Length}, \text{Features})
$$

### Architektur

Verwendung eines **GRU-** oder **LSTM-Netzwerks**, um langfristige Abhängigkeiten im Systemverhalten zu modellieren und gleichzeitig das Problem verschwindender Gradienten zu minimieren.

---

## 4. Evaluation Strategy

### Strict Time-Splitting

Standardmäßige **K-Fold Cross-Validation** führt bei Zeitreihendaten zu schwerwiegendem **Data Leakage**.

Dieses Projekt erzwingt daher:

- Chronologische Train/Test-Splits
- Alternativ `TimeSeriesSplit`

Dadurch wird garantiert, dass das Modell niemals auf zukünftigen Informationen trainiert wird.

### Metriken

Bewertung mittels:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

---

# 🚀 Getting Started

## Prerequisites

Stelle sicher, dass **Miniforge** oder **Anaconda** installiert ist.

## Setup Environment

Repository klonen und die exakte Umgebung mittels Conda reproduzieren:

```bash
# Create the environment from the locked specification
conda env create -f environment.yml

# Activate the project environment
conda activate infra-insight
```

## Running the Pipeline

Die Entwicklungsschritte können innerhalb des Verzeichnisses `notebooks/` nachvollzogen werden oder direkt über die modulare Trainings-Pipeline ausgeführt werden:

```bash
python src/train.py
```

---

# 🔮 Future Roadmap

Die Architektur wurde explizit für Skalierbarkeit entwickelt.

Geplante Erweiterungen umfassen:

- Migration des Feature-Engineering-Layers nach **Apache Spark (PySpark)**
- Persistierung der Daten in einem **Data Vault 2.0** Schema
- Enterprise-taugliche Nachvollziehbarkeit (*Auditability*)
- Skalierbare Batch-Verarbeitung großer Infrastrukturdatenmengen
- Verbesserte Governance- und Compliance-Fähigkeiten
