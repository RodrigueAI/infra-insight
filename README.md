# InfraInsight: Predictive System Analytics with PyTorch

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI/CD: Code Quality](https://github.com/rodrigueai/infra-insight/actions/workflows/ci.yml/badge.svg)](https://github.com/DEIN-GITHUB-BENUTZERNAME/infra-insight/actions)

An end-to-end Machine Learning pipeline designed to forecast infrastructure resource utilization (e.g., CPU, RAM, Network I/O). By leveraging deep learning sequentially, this project proactively detects potential system bottlenecks or over-provisioning before they affect production environments.

This repository serves as a professional showcase demonstrating strict software engineering principles applied to Data Science: explicit environment management, modular code design, centralized logging, and automated CI/CD quality gates.

👉 **Detailed technical architectural documentation, mathematical features, and pipeline internals can be found in the [Project Wiki](../../wiki).**

---

## 📊 Current Baseline Performance

The pipeline currently runs an optimized **Gated Recurrent Unit (GRU)** network utilizing Apple Silicon hardware acceleration (`mps`).

* **Lookback-Window:** 12 steps (1 hour historical context)
* **Mean Absolute Error (MAE):** `3.19%` CPU Deviation on Test Set
* **Root Mean Squared Error (RMSE):** `5.61%` CPU Deviation on Test Set

| Training Loss Trend | Actual vs. Predicted CPU |
|---|---|
| ![Loss Curve](notebooks/loss_curve.png) | ![Predictions](notebooks/predictions_vs_actual.png) |

---

## 🛠️ Tech Stack & Production Standards

* **Core Frameworks:** PyTorch (Deep Learning), Scikit-Learn (Preprocessing)
* **Data Engineering:** Pandas & NumPy
* **Quality & Automation:** GitHub Actions (CI), Black Formatter, Ruff Linter
* **Architecture:** Explicit YAML configurations (`config.yaml`), production-grade logging setups.

---

## 🚀 Getting Started

### Prerequisites
Ensure **Miniforge** or **Anaconda** is installed on your local machine.

### 1. Setup Environment
Clone the repository and reproduce the exact environment using the locked Conda specification:
```bash
conda env create -f environment.yml
conda activate infra-insight
