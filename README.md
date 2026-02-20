# SAFE — Secure Anomaly Detection Edge AI System for Critical Environments

<p align="center">
  <strong>End-to-end anomaly prediction for industrial rotating machinery — from raw sensors to INT8-quantized edge inference.</strong>
</p>

<p align="center">
  <a href="#architecture">Architecture</a> •
  <a href="#modules">Modules</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#edge-deployment">Edge Deployment</a> •
  <a href="#license">License</a>
</p>

---

## Overview

**SAFE** is a research project developed by [IFAB Foundation](https://www.intfrancesintetica.it/) in collaboration with the **University of Bologna**, **Bonfiglioli S.p.A.**, **Tampieri S.p.A.**, and **SECO S.p.A.**

The system covers the full lifecycle of predictive maintenance on industrial rotating machinery:

1. **Data preprocessing** — sensor signal alignment, feature engineering, cross-domain (lab ↔ field) feature matching.
2. **Multi-model forecasting** — deep learning (CNN, LSTM, CNN-LSTM, TCN, Transformer) and classical/ML models (ARIMA, Prophet, VAR, XGBoost, LightGBM, Random Forest), with Optuna hyperparameter optimization.
3. **Multi-paradigm anomaly detection** — forecast-residual methods (Z-score, MAD, CUSUM, EWMA, Isolation Forest, LOF, composite Health Index), graph-theoretic analysis (Natural/Horizontal Visibility Graphs, OddBall, community detection, centrality metrics), and cross-domain validation pipelines.
4. **Edge deployment** — full INT8 quantization via TFLite, firmware for **STM32** (Cortex-M33, X-CUBE-AI) and **ESP32** (TFLite Micro) microcontrollers, with Modbus RTU industrial communication.

> **Note:** This repository contains **source code only**. Data, trained models, notebooks, and result figures are not included as they were produced under an industrial partnership agreement.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────────┐     ┌───────────────────┐
│  Preprocessing   │ ──▶ │    Forecasting    │ ──▶ │   Anomaly Detection   │ ──▶ │  Edge Deployment  │
│                  │     │                   │     │                       │     │                   │
│ • Data loading   │     │ • CNN / LSTM      │     │ • Forecast-residual   │     │ • INT8 quantize   │
│ • Feature eng.   │     │ • CNN-LSTM / TCN  │     │ • Graph-based (VG)    │     │ • STM32 firmware  │
│ • Feature match  │     │ • Transformer     │     │ • OddBall / community │     │ • ESP32 firmware  │
│ • Domain adapt.  │     │ • ARIMA / Prophet  │     │ • Centrality analysis │     │ • Modbus RTU      │
│                  │     │ • XGB / LightGBM  │     │ • Cross-domain test   │     │ • Validation      │
└─────────────────┘     └──────────────────┘     └───────────────────────┘     └───────────────────┘
```

---

## Repository Structure

```
safe/
├── src/
│   ├── preprocessing/              # Data loading, feature engineering, exploration
│   │   ├── data_loader.py          # Raw data ingestion (CSV / HDF5)
│   │   ├── feature_engineering.py  # Rolling statistics on vibration & temperature
│   │   ├── feature_matching.py     # Lab ↔ Field signal matching (Pearson, DTW, KS)
│   │   ├── exploration.py          # Exploratory data analysis utilities
│   │   └── config.yaml             # Signal and path configuration
│   │
│   ├── forecasting/                # Time-series forecasting models
│   │   ├── models.py               # Architectures: CNN, LSTM, CNN-LSTM, TCN, Transformer
│   │   ├── train_optimize.py       # Training loop + Optuna HPO
│   │   ├── data_loader.py          # Windowing, downsampling, standardization
│   │   ├── utils.py                # Reproducibility seeds, GPU config
│   │   └── statistical_models/     # ARIMA, Prophet, VAR, XGBoost, LightGBM, RF
│   │
│   ├── anomaly_detection/          # Anomaly detection methods
│   │   ├── forecast_based.py       # Residual-based: Z-score, MAD, CUSUM, EWMA, IF, LOF
│   │   ├── classification.py       # Evaluation approaches A (independent) & B (rolling)
│   │   ├── testing_pipeline.py     # Cross-domain validation (normal → fatigue → field)
│   │   └── graph_based/            # Graph-theoretic anomaly detection
│   │       ├── visibility_graphs.py    # NVG / HVG construction from sensor signals
│   │       ├── network_analysis.py     # Degree distributions & graph metrics
│   │       ├── build_graphs/           # Graph builders (Bonfiglioli & Tampieri)
│   │       ├── centralities/           # Betweenness, closeness, clustering (NetworKit)
│   │       ├── communities/            # VGCD community detection (pyiomica)
│   │       ├── multiplex/              # Average edge overlap across VG layers
│   │       └── oddball/                # OddBall anomaly scoring (clique/star, dominant pair)
│   │
│   ├── edge_deployment/            # Embedded inference pipeline
│   │   ├── train_cnn_lstm.py       # HPC-optimized training script
│   │   ├── export_model.py         # Keras → TF SavedModel conversion
│   │   ├── quantization/           # INT8 TFLite conversion & benchmarking
│   │   │   ├── convert_model.py    # Full INT8 quantization
│   │   │   ├── benchmark.py        # Accuracy & latency comparison
│   │   │   ├── make_representative.py  # Calibration dataset generator
│   │   │   └── sweep_benchmark.py  # Multi-config quantization sweep
│   │   ├── stm32/                  # STM32U545RE firmware (C, X-CUBE-AI, Modbus RTU)
│   │   ├── esp32/                  # ESP32-WROVER firmware (TFLite Micro, UART)
│   │   └── validation/             # Post-deployment validation & comparison
│   │
│   └── pipeline/                   # Orchestration
│       ├── main.py                 # CLI entry point (model selection, training, inference)
│       └── run_experiments.py      # Batch experiment runner across scenarios
│
├── requirements.txt                # Python dependencies
└── LICENSE                         # GNU General Public License v3.0
```

---

## Modules

### Preprocessing

- **`FeatureEngineer`** — computes rolling statistics (mean, std, min, max) over configurable windows on vibration and temperature signals, driven by YAML configuration.
- **`FeatureMatcher`** — finds the best correspondence between laboratory and field sensor signals using Pearson correlation, DTW distance, Kolmogorov–Smirnov tests, and cross-correlation.

### Forecasting

| Family | Models | Implementation |
|--------|--------|----------------|
| **Deep Learning** | CNN, LSTM, CNN-LSTM, TCN, Transformer | PyTorch + Optuna HPO |
| **Statistical** | ARIMA (Auto-ARIMA via pmdarima), Prophet, VAR | statsmodels / pmdarima / prophet |
| **Machine Learning** | XGBoost, LightGBM, Random Forest, HistGradientBoosting | scikit-learn / xgboost / lightgbm |

- Default configuration: 24-hour input window → 24-hour prediction horizon (2 880 samples at 30 s).
- Optuna-driven hyperparameter search with MSE, MAE, R², MAPE, MASE metrics.

### Anomaly Detection

**Forecast-residual methods:**
- Rolling Z-score, Median Absolute Deviation (MAD), peak detection
- CUSUM, EWMA for trend analysis
- Isolation Forest, Local Outlier Factor (LOF)
- Composite Health Index combining multiple indicators

**Graph-based methods:**
- Natural Visibility Graphs (NVG) and Horizontal Visibility Graphs (HVG) constructed from 16 sensor signals
- **OddBall** anomaly scoring — clique/star patterns, dominant pair, heavy vicinity
- Community detection via VGCD algorithm (pyiomica)
- Centrality analysis — betweenness, closeness, local clustering coefficient (NetworKit)
- Multiplex edge overlap analysis across graph layers

**Cross-domain testing pipeline:**
- Trains on normal (characterization) data
- Tests on fatigue data (should trigger anomalies)
- Validates on field data (should not false-alarm)
- Sweeps over multiple classifiers (Isolation Forest, One-Class SVM, LOF) and scalers

### Edge Deployment

| Target | Processor | Framework | Quantization |
|--------|-----------|-----------|-------------|
| **STM32U545RE** | Cortex-M33 @ 160 MHz | X-CUBE-AI | INT8 (full) |
| **ESP32-WROVER** | Xtensa LX6 (SECO EasyEdge) | TFLite Micro | Float32 / INT8 |

- Full INT8 post-training quantization via TFLite converter with representative calibration dataset.
- STM32 firmware includes **Modbus RTU** passive sniffing (RS-485, 9600 bps) for industrial bus integration.
- Accuracy and latency benchmarking tooling to compare quantized vs. original inference.

---

## Quick Start

### Prerequisites

- Python 3.12+
- [micromamba](https://mamba.readthedocs.io/) (recommended) or conda

### Environment Setup

```bash
# Create and activate environment
micromamba create -n safe python=3.12 -y
micromamba activate safe

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Full pipeline (select models to train and evaluate)
python src/pipeline/main.py --models cnn cnn_lstm --data path/to/data.csv

# List available models
python src/pipeline/main.py --list-models

# Run all models
python src/pipeline/main.py --all --data path/to/data.csv

# Batch experiments across scenarios
python src/pipeline/run_experiments.py
```

### Individual Components

```bash
# Forecast model training with Optuna optimization
python src/forecasting/train_optimize.py

# Forecast-residual anomaly detection
python src/anomaly_detection/forecast_based.py

# Graph-based analysis (build visibility graphs)
python src/anomaly_detection/graph_based/visibility_graphs.py

# Cross-domain anomaly detection testing
python src/anomaly_detection/testing_pipeline.py
```

### Edge Deployment

```bash
# Export Keras model to SavedModel format
python src/edge_deployment/export_model.py

# Generate representative calibration dataset
python src/edge_deployment/quantization/make_representative.py

# Convert to INT8 TFLite
python src/edge_deployment/quantization/convert_model.py

# Benchmark quantized vs. original model
python src/edge_deployment/quantization/benchmark.py
```

---

## Dependencies

Core dependencies include:

- **Scientific stack:** NumPy, Pandas, SciPy, scikit-learn
- **Deep Learning:** TensorFlow/Keras, PyTorch, PyTorch Lightning
- **Time Series:** statsmodels, pmdarima, Prophet, keras-tcn
- **Gradient Boosting:** XGBoost, LightGBM
- **Graph Analysis:** NetworkX, ts2vg, igraph, NetworKit, pyiomica, pyunicorn, pyflagser, giotto-ph
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Data I/O:** h5py, PyTables, openpyxl

See [requirements.txt](requirements.txt) for the complete list.

---

## Team

| Name | Role | Affiliation |
|------|------|-------------|
| Orso Peruzzi | Project Lead, Senior Data Scientist | IFAB Foundation |
| Benedetta Baldini | Senior Data Scientist, Coordinator | IFAB Foundation |
| Giacomo Piergentili | Research Fellow — Preprocessing & Feature Engineering | University of Bologna |
| Lucia Gasperini | Research Fellow — Forecasting & Anomaly Detection | University of Bologna |
| Ester Cima | Research Fellow — Graph-Based Analysis | University of Bologna |
| Francesco Simoni | Research Fellow — Edge Deployment & Quantization | University of Bologna |

---

## License

This project is licensed under the **GNU General Public License v3.0** — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{safe2025,
  title   = {SAFE --- Secure Anomaly Detection Edge AI System for Critical Environments},
  author  = {Peruzzi, Orso and Baldini, Benedetta and Piergentili, Giacomo and
             Gasperini, Lucia and Cima, Ester and Simoni, Francesco},
  year    = {2025},
  url     = {https://github.com/ifabfoundation/SAFE},
  license = {GPL-3.0}
}
```

---

*SAFE Project — IFAB Foundation & University of Bologna, 2025*