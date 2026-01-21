# Deep Learning Models and Models Anomaly Detection

## Overview
This module implements Deep Learning algorithms for anomaly detection in time series data. It combines forecasting models with statistical anomaly detection methods to identify anomalies based on forecast residuals.

## Features
- **Multiple Forecasting Models**: CNN, LSTM, CNN-LSTM, TCN, Transformer and Gradient Boosting
- **Anomaly Detection Methods**: 
  - Z-score (rolling window)
  - MAD (Median Absolute Deviation)
  - Peak Detection
  - Trend Analysis
  - CUSUM (Cumulative Sum Control Chart)
  - EWMA (Exponentially Weighted Moving Average)
  - Isolation Forest
  - Local Outlier Factor
  - Health Index Computation
- **Model Optimization**: Optuna hyperparameter optimization and grid search
- **Multi-approach Forecasting**: Compare different forecasting approaches

## Project Structure

```
DL-anomaly-detection/
├── main.py                  # Training and forecasting pipeline
├── models.py                # Model definitions (CNN, LSTM, SimpleRNN, CNN-LSTM, TCN, Transformer)
├── anomaly_detection.py     # Anomaly detection algorithms
├── data_loader.py           # Data loading and preprocessing
├── train_optimize.py        # Training and optimization utilities
├── utils.py                 # Helper functions
├── requirements.txt         # Python dependencies
├── __init__.py              # Package initialization
├── pretrained_models/       # Pre-trained model weights
├── results/                 # Results and analysis outputs
│   ├── anomaly_plots/       # Visualization plots
│   ├── forecasting/         # Forecasting results
│   │   ├── approach_A/
│   │   ├── approach_B/
│   │   └── comparison/
│   ├── anomaly_detection_results.json
│   └── model_comparison.csv
└── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA/GPU support (optional, for faster training)

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training
```bash
python main.py --models cnn cnn_lstm --data path/to/data.csv
```

### Train All Available Models
```bash
python main.py --all --data path/to/data.csv
```

### List Available Models
```bash
python main.py --list-models
```

### Using Pre-trained Models
Pre-trained models are available in the `pretrained_models/` directory:
- `cnn_lstm_W2880_F2880_E30_D15_LR2e-3.keras`
- `cnn_pretrained.keras`

## Core Modules

### models.py
Defines all forecasting architectures:
- **CNN**: Convolutional Neural Network for spatial feature extraction
- **LSTM**: Long Short-Term Memory for temporal dependencies
- **SimpleRNN**: Simple Recurrent Neural Network
- **CNN-LSTM**: Hybrid architecture combining CNN and LSTM
- **TCN**: Temporal Convolutional Network (if available)
- **Transformer**: Attention-based architecture
- **HistGradientBoosting**: Scikit-learn ensemble method

### anomaly_detection.py
Implements multiple anomaly detection methods:
- Statistical methods (Z-score, MAD, EWMA, CUSUM)
- ML-based methods (Isolation Forest, Local Outlier Factor)
- Specialized methods (Peak Detection, Trend Analysis, Health Index)

### data_loader.py
Handles data loading and preprocessing:
- CSV data loading
- Automatic column type identification
- Train/test/validation split
- Data normalization

### train_optimize.py
Training utilities and optimization:
- Model training with various configurations
- Hyperparameter optimization (Optuna, Grid Search)
- Performance metrics and reporting

## Results

Results are saved in the `results/` directory:
- **anomaly_detection_results.json**: Detailed anomaly detection metrics
- **model_comparison.csv**: Performance comparison across models
- **anomaly_plots/**: Visualization of detected anomalies
- **forecasting/**: Forecasting results for different approaches

## Key Components

### AnomalyConfig
Configuration class for anomaly detection parameters:
- Z-score window and threshold
- MAD threshold
- Trend analysis parameters
- CUSUM baseline samples
- Peak detection threshold

### ModelConfig
Base configuration for all models:
- Learning rate
- Dropout rate
- Batch size
- Number of epochs

### TrainingConfig
Training pipeline configuration with various hyperparameters

## Supported Data Formats

- CSV files with numeric columns
- Automatic detection of temperature and vibration columns
- Flexible column selection for model training

## Output

The module generates:
- Trained model files (.keras/.h5 format)
- Anomaly detection reports (JSON)
- Comparative analysis (CSV)
- Visualization plots (PNG)
- Forecast residuals and anomaly scores
