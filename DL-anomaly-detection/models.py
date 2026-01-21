"""
Models Module
=============
Definition of forecasting models for time series.
Pure classes and functions to build CNN, LSTM, TCN, Transformer models.
No direct data access.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

# TensorFlow / Keras
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout, LSTM, Dense,
    Reshape, Flatten, SimpleRNN, RepeatVector, TimeDistributed,
    BatchNormalization
)
from keras.optimizers import Adam

# PyTorch per Transformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# TCN (optional)
try:
    from tcn import TCN
    TCN_AVAILABLE = True
except ImportError:
    TCN_AVAILABLE = False

# Scikit-learn per HistGradientBoosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd


@dataclass
class ModelConfig:
    """Base configuration for models."""
    learning_rate: float = 1e-3
    dropout_rate: float = 0.3
    batch_size: int = 32
    epochs: int = 50
    patience: int = 10


# =============================================================================
# Keras/TensorFlow Models
# =============================================================================

def build_cnn_model(
    window_size: int,
    forecast_length: int,
    num_features: int,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.3,
    filters: Tuple[int, ...] = (64, 128),
    dense_units: Tuple[int, ...] = (64,)
) -> Model:
    """
    Build a 1D CNN model for multi-step prediction.
    
    Args:
        window_size: Input window size
        forecast_length: Number of steps to predict
        num_features: Number of features (sensors)
        learning_rate: Learning rate for the optimizer
        dropout_rate: Dropout rate
        filters: Tuple with number of filters for each conv layer
        dense_units: Tuple with units for each dense layer
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    model.add(Input(shape=(window_size, num_features)))
    
    for f in filters:
        model.add(Conv1D(filters=f, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_rate))
    
    model.add(Flatten())
    
    for u in dense_units:
        model.add(Dense(u, activation='relu'))
    
    model.add(Dense(forecast_length * num_features))
    model.add(Reshape((forecast_length, num_features)))
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def build_cnn_lstm_model(
    window_size: int,
    forecast_length: int,
    num_features: int,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.3,
    conv_filters: Tuple[int, ...] = (64, 128),
    lstm_units: int = 128
) -> Model:
    """
    Build a CNN + LSTM model for time series.
    
    Args:
        window_size: Input window size
        forecast_length: Number of steps to predict
        num_features: Number of features
        learning_rate: Learning rate
        dropout_rate: Dropout rate
        conv_filters: Filters for convolutional layers
        lstm_units: LSTM units
        
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=(window_size, num_features))
    x = inputs
    
    for f in conv_filters:
        x = Conv1D(f, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(dropout_rate)(x)
    
    x = LSTM(lstm_units, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x = LSTM(lstm_units, return_sequences=False)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dense(forecast_length * num_features)(x)
    outputs = Reshape((forecast_length, num_features))(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def build_lstm_model(
    window_size: int,
    forecast_length: int,
    num_features: int,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.3,
    lstm_units: List[int] = [64, 64]
) -> Model:
    """
    Build a pure LSTM model.
    
    Args:
        window_size: Input window size
        forecast_length: Number of steps to predict
        num_features: Number of features
        learning_rate: Learning rate
        dropout_rate: Dropout rate
        lstm_units: List with units for each LSTM layer
        
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=(window_size, num_features))
    x = inputs
    
    for i, units in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1
        x = LSTM(units, return_sequences=return_seq)(x)
        x = Dropout(dropout_rate)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dense(forecast_length * num_features)(x)
    outputs = Reshape((forecast_length, num_features))(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def build_tcn_model(
    window_size: int,
    forecast_length: int,
    num_features: int,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.3,
    nb_filters: int = 64,
    kernel_size: int = 3,
    dilations: Tuple[int, ...] = (1, 2, 4, 8)
) -> Model:
    """
    Build a TCN (Temporal Convolutional Network) model.
    
    Args:
        window_size: Input window size
        forecast_length: Number of steps to predict
        num_features: Number of features
        learning_rate: Learning rate
        dropout_rate: Dropout rate
        nb_filters: Number of TCN filters
        kernel_size: Kernel size
        dilations: Dilation factors
        
    Returns:
        Compiled Keras model
        
    Raises:
        RuntimeError: If TCN package is not available
    """
    if not TCN_AVAILABLE:
        raise RuntimeError("Package 'tcn' not installed. Use: pip install keras-tcn")
    
    model = Sequential()
    model.add(Input(shape=(window_size, num_features)))
    model.add(TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        dilations=list(dilations),
        dropout_rate=dropout_rate,
        return_sequences=False
    ))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(forecast_length * num_features))
    model.add(Reshape((forecast_length, num_features)))
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def build_tcn_lstm_model(
    window_size: int,
    forecast_length: int,
    num_features: int,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.3,
    nb_filters: int = 64,
    kernel_size: int = 3,
    dilations: Tuple[int, ...] = (1, 2, 4, 8),
    lstm_units: int = 64
) -> Model:
    """
    Build a combined TCN + LSTM model.
    
    Args:
        window_size: Input window size
        forecast_length: Number of steps to predict
        num_features: Number of features
        learning_rate: Learning rate
        dropout_rate: Dropout rate
        nb_filters: Number of TCN filters
        kernel_size: Kernel size
        dilations: Dilation factors
        lstm_units: LSTM units
        
    Returns:
        Compiled Keras model
    """
    if not TCN_AVAILABLE:
        raise RuntimeError("Package 'tcn' not installed.")
    
    inputs = Input(shape=(window_size, num_features))
    
    x = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        dilations=list(dilations),
        dropout_rate=dropout_rate,
        return_sequences=True
    )(inputs)
    x = Dropout(dropout_rate)(x)
    x = LSTM(lstm_units, return_sequences=False)(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(forecast_length * num_features)(x)
    outputs = Reshape((forecast_length, num_features))(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def build_deepant_model(
    window_size: int,
    forecast_length: int,
    num_features: int,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.3
) -> Model:
    """
    Build the DeepAnt model for forecast-based anomaly detection.
    
    Reference: "DeepAnt: A Deep Learning Approach for Unsupervised Anomaly Detection"
    
    Args:
        window_size: Input window size
        forecast_length: Number of steps to predict
        num_features: Number of features
        learning_rate: Learning rate
        dropout_rate: Dropout rate
        
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=(window_size, num_features))
    
    # Two convolutional blocks as in the original paper
    x = Conv1D(32, 2, activation='relu')(inputs)
    x = MaxPooling1D(2)(x)
    x = Conv1D(32, 2, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    
    # Fully connected
    x = Dense(64, activation='relu')(x)
    x = Dense(forecast_length * num_features)(x)
    outputs = Reshape((forecast_length, num_features))(x)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def build_seq2seq_model(
    window_size: int,
    forecast_length: int,
    num_features: int,
    learning_rate: float = 1e-3,
    rnn_units: int = 64,
    dense_units: int = 32
) -> Model:
    """
    Build a Seq2Seq model based on SimpleRNN.
    
    Args:
        window_size: Input window size
        forecast_length: Number of steps to predict
        num_features: Number of features
        learning_rate: Learning rate
        rnn_units: RNN units
        dense_units: Dense units
        
    Returns:
        Compiled Keras model
    """
    # Encoder
    encoder_inputs = Input(shape=(window_size, num_features))
    encoder = SimpleRNN(rnn_units, activation='tanh', return_sequences=False)(encoder_inputs)
    encoder_dense = Dense(dense_units, activation='relu')(encoder)
    
    # Decoder
    decoder_repeat = RepeatVector(forecast_length)(encoder_dense)
    decoder_rnn = SimpleRNN(rnn_units, activation='tanh', return_sequences=True)(decoder_repeat)
    decoder_outputs = TimeDistributed(Dense(num_features))(decoder_rnn)
    
    model = Model(encoder_inputs, decoder_outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def build_rnn_micro_model(
    window_size: int,
    forecast_length: int,
    num_features: int,
    learning_rate: float = 1e-3,
    rnn_units: int = 16,
    dense_units: int = 16
) -> Model:
    """
    Build a minimalist RNN model for quick tests.
    
    Args:
        window_size: Input window size
        forecast_length: Number of steps to predict
        num_features: Number of features
        learning_rate: Learning rate
        rnn_units: RNN units
        dense_units: Dense units
        
    Returns:
        Compiled Keras model
    """
    inputs = Input(shape=(window_size, num_features))
    x = SimpleRNN(rnn_units, activation='tanh')(inputs)
    x = Dense(dense_units, activation='relu')(x)
    outputs = Dense(forecast_length * num_features)(x)
    outputs = Reshape((forecast_length, num_features))(outputs)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# =============================================================================
# PyTorch Models - Transformer
# =============================================================================

class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward."""
    
    def __init__(self, embed_size: int, num_heads: int, drop_prob: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.LeakyReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(drop_prob)
        self.ln1 = nn.LayerNorm(embed_size, eps=1e-6)
        self.ln2 = nn.LayerNorm(embed_size, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x)
        x = self.ln1(x + self.dropout(attn_output))
        ff_output = self.fc(x)
        x = self.ln2(x + self.dropout(ff_output))
        return x


class TransformerForecaster(nn.Module):
    """
    Transformer model for time series forecasting.
    
    Args:
        input_dim: Input dimension (number of features)
        embed_size: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        forecast_length: Number of steps to predict
        drop_prob: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        forecast_length: int = 1,
        drop_prob: float = 0.1
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_size)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, drop_prob=drop_prob)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, input_dim)
        self.forecast_length = forecast_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, window, features]
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        last = x[:, -1, :]  # [batch, embed]
        out = self.fc_out(last)  # [batch, input_dim]
        out = out.unsqueeze(1).repeat(1, self.forecast_length, 1)
        return out


class SequenceDataset(Dataset):
    """PyTorch Dataset for time-series sequences."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_transformer_model(
    input_dim: int,
    forecast_length: int,
    embed_size: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    drop_prob: float = 0.1
) -> TransformerForecaster:
    """
    Factory function to create a TransformerForecaster.
    
    Args:
        input_dim: Number of features
        forecast_length: Number of steps to predict
        embed_size: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of layers
        drop_prob: Dropout probability
        
    Returns:
        TransformerForecaster model
    """
    return TransformerForecaster(
        input_dim=input_dim,
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        forecast_length=forecast_length,
        drop_prob=drop_prob
    )


# =============================================================================
# Factory and utilities
# =============================================================================

def build_transformer_wrapper(
    window_size: int,
    forecast_length: int,
    num_features: int,
    learning_rate: float = 1e-3,
    dropout_rate: float = 0.1,
    embed_size: int = 64,
    num_heads: int = 4,
    num_layers: int = 2
) -> TransformerForecaster:
    """
    Wrapper for TransformerForecaster with standard interface.
    
    Args:
        window_size: Input window size (not used directly, kept for interface compatibility)
        forecast_length: Number of steps to predict
        num_features: Number of features
        learning_rate: Learning rate (stored as attribute for external use)
        dropout_rate: Dropout probability
        embed_size: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        
    Returns:
        TransformerForecaster model (PyTorch)
    """
    model = TransformerForecaster(
        input_dim=num_features,
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        forecast_length=forecast_length,
        drop_prob=dropout_rate
    )
    # Store learning_rate as attribute for training
    model.learning_rate = learning_rate
    return model


MODEL_BUILDERS = {
    'cnn': build_cnn_model,
    'cnn_lstm': build_cnn_lstm_model,
    'lstm': build_lstm_model,
    'deepant': build_deepant_model,
    'seq2seq': build_seq2seq_model,
    'rnn_micro': build_rnn_micro_model,
    'transformer': build_transformer_wrapper,
}

# Add TCN only if available
if TCN_AVAILABLE:
    MODEL_BUILDERS['tcn'] = build_tcn_model
    MODEL_BUILDERS['tcn_lstm'] = build_tcn_lstm_model


def get_model(
    model_name: str,
    window_size: int,
    forecast_length: int,
    num_features: int,
    **kwargs
) -> Model:
    """
    Factory function to get a model by name.
    
    Args:
        model_name: Model name ('cnn', 'cnn_lstm', 'lstm', 'tcn', 'tcn_lstm',
                   'deepant', 'seq2seq', 'rnn_micro', 'transformer')
        window_size: Input window size
        forecast_length: Number of steps to predict
        num_features: Number of features
        **kwargs: Additional model-specific parameters
        
    Returns:
        Compiled model
        
    Raises:
        ValueError: If model is not supported
        
    Note:
        For HistGradientBoosting, use train_boosting() directly instead.
    """
    model_name = model_name.lower()
    
    if model_name not in MODEL_BUILDERS:
        available = list(MODEL_BUILDERS.keys())
        raise ValueError(f"Model '{model_name}' not supported. Available: {available}")
    
    builder = MODEL_BUILDERS[model_name]
    return builder(
        window_size=window_size,
        forecast_length=forecast_length,
        num_features=num_features,
        **kwargs
    )


def get_available_models() -> List[str]:
    """Return the list of available neural network models."""
    return list(MODEL_BUILDERS.keys())


def get_all_available_models() -> List[str]:
    """Return the list of all available models including boosting."""
    models = list(MODEL_BUILDERS.keys())
    models.append('boosting')  # HistGradientBoosting (use train_boosting)
    return models


def is_tcn_available() -> bool:
    """Check if TCN is available."""
    return TCN_AVAILABLE


if __name__ == "__main__":
    # Test models
    print("Models Module - Test")
    print("=" * 50)
    
    # Default values (same as main.py)
    window_size = 2880      # 24 hours at 30s intervals
    forecast_length = 2880  # 24 hours at 30s intervals
    num_features = 5
    
    print(f"\nAvailable models: {get_available_models()}")
    print(f"TCN available: {is_tcn_available()}")
    
    # Test CNN
    print("\n--- Test CNN ---")
    model_cnn = get_model('cnn', window_size, forecast_length, num_features)
    model_cnn.summary()
    
    # Test CNN-LSTM
    print("\n--- Test CNN-LSTM ---")
    model_cnn_lstm = get_model('cnn_lstm', window_size, forecast_length, num_features)
    print(f"Input shape: {model_cnn_lstm.input_shape}")
    print(f"Output shape: {model_cnn_lstm.output_shape}")
    
    # Test Transformer
    print("\n--- Test Transformer ---")
    transformer = build_transformer_model(num_features, forecast_length)
    x_test = torch.randn(2, window_size, num_features)
    y_test = transformer(x_test)
    print(f"Transformer output shape: {y_test.shape}")


# =============================================================================
# HistGradientBoosting Model (Scikit-learn)
# =============================================================================

def make_supervised(
    df: pd.DataFrame,
    lags: Tuple[int, ...] = (1, 2, 3, 5, 10),
    roll_windows: Tuple[int, ...] = (3, 5, 10),
    horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lag and rolling features for supervised learning.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data
    lags : Tuple[int, ...]
        Lag values to create features from
    roll_windows : Tuple[int, ...]
        Window sizes for rolling statistics
    horizon : int
        Forecast horizon (number of steps to predict)
        
    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target matrix (multi-step if horizon > 1)
    """
    feats = df.copy()
    
    # Lag features
    for col in df.columns:
        for lag in lags:
            feats[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    # Rolling features
    for col in df.columns:
        for w in roll_windows:
            feats[f'{col}_roll_mean{w}'] = df[col].shift(1).rolling(w).mean()
            feats[f'{col}_roll_std{w}'] = df[col].shift(1).rolling(w).std()
            feats[f'{col}_roll_min{w}'] = df[col].shift(1).rolling(w).min()
            feats[f'{col}_roll_max{w}'] = df[col].shift(1).rolling(w).max()
    
    # Multi-step target
    targets = []
    for h in range(1, horizon + 1):
        for col in df.columns:
            targets.append(df[col].shift(-h).rename(f'{col}_t+{h}'))
    target_df = pd.concat(targets, axis=1)
    
    # Drop NaN rows
    combined = pd.concat([feats, target_df], axis=1).dropna()
    
    X = combined[feats.columns].values
    y = combined[target_df.columns].values
    
    return X, y


def train_boosting(
    df: pd.DataFrame,
    horizon: int = 1,
    test_size: float = 0.2,
    learning_rate: float = 0.06,
    max_iter: int = 100,
    max_depth: Optional[int] = None,
    save_path: Optional[str] = None
) -> dict:
    """
    Train HistGradientBoosting with MultiOutputRegressor for multi-step forecasting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data
    horizon : int
        Forecast horizon (number of steps)
    test_size : float
        Fraction of data for testing
    learning_rate : float
        Learning rate for boosting
    max_iter : int
        Maximum number of boosting iterations
    max_depth : int, optional
        Maximum depth of trees (None for unlimited)
    save_path : str, optional
        Path to save the trained model (without extension)
        
    Returns
    -------
    dict with keys:
        - model: trained MultiOutputRegressor
        - scaler_X: StandardScaler for features
        - scaler_y: StandardScaler for targets
        - y_test: actual test values
        - y_pred: predicted test values
        - report: dict with metrics (MSE, MAE, R2 per target)
    """
    # Create supervised dataset
    X, y = make_supervised(df, horizon=horizon)
    
    # Split data (preserving temporal order)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features and targets
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    # Build model
    base_model = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=max_depth,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1
    )
    
    model = MultiOutputRegressor(base_model, n_jobs=-1)
    
    # Train
    print(f"Training HistGradientBoosting (horizon={horizon}, max_iter={max_iter})...")
    model.fit(X_train_scaled, y_train_scaled)
    
    # Predict
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    report = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mse_per_target': [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])],
        'mae_per_target': [mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])],
        'r2_per_target': [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
    }
    
    print(f"Overall MSE: {report['mse']:.6f}")
    print(f"Overall MAE: {report['mae']:.6f}")
    print(f"Overall R²: {report['r2']:.4f}")
    
    # Save model
    if save_path:
        model_data = {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'horizon': horizon,
            'feature_columns': list(df.columns)
        }
        joblib.dump(model_data, f"{save_path}_boosting.joblib")
        print(f"Model saved to {save_path}_boosting.joblib")
    
    return {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'y_test': y_test,
        'y_pred': y_pred,
        'report': report
    }


def load_boosting_model(path: str) -> dict:
    """
    Load a saved boosting model.
    
    Parameters
    ----------
    path : str
        Path to the saved model file
        
    Returns
    -------
    dict with model, scalers, and metadata
    """
    return joblib.load(path)


def predict_boosting(
    model_data: dict,
    df: pd.DataFrame,
    horizon: Optional[int] = None
) -> np.ndarray:
    """
    Make predictions with a trained boosting model.
    
    Parameters
    ----------
    model_data : dict
        Dictionary from train_boosting or load_boosting_model
    df : pd.DataFrame
        Input data for prediction
    horizon : int, optional
        Forecast horizon (uses model's horizon if not specified)
        
    Returns
    -------
    np.ndarray
        Predictions (inverse transformed to original scale)
    """
    horizon = horizon or model_data['horizon']
    X, _ = make_supervised(df, horizon=horizon)
    
    X_scaled = model_data['scaler_X'].transform(X)
    y_pred_scaled = model_data['model'].predict(X_scaled)
    y_pred = model_data['scaler_y'].inverse_transform(y_pred_scaled)
    
    return y_pred
