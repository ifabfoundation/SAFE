"""
Data Loader Module
===================
Handles data reading, preprocessing, feature engineering and standardization.
No training or anomaly detection logic.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict


@dataclass
class DataConfig:
    """
    Configuration for data loading and preprocessing.
    
    Temporal Configuration:
        - Original sampling: every 2 seconds
        - Downsampled sampling: every 30 seconds (factor=15)
        - Window: 24 hours = 24 * 3600 / 30 = 2880 samples
        - Forecast: 24 hours = 2880 samples
    """
    # Sampling parameters
    SAMPLING_RATE_SECONDS: int = 2  # Original data: 1 sample every 2 seconds
    SAMPLES_PER_HOUR_ORIGINAL: int = 3600 // 2  # 1800 samples/hour at 2s
    
    # Downsampling: from 2s to 30s (factor = 15, since 2 * 15 = 30)
    DOWNSAMPLE_FACTOR: int = 15
    
    # Temporal windows (in hours)
    # After downsampling: 24h * 120 samples/h = 2880 samples
    WINDOW_HOURS_REAL: float = 24.0  # 24-hour input window
    FORECAST_HOURS_REAL: float = 24.0  # 24-hour forecast horizon
    
    # Data split
    TEST_SIZE: float = 0.2
    SHUFFLE: bool = False
    
    # Seed for reproducibility
    RANDOM_SEED: int = 42
    
    @property
    def effective_sampling_rate(self) -> int:
        return self.SAMPLING_RATE_SECONDS * self.DOWNSAMPLE_FACTOR
    
    @property
    def samples_per_hour(self) -> int:
        return self.SAMPLES_PER_HOUR_ORIGINAL // self.DOWNSAMPLE_FACTOR
    
    @property
    def window_size_samples(self) -> int:
        return int(self.WINDOW_HOURS_REAL * self.samples_per_hour)
    
    @property
    def forecast_samples(self) -> int:
        return int(self.FORECAST_HOURS_REAL * self.samples_per_hour)


def load_csv(filepath: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load a CSV file.
    
    Args:
        filepath: Path to the CSV file
        verbose: If True, print dataset information
        
    Returns:
        DataFrame with loaded data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    if verbose:
        print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
    
    return df


def load_multiple_csv(
    directory: str,
    filenames: List[str],
    dataset_type: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load and concatenate multiple CSV files from a directory.
    
    Args:
        directory: Directory containing the files
        filenames: List of file names to load
        dataset_type: Dataset type (optional, added as column)
        verbose: If True, print information
        
    Returns:
        Concatenated DataFrame
    """
    dataframes = []
    
    for filename in filenames:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            df_temp = pd.read_csv(filepath)
            df_temp['source_file'] = filename
            if dataset_type:
                df_temp['dataset_type'] = dataset_type
            dataframes.append(df_temp)
            if verbose:
                print(f"Loaded: {filename} ({len(df_temp):,} samples)")
        else:
            if verbose:
                print(f"File not found: {filename}")
    
    if not dataframes:
        raise RuntimeError(f"No files found in {directory}")
    
    df_combined = pd.concat(dataframes, ignore_index=True)
    
    if verbose:
        print(f"\nTotal: {len(df_combined):,} samples from {len(dataframes)} files")
    
    return df_combined


def get_numeric_columns(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None
) -> List[str]:
    """
    Return the numeric columns of a DataFrame.
    
    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude
        
    Returns:
        List of numeric columns
    """
    if exclude_cols is None:
        exclude_cols = ['time', 'anomaly', 'anomaly_score', 'source_file', 'dataset_type']
    
    numeric_cols = [
        col for col in df.columns
        if col not in exclude_cols and df[col].dtype in ['float64', 'int64', 'float32', 'int32']
    ]
    
    return numeric_cols


def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'ffill_bfill',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: Input DataFrame
        method: Handling method ('ffill_bfill', 'drop', 'mean')
        verbose: If True, print information
        
    Returns:
        DataFrame without missing values
    """
    df_clean = df.copy()
    missing_before = df_clean.isna().sum().sum()
    
    if missing_before > 0:
        if verbose:
            print(f"Missing values found: {missing_before}")
        
        if method == 'ffill_bfill':
            df_clean = df_clean.ffill().bfill()
        elif method == 'drop':
            df_clean = df_clean.dropna()
        elif method == 'mean':
            numeric_cols = get_numeric_columns(df_clean)
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
                df_clean[numeric_cols].mean()
            )
        
        missing_after = df_clean.isna().sum().sum()
        if verbose:
            print(f"Missing values after: {missing_after}")
    
    return df_clean


def downsample_data(
    df: pd.DataFrame,
    factor: int,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Apply downsampling to the DataFrame.
    
    Args:
        df: Input DataFrame
        factor: Downsampling factor (takes 1 sample every 'factor')
        verbose: If True, print information
        
    Returns:
        Downsampled DataFrame
    """
    if factor <= 1:
        return df
    
    df_downsampled = df.iloc[::factor].reset_index(drop=True)
    
    if verbose:
        print(f"Downsampling: {len(df):,} → {len(df_downsampled):,} samples (factor={factor})")
    
    return df_downsampled


def standardize_data(
    data: np.ndarray,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize data using StandardScaler.
    
    Args:
        data: Numpy array of data
        scaler: Existing scaler (optional)
        fit: If True, fit the scaler on data
        
    Returns:
        Tuple (standardized data, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        data_scaled = scaler.fit_transform(data).astype(np.float32)
    else:
        data_scaled = scaler.transform(data).astype(np.float32)
    
    return data_scaled, scaler


def create_sequences(
    data: np.ndarray,
    window_size: int,
    forecast_length: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time-series models (X: input, y: target).
    
    Args:
        data: Numpy array [num_samples, num_features]
        window_size: Number of historical timesteps for input
        forecast_length: Number of future timesteps to predict
        
    Returns:
        Tuple (X, y) where:
            X: [num_sequences, window_size, num_features]
            y: [num_sequences, forecast_length, num_features]
    """
    X, y = [], []
    
    for i in range(len(data) - window_size - forecast_length + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + forecast_length])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    return X, y


def split_sequences(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    shuffle: bool = False,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Temporal split of sequences into train/test.
    
    Args:
        X: Input sequences
        y: Target sequences
        test_size: Test set proportion
        shuffle: If True, shuffle data (False for temporal data)
        random_state: Seed for reproducibility
        
    Returns:
        Tuple (X_train, X_test, y_train, y_test)
    """
    if shuffle:
        return train_test_split(
            X, y,
            test_size=test_size,
            shuffle=True,
            random_state=random_state
        )
    else:
        # Temporal split without shuffle
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test


def prepare_data_for_training(
    df: pd.DataFrame,
    config: Optional[DataConfig] = None,
    window_size: Optional[int] = None,
    forecast_length: Optional[int] = None,
    downsample_factor: int = 1,
    exclude_current: bool = True,
    test_size: Optional[float] = None,
    verbose: bool = True
) -> Dict:
    """
    Complete data preparation pipeline for training.
    
    Args:
        df: Input DataFrame
        config: Configuration (optional)
        window_size: Window size (overrides config)
        forecast_length: Forecast length (overrides config)
        downsample_factor: Downsampling factor
        exclude_current: If True, exclude current columns from training features
                        (current is typically used only for anomaly detection)
        test_size: Test set proportion (overrides config, use 0.0 for no split)
        verbose: If True, print information
        
    Returns:
        Dictionary with all prepared data
    """
    if config is None:
        config = DataConfig()
        config.DOWNSAMPLE_FACTOR = downsample_factor
    
    # Get parameters
    ws = window_size if window_size else config.window_size_samples
    fl = forecast_length if forecast_length else config.forecast_samples
    
    # Preprocessing
    df_clean = handle_missing_values(df, verbose=verbose)
    df_down = downsample_data(df_clean, downsample_factor, verbose=verbose)
    
    # Extract numeric columns
    numeric_cols = get_numeric_columns(df_down)
    
    # Exclude current columns if requested (for model training)
    # Current is typically used only for anomaly detection, not forecasting
    current_cols = []
    if exclude_current:
        current_cols = identify_current_columns(df_down)
        training_cols = [col for col in numeric_cols if col not in current_cols]
        if verbose and current_cols:
            print(f"Excluding current columns from training: {current_cols}")
    else:
        training_cols = numeric_cols
    
    data_numeric = df_down[training_cols].values.astype(np.float32)
    
    if verbose:
        print(f"Training features: {len(training_cols)} columns")
    
    # Standardization
    data_scaled, scaler = standardize_data(data_numeric)
    
    if verbose:
        print(f"Standardization completed")
    
    # Create sequences
    X, y = create_sequences(data_scaled, ws, fl)
    
    if verbose:
        print(f"Sequences created - X: {X.shape}, y: {y.shape}")
    
    # Split
    actual_test_size = test_size if test_size is not None else config.TEST_SIZE
    
    if actual_test_size > 0:
        X_train, X_test, y_train, y_test = split_sequences(
            X, y,
            test_size=actual_test_size,
            shuffle=config.SHUFFLE,
            random_state=config.RANDOM_SEED
        )
        if verbose:
            print(f"Split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    else:
        # No split, all data goes to training
        X_train, y_train = X, y
        X_test, y_test = np.array([]), np.array([])
        if verbose:
            print(f"No split - All data for training: {X_train.shape[0]} samples")
    
    # Also prepare current data for anomaly detection (if available)
    current_data = None
    if current_cols:
        current_data = df_down[current_cols].values.astype(np.float32)
        if verbose:
            print(f"Current data for anomaly detection: {current_cols}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': training_cols,
        'num_features': len(training_cols),
        'window_size': ws,
        'forecast_length': fl,
        'config': config,
        # Additional data for anomaly detection
        'current_cols': current_cols,
        'current_data': current_data,
        'all_numeric_cols': numeric_cols,
        'df_downsampled': df_down,  # Keep for anomaly detection
    }


def make_supervised_features(
    df: pd.DataFrame,
    lags: Tuple[int, ...] = (1, 2, 3, 5, 10),
    roll_windows: Tuple[int, ...] = (3, 5, 10),
    horizon: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create lag/rolling features and multi-step target for tabular models.
    
    Args:
        df: Input DataFrame
        lags: Lag features to create
        roll_windows: Windows for rolling statistics
        horizon: Number of future steps to predict
        
    Returns:
        Tuple (X, y) with features and target
    """
    df = df.copy().sort_index()
    df = df.ffill().bfill()
    
    feats = []
    
    # Lag features
    for L in lags:
        lag_df = df.shift(L)
        lag_df.columns = [f"{c}__lag{L}" for c in df.columns]
        feats.append(lag_df)
    
    # Rolling statistics
    for W in roll_windows:
        feats.append(df.rolling(W, min_periods=1).mean().add_suffix(f"__rmean{W}"))
        feats.append(df.rolling(W, min_periods=1).std().add_suffix(f"__rstd{W}"))
        feats.append(df.rolling(W, min_periods=1).min().add_suffix(f"__rmin{W}"))
        feats.append(df.rolling(W, min_periods=1).max().add_suffix(f"__rmax{W}"))
    
    X = pd.concat(feats, axis=1)
    
    # Multi-step target
    y = pd.concat([df.shift(-h) for h in range(1, horizon + 1)], axis=1)
    y.columns = [f"{c}_t+{h}" for h in range(1, horizon + 1) for c in df.columns]
    
    data = pd.concat([X, y], axis=1).dropna()
    X = data[X.columns]
    y = data[y.columns]
    
    return X, y


# Utility functions for domain-specific columns
def identify_temperature_columns(df: pd.DataFrame) -> List[str]:
    """Identify temperature columns."""
    return [col for col in df.columns if 'temp' in col.lower()]


def identify_vibration_columns(df: pd.DataFrame) -> List[str]:
    """Identify vibration columns."""
    return [col for col in df.columns if 'vib' in col.lower() or 'acc' in col.lower()]


def identify_current_columns(df: pd.DataFrame) -> List[str]:
    """Identify current columns."""
    return [col for col in df.columns if 'cur' in col.lower() or 'current' in col.lower()]


def get_time_series(df: pd.DataFrame, time_col: str = 'time') -> pd.Series:
    """
    Extract the time series from the DataFrame.
    
    Args:
        df: Input DataFrame
        time_col: Name of the time column
        
    Returns:
        Time series (or index if not present)
    """
    if time_col in df.columns:
        return df[time_col]
    else:
        return pd.Series(np.arange(len(df)), index=df.index)


if __name__ == "__main__":
    # Usage example
    print("Data Loader Module - Test")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    data = np.random.randn(n_samples, n_features)
    df_test = pd.DataFrame(
        data,
        columns=[f'sensor_{i}' for i in range(n_features)]
    )
    df_test['time'] = np.arange(n_samples)
    
    print(f"Test dataset created: {df_test.shape}")
    
    # Test pipeline with default values (same as main.py)
    result = prepare_data_for_training(
        df_test,
        window_size=2880,      # 24 hours at 30s intervals
        forecast_length=2880,  # 24 hours at 30s intervals
        downsample_factor=15,
        verbose=True
    )
    
    print(f"\nResult:")
    print(f"  X_train shape: {result['X_train'].shape}")
    print(f"  y_train shape: {result['y_train'].shape}")
    print(f"  Features: {result['feature_names']}")
