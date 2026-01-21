"""
Train & Optimize Module
=======================
Training loop, validation, hyperparameter optimization (Optuna).
Forecasting metrics and model saving.
"""

import os
import time
import random
from itertools import product
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error  # MAPE
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# MASE from sktime (optional)
try:
    from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
    SKTIME_AVAILABLE = True
except ImportError:
    SKTIME_AVAILABLE = False

# TensorFlow / Keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Optuna (optional)
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.2
    patience: int = 10
    learning_rate: float = 1e-3
    models_save_path: str = './pretrained_models'
    random_seed: int = 42
    # Early stopping parameters
    early_stopping_monitor: str = 'val_loss'
    early_stopping_mode: str = 'min'  # 'min' for loss, 'max' for accuracy
    early_stopping_min_delta: float = 0.0  # Minimum change to qualify as improvement
    restore_best_weights: bool = True
    # Learning rate reduction parameters
    reduce_lr_on_plateau: bool = True
    reduce_lr_factor: float = 0.5
    reduce_lr_patience: int = 5
    reduce_lr_min_lr: float = 1e-6


# =============================================================================
# Regression Metrics
# =============================================================================

def regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    sensor_names: Optional[List[str]] = None
) -> Dict:
    """
    Calculate global and per-feature regression metrics.
    
    Args:
        y_true: Actual values [samples, forecast_length, features] or [samples, features]
        y_pred: Predicted values (same shape as y_true)
        y_train: Training data (required for MASE calculation)
        sensor_names: Feature names (optional)
        
    Returns:
        Dictionary with global and per-feature metrics
    """
    # Flatten for global metrics
    if y_true.ndim == 3:
        y_true_f = y_true.reshape(-1, y_true.shape[-1])
        y_pred_f = y_pred.reshape(-1, y_pred.shape[-1])
    else:
        y_true_f = y_true
        y_pred_f = y_pred
    
    if y_train is not None and y_train.ndim == 3:
        y_train_f = y_train.reshape(-1, y_train.shape[-1])
    elif y_train is not None:
        y_train_f = y_train
    else:
        y_train_f = None

    mse = mean_squared_error(y_true_f, y_pred_f)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_f, y_pred_f)
    r2 = r2_score(y_true_f, y_pred_f)
    # MAPE: sklearn returns ratio, multiply by 100 for percentage
    mape = mean_absolute_percentage_error(y_true_f.flatten(), y_pred_f.flatten()) * 100
    
    # MASE requires y_train and sktime
    if y_train_f is not None and SKTIME_AVAILABLE:
        mase = mean_absolute_scaled_error(y_true_f.flatten(), y_pred_f.flatten(), y_train=y_train_f.flatten())
    else:
        mase = np.nan  # MASE not available without y_train

    report = {
        'global': {
            'RMSE': rmse,
            'R2': r2,
            'MAE': mae,
            'MAPE': mape,
            'MASE': mase,
        }
    }

    # Per-feature metrics
    n_features = y_true_f.shape[-1]
    if sensor_names is None:
        sensor_names = [f'feature_{i}' for i in range(n_features)]

    per_feature = {}
    for j in range(n_features):
        yt = y_true_f[:, j]
        yp = y_pred_f[:, j]
        mse_j = mean_squared_error(yt, yp)
        mae_j = mean_absolute_error(yt, yp)
        r2_j = r2_score(yt, yp)
        mape_j = mean_absolute_percentage_error(yt, yp) * 100
        
        if y_train_f is not None and SKTIME_AVAILABLE:
            yt_train = y_train_f[:, j] if y_train_f.ndim > 1 else y_train_f
            mase_j = mean_absolute_scaled_error(yt, yp, y_train=yt_train)
        else:
            mase_j = np.nan
            
        per_feature[sensor_names[j]] = {
            'RMSE': np.sqrt(mse_j),
            'R2': r2_j,
            'MAE': mae_j,
            'MAPE': mape_j,
            'MASE': mase_j,
        }
    report['per_feature'] = per_feature
    
    return report


def print_metrics(name: str, report: Dict):
    """Print global metrics from a report."""
    g = report['global']
    print(f"\n=== {name} ===")
    print(f"RMSE : {g['RMSE']:.6f}")
    print(f"R²   : {g['R2']:.6f}")
    print(f"MAE  : {g['MAE']:.6f}")
    print(f"MAPE : {g['MAPE']:.2f}%")
    print(f"MASE : {g['MASE']:.4f}")


def regression_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    sensor_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Create a DataFrame with metrics for each sensor.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_train: Training data (required for MASE calculation)
        sensor_names: Sensor names
        
    Returns:
        DataFrame with per-sensor metrics
    """
    if y_true.ndim == 3:
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    
    if y_train is not None and y_train.ndim == 3:
        y_train = y_train.reshape(-1, y_train.shape[-1])
    
    num_sensori = y_true.shape[1]
    if sensor_names is None:
        sensor_names = [f"Sensor_{i+1}" for i in range(num_sensori)]
    
    metrics = {
        'Sensor': sensor_names,
        'RMSE': [], 'R2': [], 'MAE': [], 'MAPE (%)': [], 'MASE': []
    }
    
    for i in range(num_sensori):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        metrics['RMSE'].append(np.sqrt(mse))
        metrics['R2'].append(r2_score(y_true[:, i], y_pred[:, i]))
        metrics['MAE'].append(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        metrics['MAPE (%)'].append(mean_absolute_percentage_error(y_true[:, i], y_pred[:, i]) * 100)
        
        if y_train is not None and SKTIME_AVAILABLE:
            y_train_col = y_train[:, i] if y_train.ndim > 1 else y_train
            metrics['MASE'].append(mean_absolute_scaled_error(y_true[:, i], y_pred[:, i], y_train=y_train_col))
        else:
            metrics['MASE'].append(np.nan)
    
    return pd.DataFrame(metrics)


# =============================================================================
# Keras Training
# =============================================================================

def train_keras_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Optional[TrainingConfig] = None,
    verbose: int = 1,
    save_path: Optional[str] = None
) -> Dict:
    """
    Train a Keras model with early stopping.
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_test, y_test: Test data
        config: Training configuration
        verbose: Verbosity level
        save_path: Path to save the model
        
    Returns:
        Dictionary with training results
    """
    if config is None:
        config = TrainingConfig()
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.patience,
            restore_best_weights=True,
            verbose=verbose
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=verbose
        )
    ]
    
    if save_path:
        callbacks.append(ModelCheckpoint(
            save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=verbose
        ))
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    
    training_time = time.time() - start_time
    epochs_trained = len(history.history['loss'])
    
    # Predictions and metrics
    y_pred = model.predict(X_test, verbose=0)
    report = regression_report(y_test, y_pred, y_train=y_train)
    
    result = {
        'model': model,
        'history': history.history,
        'epochs_trained': epochs_trained,
        'training_time': training_time,
        'y_pred': y_pred,
        'y_test': y_test,
        'report': report,
        'r2': report['global']['R2'],
        'rmse': report['global']['RMSE'],
    }
    
    if save_path:
        result['saved_path'] = save_path
    
    return result


def train_boosting(
    df: pd.DataFrame,
    horizon: int = 1,
    test_size: float = 0.2,
    learning_rate: float = 0.06,
    max_iter: int = 100,
    save_path: Optional[str] = None
) -> Dict:
    """
    Train HistGradientBoosting for tabular forecasting.
    
    Args:
        df: DataFrame with data
        horizon: Prediction horizon
        test_size: Test set proportion
        learning_rate: Learning rate
        max_iter: Maximum number of iterations
        save_path: Path to save the model
        
    Returns:
        Dictionary with results
    """
    from data_loader import make_supervised_features
    
    X, y = make_supervised_features(df, horizon=horizon)
    n = len(X)
    split = int(n * (1 - test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_iter=max_iter,
        early_stopping=True,
        random_state=42,
        verbose=0
    )
    model = MultiOutputRegressor(base)
    model.fit(X_train_scaled, y_train.values)

    if save_path:
        joblib.dump({'model': model, 'scaler': scaler}, save_path)
        print(f"Model saved to: {save_path}")

    y_pred = model.predict(X_test_scaled)
    num_features = df.shape[1]
    y_pred_seq = y_pred.reshape(-1, horizon, num_features)
    y_test_seq = y_test.values.reshape(-1, horizon, num_features)
    y_train_seq = y_train.values.reshape(-1, horizon, num_features) if hasattr(y_train, 'values') else y_train.reshape(-1, horizon, num_features)

    report = regression_report(y_test_seq, y_pred_seq, y_train=y_train_seq, sensor_names=df.columns.tolist())
    print_metrics("HistGradientBoosting", report)

    return {
        'model': model,
        'scaler': scaler,
        'y_test_seq': y_test_seq,
        'y_pred_seq': y_pred_seq,
        'report': report,
    }


# =============================================================================
# PyTorch Training (Transformer)
# =============================================================================

def train_transformer(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Dict:
    """
    Train a PyTorch Transformer model.
    
    Args:
        model: PyTorch model
        X_train, y_train: Training data
        X_test, y_test: Test data
        config: Training configuration
        device: PyTorch device (CPU/GPU)
        feature_names: Feature names
        save_path: Path to save weights
        
    Returns:
        Dictionary with results
    """
    from models import SequenceDataset
    
    if config is None:
        config = TrainingConfig()
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    train_ds = SequenceDataset(X_train, y_train)
    test_ds = SequenceDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, config.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item()
        val_loss /= len(test_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch}/{config.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best weights
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))

    # Final predictions
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            preds = model(xb)
            all_preds.append(preds.cpu().numpy())
    y_pred = np.concatenate(all_preds, axis=0)
    
    report = regression_report(y_test, y_pred, y_train=y_train, sensor_names=feature_names)
    print_metrics("Transformer", report)

    return {
        'model': model,
        'history': history,
        'epochs_trained': epoch,
        'y_pred': y_pred,
        'y_test': y_test,
        'report': report,
        'r2': report['global']['R2'],
    }


# =============================================================================
# Grid Search
# =============================================================================

class KerasGridSearch:
    """
    Grid search for Keras models with best model saving.
    """
    
    def __init__(
        self,
        data_scaled: np.ndarray,
        feature_names: List[str],
        config: Optional[TrainingConfig] = None
    ):
        self.data_scaled = data_scaled
        self.feature_names = feature_names
        self.config = config or TrainingConfig()
        self.results = []

    def create_sequences(
        self,
        window_size: int,
        forecast_horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences and train/test split."""
        from data_loader import create_sequences, split_sequences
        X, y = create_sequences(self.data_scaled, window_size, forecast_horizon)
        return split_sequences(X, y, test_size=0.2, shuffle=False)

    def run(
        self,
        model_name: str,
        build_fn: Callable,
        param_grid: Dict,
        max_configs: int = 10
    ) -> Tuple[Optional[Dict], List[Dict]]:
        """
        Run grid search on a model.
        
        Args:
            model_name: Model name
            build_fn: Function to build the model
            param_grid: Parameter grid
            max_configs: Maximum number of configurations to test
            
        Returns:
            Tuple (best result, all results)
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        all_combos = list(product(*values))
        
        if len(all_combos) > max_configs:
            print(f"[{model_name}] {len(all_combos)} combinations, using {max_configs} random")
            np.random.shuffle(all_combos)
            all_combos = all_combos[:max_configs]

        best = None
        best_r2 = -np.inf
        all_results = []
        num_features = len(self.feature_names)

        for i, combo in enumerate(all_combos, 1):
            cfg = dict(zip(keys, combo))
            window_size = cfg['window_size']
            forecast_horizon = cfg['forecast_horizon']
            batch_size = cfg.get('batch_size', self.config.batch_size)
            epochs = cfg.get('epochs', self.config.epochs)
            lr = cfg.get('learning_rate', self.config.learning_rate)
            dr = cfg.get('dropout_rate', 0.3)

            print(f"\n[{model_name}] Config {i}/{len(all_combos)}")
            print(f"  Window: {window_size}, Forecast: {forecast_horizon}")
            print(f"  Batch: {batch_size}, Epochs: {epochs}, LR: {lr}")

            X_train, X_test, y_train, y_test = self.create_sequences(window_size, forecast_horizon)

            # Costruisci modello
            extra = {k: v for k, v in cfg.items()
                     if k not in ['window_size', 'forecast_horizon', 'batch_size', 
                                  'epochs', 'learning_rate', 'dropout_rate']}
            
            model = build_fn(
                window_size=window_size,
                forecast_length=forecast_horizon,
                num_features=num_features,
                learning_rate=lr,
                dropout_rate=dr,
                **extra
            )

            # Training
            train_config = TrainingConfig(
                epochs=epochs,
                batch_size=batch_size,
                patience=self.config.patience
            )
            
            result = train_keras_model(
                model, X_train, y_train, X_test, y_test,
                config=train_config,
                verbose=0
            )
            
            result['model_name'] = model_name
            result['config'] = cfg
            all_results.append(result)

            r2 = result['r2']
            rmse = result['rmse']
            print(f"  R²: {r2:.4f} | RMSE: {rmse:.6f}")
            
            if r2 > best_r2:
                print(f"  >> New best! (previous: {best_r2:.4f})")
                best_r2 = r2
                best = result

        # Save best model
        if best is not None:
            os.makedirs(self.config.models_save_path, exist_ok=True)
            cfg = best['config']
            save_name = f"{model_name}_best_W{cfg['window_size']}_F{cfg['forecast_horizon']}_epochs{best['epochs_trained']}.h5"
            save_path = os.path.join(self.config.models_save_path, save_name)
            best['model'].save(save_path)
            best['saved_path'] = save_path
            print(f"\n[{model_name}] Best model saved: {save_path}")

        self.results = all_results
        return best, all_results


def run_grid_search(
    model_name: str,
    build_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    window_size: int,
    forecast_length: int,
    num_features: int,
    param_grid: Dict[str, List],
    max_configs: int = 10,
    epochs: int = 20,
    batch_size: int = 32,
    patience: int = 5
) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Run grid search on a model with pre-created sequences.
    
    Simpler interface than KerasGridSearch class.
    
    Args:
        model_name: Model name for logging
        build_fn: Function to build model(window_size, forecast_length, num_features, learning_rate, dropout_rate)
        X_train, y_train, X_test, y_test: Pre-split data
        window_size, forecast_length, num_features: Model dimensions
        param_grid: Dict with parameter lists, e.g. {'learning_rate': [1e-3, 1e-4], 'dropout_rate': [0.2, 0.3]}
        max_configs: Maximum configurations to test
        epochs, batch_size, patience: Training parameters
        
    Returns:
        Tuple of (best_result dict, all_results list)
    """
    from itertools import product
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combos = list(product(*values))
    
    if len(all_combos) > max_configs:
        print(f"[{model_name}] {len(all_combos)} combinations, testing {max_configs} random")
        np.random.shuffle(all_combos)
        all_combos = all_combos[:max_configs]
    else:
        print(f"[{model_name}] Testing {len(all_combos)} configurations")
    
    best_result = None
    best_rmse = np.inf
    all_results = []
    
    for i, combo in enumerate(all_combos, 1):
        cfg = dict(zip(keys, combo))
        lr = cfg.get('learning_rate', 1e-3)
        dr = cfg.get('dropout_rate', 0.3)
        
        print(f"\n  [{i}/{len(all_combos)}] lr={lr:.0e}, dropout={dr:.2f}", end=" ")
        
        try:
            model = build_fn(
                window_size=window_size,
                forecast_length=forecast_length,
                num_features=num_features,
                learning_rate=lr,
                dropout_rate=dr
            )
            
            train_config = TrainingConfig(
                epochs=epochs,
                batch_size=batch_size,
                patience=patience
            )
            
            result = train_keras_model(
                model, X_train, y_train, X_test, y_test,
                config=train_config,
                verbose=0
            )
            
            result['learning_rate'] = lr
            result['dropout_rate'] = dr
            result['config'] = cfg
            all_results.append(result)
            
            rmse = result['rmse']
            r2 = result['r2']
            print(f"→ RMSE={rmse:.6f}, R²={r2:.4f}", end="")
            
            if rmse < best_rmse:
                print(" ★ BEST")
                best_rmse = rmse
                best_result = result
            else:
                print()
                
        except Exception as e:
            print(f"→ ERROR: {e}")
    
    if best_result:
        print(f"\n[{model_name}] Best: lr={best_result['learning_rate']:.0e}, "
              f"dropout={best_result['dropout_rate']:.2f}, RMSE={best_result['rmse']:.6f}")
    
    return best_result, all_results


# =============================================================================
# Optuna Optimization
# =============================================================================

def create_optuna_objective(
    build_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    window_size: int,
    forecast_length: int,
    num_features: int,
    param_space: Dict[str, Dict]
) -> Callable:
    """
    Create an objective function for Optuna.
    
    Args:
        build_fn: Function to build the model
        X_train, y_train: Training data
        window_size, forecast_length, num_features: Model parameters
        param_space: Parameter space (e.g.: {'filters': {'low': 32, 'high': 128}})
        
    Returns:
        Objective function for Optuna
    """
    def objective(trial):
        # Suggest parameters
        params = {}
        for name, space in param_space.items():
            if 'choices' in space:
                params[name] = trial.suggest_categorical(name, space['choices'])
            elif 'low' in space and 'high' in space:
                if isinstance(space['low'], float):
                    log = space.get('log', False)
                    params[name] = trial.suggest_float(name, space['low'], space['high'], log=log)
                else:
                    step = space.get('step', 1)
                    params[name] = trial.suggest_int(name, space['low'], space['high'], step=step)
        
        # Build and train model
        model = build_fn(
            window_size=window_size,
            forecast_length=forecast_length,
            num_features=num_features,
            **params
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
        )
        
        return history.history["val_loss"][-1]
    
    return objective


def run_optuna_optimization(
    model_name: str,
    build_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    window_size: int,
    forecast_length: int,
    num_features: int,
    param_space: Dict[str, Dict],
    n_trials: int = 20,
    timeout: Optional[int] = None
) -> Optional[Dict]:
    """
    Run hyperparameter optimization with Optuna.
    
    Args:
        model_name: Model name
        build_fn: Function to build the model
        X_train, y_train: Training data
        window_size, forecast_length, num_features: Model parameters
        param_space: Parameter space
        n_trials: Number of trials
        timeout: Timeout in seconds (optional)
        
    Returns:
        Dictionary with results or None if Optuna not available
    """
    if not OPTUNA_AVAILABLE:
        print("Optuna not available. Install with: pip install optuna")
        return None
    
    objective = create_optuna_objective(
        build_fn, X_train, y_train,
        window_size, forecast_length, num_features,
        param_space
    )
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    print(f"\n[{model_name}] Optimization completed")
    print(f"  Best val_loss: {study.best_value:.6f}")
    print(f"  Best params: {study.best_params}")
    
    return {
        'study': study,
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials)
    }


# =============================================================================
# Utilities
# =============================================================================

def set_random_seeds(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except:
        pass


def save_training_results(
    results: Dict,
    save_dir: str,
    model_name: str
):
    """
    Save training results.
    
    Args:
        results: Dictionary with results
        save_dir: Save directory
        model_name: Model name
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save history
    if 'history' in results:
        history_path = os.path.join(save_dir, f"{model_name}_history.csv")
        pd.DataFrame(results['history']).to_csv(history_path, index=False)
    
    # Save metrics
    if 'report' in results:
        metrics_path = os.path.join(save_dir, f"{model_name}_metrics.json")
        import json
        with open(metrics_path, 'w') as f:
            json.dump(results['report'], f, indent=2)


if __name__ == "__main__":
    # Module test
    print("Train & Optimize Module - Test")
    print("=" * 50)
    
    # Create sample data
    # Using smaller values for quick testing
    # Real defaults: window_size=2880, forecast_length=2880
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    window_size = 288  # ~2.4 hours at 30s sampling (smaller for test)
    forecast_length = 288  # same as window_size (smaller for test)
    
    # Simulate sequences
    X = np.random.randn(n_samples, window_size, n_features).astype(np.float32)
    y = np.random.randn(n_samples, forecast_length, n_features).astype(np.float32)
    
    # Split
    split = int(n_samples * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # Test metrics
    y_pred_test = y_test + np.random.randn(*y_test.shape) * 0.1
    report = regression_report(y_test, y_pred_test, y_train=y_train)
    print_metrics("Test", report)
    
    print("\nTest completed!")
