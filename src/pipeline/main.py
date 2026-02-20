#!/usr/bin/env python
"""
Main Script - Training and Forecasting Pipeline
==============================================

Allows to:
- Select which models to train/use
- Load pretrained models if existing
- Save trained models in .h5 format
- Run anomaly detection on results

Usage:
    python main.py --models cnn cnn_lstm --data path/to/data.csv
    python main.py --all --data path/to/data.csv
    python main.py --list-models
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Import moduli locali
from data_loader import (
    load_csv, prepare_data_for_training, DataConfig,
    get_numeric_columns, identify_temperature_columns, identify_vibration_columns
)
from models import (
    get_model, get_available_models, get_all_available_models, is_tcn_available,
    train_boosting, load_boosting_model, predict_boosting
)
from train_optimize import (
    TrainingConfig, train_keras_model, train_transformer, regression_report,
    print_metrics, set_random_seeds, regression_report_df,
    run_optuna_optimization, OPTUNA_AVAILABLE, run_grid_search
)
import torch
from anomaly_detection import (
    AnomalyConfig, compute_forecast_residuals, detect_anomalies_from_residuals,
    analyze_dataframe_anomalies, compute_health_index, save_thresholds, load_thresholds,
    ForecastAnomalyConfig, calibrate_thresholds_from_residuals,
    validate_thresholds_on_test, apply_thresholds_to_residuals,
    create_anomaly_dataframe, combine_thresholds_weighted,
    DEFAULT_CUSUM_COMBINATIONS, compute_cusum_multi_params, select_best_cusum_params,
    plot_anomalies_with_colors, plot_all_features_anomalies, plot_anomaly_summary
)
from utils import (
    configure_gpu, ensure_dir, save_json, print_separator,
    format_time, get_memory_usage
)


# =============================================================================
# Default Configuration
# =============================================================================

# Default dataset paths (relative to notebooks/files or absolute)
# 
# Dataset descriptions:
#   - caratterizzazione: Lab data, NORMAL operation - used to learn baseline thresholds
#   - fatica: Lab data, STRESS TEST operation - used to validate thresholds detect anomalies  
#   - real_data (tampieri): REAL MACHINE in production - actual anomaly detection target
#
DEFAULT_DATASETS = {
    # Lab data - normal operation baseline for threshold calibration
    #'caratterizzazione': '../../data/final/bonfiglioli_caratterizzazione_continuous.csv',
    # Lab data - stress test for threshold validation
    #'fatica': '../../data/final/bonfiglioli_fatica_continuous.csv',
    # Real machine data (Tampieri) - production anomaly detection
    #'real_data': '../../data/final/tampieri_continuous.csv',

    'caratterizzazione': '/home/projects/safe/data/final/bonfiglioli_caratterizzazione_continuous.csv',
    'fatica': '/home/projects/safe/data/final/bonfiglioli_fatica_continuous.csv',
    'real_data': '/home/projects/safe/data/final/tampieri_continuous.csv',
}

DEFAULT_CONFIG = {
    # ==========================================================================
    # Temporal Configuration: 24-hour window sampled every 30 seconds
    # ==========================================================================
    # Original data sampling: every 2 seconds
    # Target sampling: every 30 seconds (downsample factor = 15)
    # Window duration: 24 hours = 24 * 60 * 60 / 30 = 2880 samples
    # ==========================================================================
    'window_size': 2880,  # 24h at 30s intervals
    'forecast_length': 2880,  # 24h forecast (same as input window)
    'downsample_factor': 15,  # from 2s to 30s (2 * 15 = 30)
    # Forecast accuracy note:
    #   - Hours 1-6: fairly accurate prediction
    #   - Hours 6-12: degrading accuracy
    #   - Hours 12-24: essentially "noise" / statistical mean
    'epochs': 50,
    'batch_size': 32,
    'patience': 10,
    'learning_rate': 1e-3,
    'dropout_rate': 0.3,
    'random_seed': 42,
    'models_dir': './pretrained_models',
    'results_dir': './results',
    # Dataset split (train/val/test)
    'train_split': 0.7,  # 70% for training
    'val_split': 0.1,    # 10% for validation
    'test_split': 0.2,   # 20% for test
    # Default dataset paths
    'calib_data': DEFAULT_DATASETS['caratterizzazione'],
    'fatica_data': DEFAULT_DATASETS['fatica'],
    'apply_data': DEFAULT_DATASETS['real_data'],
    # Default thresholds file (auto-generated from calibration)
    'thresholds_file': './results/anomaly_thresholds.json',
}


# =============================================================================
# Pretrained Models Management
# =============================================================================

# PyTorch models list
PYTORCH_MODELS = ['transformer']


def is_pytorch_model(model_name: str) -> bool:
    """Check if the model is a PyTorch model."""
    return model_name.lower() in PYTORCH_MODELS


def get_model_path(
    models_dir: str,
    model_name: str,
    window_size: int,
    forecast_length: int,
    epochs: int = None,
    downsample_factor: int = None,
    learning_rate: float = None
) -> str:
    """
    Generate the standard path for a model.
    
    Naming convention:
        {model}_W{window}_F{forecast}_E{epochs}_D{downsample}_LR{lr}.{ext}
    
    Example:
        cnn_W2880_F2880_E50_D15_LR0.001.keras
    """
    ext = '.pt' if is_pytorch_model(model_name) else '.keras'
    
    # Build name parts
    parts = [model_name, f"W{window_size}", f"F{forecast_length}"]
    
    if epochs is not None:
        parts.append(f"E{epochs}")
    
    if downsample_factor is not None:
        parts.append(f"D{downsample_factor}")
    
    if learning_rate is not None:
        # Format learning rate (e.g., 0.001 -> LR1e-3)
        lr_str = f"{learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        parts.append(f"LR{lr_str}")
    
    filename = "_".join(parts) + ext
    return os.path.join(models_dir, filename)


def find_best_model(
    models_dir: str,
    model_name: str,
    window_size: int,
    forecast_length: int
) -> str:
    """
    Find the best available pretrained model matching the parameters.
    
    Searches for models with matching window_size and forecast_length,
    regardless of epochs/downsample/lr. Returns the most recently modified.
    """
    ext = '.pt' if is_pytorch_model(model_name) else '.keras'
    pattern = f"{model_name}_W{window_size}_F{forecast_length}*{ext}"
    
    matches = list(Path(models_dir).glob(pattern))
    
    if not matches:
        return None
    
    # Return most recently modified
    return str(max(matches, key=lambda p: p.stat().st_mtime))


def model_exists(model_path: str) -> bool:
    """Check if a pretrained model exists."""
    return os.path.exists(model_path)


def load_pretrained_model(model_path: str, model_name: str = None, **model_kwargs):
    """
    Load a pretrained model (Keras or PyTorch).
    
    Args:
        model_path: Path to the saved model
        model_name: Model name (needed for PyTorch to rebuild architecture)
        **model_kwargs: Additional kwargs for PyTorch model rebuilding
    """
    print(f"  Loading pretrained model: {model_path}")
    
    if model_name and is_pytorch_model(model_name):
        # PyTorch model - need to rebuild architecture first
        model = get_model(model_name, **model_kwargs)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    else:
        # Keras model
        from keras.models import load_model
        try:
            model = load_model(model_path, compile=True)
        except Exception as e:
            # Try loading with compile=False and recompile
            print(f"  Warning: Could not load with compile=True, trying compile=False...")
            model = load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model


def save_model(model, model_path: str, model_name: str = None):
    """
    Save a trained model (Keras or PyTorch).
    
    Args:
        model: The model to save
        model_path: Path to save the model
        model_name: Model name to determine save method
    """
    ensure_dir(os.path.dirname(model_path))
    
    if model_name and is_pytorch_model(model_name):
        # PyTorch model
        torch.save(model.state_dict(), model_path)
    else:
        # Keras model
        model.save(model_path)
    
    print(f"  Model saved: {model_path}")


# =============================================================================
# Training Pipeline
# =============================================================================

def train_or_load_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict,
    force_retrain: bool = False
) -> Tuple[object, Dict]:
    """
    Train a model or load the pretrained one.
    
    Args:
        model_name: Model name
        X_train, y_train, X_test, y_test: Data
        config: Configuration
        force_retrain: If True, retrain even if exists
        
    Returns:
        Tuple (model, results)
    """
    window_size = X_train.shape[1]
    forecast_length = y_train.shape[1]
    num_features = X_train.shape[2]
    
    # Get training parameters
    epochs = config.get('epochs', 50)
    downsample_factor = config.get('downsample_factor', 1)
    lr = config.get(f'{model_name}_learning_rate', config.get('learning_rate', 1e-3))
    
    # Generate model path with all parameters
    model_path = get_model_path(
        config['models_dir'],
        model_name,
        window_size,
        forecast_length,
        epochs=epochs,
        downsample_factor=downsample_factor,
        learning_rate=lr
    )
    
    results = {
        'model_name': model_name,
        'window_size': window_size,
        'forecast_length': forecast_length,
        'num_features': num_features,
        'epochs': epochs,
        'downsample_factor': downsample_factor,
        'learning_rate': lr,
        'model_path': model_path,
    }
    
    # Check if exact model exists
    if model_exists(model_path) and not force_retrain:
        print(f"\n[{model_name}] Pretrained model found (exact match)!")
        model = load_pretrained_model(
            model_path, 
            model_name=model_name,
            window_size=window_size,
            forecast_length=forecast_length,
            num_features=num_features
        )
        results['pretrained'] = True
        results['trained_epochs'] = 'N/A (pretrained)'
    
    # Check for any compatible model (same window/forecast, different epochs)
    elif not force_retrain:
        best_model_path = find_best_model(
            config['models_dir'],
            model_name,
            window_size,
            forecast_length
        )
        
        if best_model_path:
            print(f"\n[{model_name}] Compatible pretrained model found!")
            print(f"  Loading: {os.path.basename(best_model_path)}")
            print(f"  (Requested epochs={epochs}, but using existing model)")
            model = load_pretrained_model(
                best_model_path,
                model_name=model_name,
                window_size=window_size,
                forecast_length=forecast_length,
                num_features=num_features
            )
            results['pretrained'] = True
            results['trained_epochs'] = 'N/A (pretrained)'
            results['model_path'] = best_model_path
        else:
            # No model found, need to train
            model = None
    else:
        model = None
    
    # Train if no pretrained model found
    if model is None:
        print(f"\n[{model_name}] Training new model...")
        print(f"  Parameters: epochs={epochs}, downsample={downsample_factor}, lr={lr}")
        
        # Get dropout rate
        dr = config.get(f'{model_name}_dropout_rate', config.get('dropout_rate', 0.3))
        
        if config.get('optimize', 'none') != 'none' and f'{model_name}_learning_rate' in config:
            print(f"  Using Optuna-optimized params: lr={lr:.6f}, dropout={dr:.2f}")
        
        # Build model
        try:
            model = get_model(
                model_name,
                window_size=window_size,
                forecast_length=forecast_length,
                num_features=num_features,
                learning_rate=lr,
                dropout_rate=dr
            )
        except Exception as e:
            print(f"  ERROR building model: {e}")
            return None, {'error': str(e), 'model_name': model_name}
        
        # Training config
        train_config = TrainingConfig(
            epochs=config.get('epochs', 50),
            batch_size=config.get('batch_size', 32),
            patience=config.get('patience', 10),
            learning_rate=lr
        )
        
        # Train (different method for PyTorch vs Keras)
        try:
            if is_pytorch_model(model_name):
                # PyTorch training (Transformer)
                train_result = train_transformer(
                    model,
                    X_train, y_train,
                    X_test, y_test,
                    config=train_config,
                    save_path=model_path
                )
            else:
                # Keras training
                train_result = train_keras_model(
                    model,
                    X_train, y_train,
                    X_test, y_test,
                    config=train_config,
                    verbose=1
                )
                # Save model
                save_model(model, model_path, model_name=model_name)
            
            results['pretrained'] = False
            results['trained_epochs'] = train_result['epochs_trained']
            results['training_time'] = train_result.get('training_time', 'N/A')
            results['history'] = train_result['history']
            
        except Exception as e:
            print(f"  ERROR training: {e}")
            return None, {'error': str(e), 'model_name': model_name}
    
    # Evaluation
    print(f"  Evaluation on test set...")
    
    if is_pytorch_model(model_name):
        # PyTorch evaluation
        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_test).float()
            y_pred = model(X_tensor).numpy()
    else:
        # Keras evaluation
        y_pred = model.predict(X_test, verbose=0)
    
    # Calculate metrics (pass y_train for MASE calculation)
    report = regression_report(y_test, y_pred, y_train=y_train)
    
    results['y_pred'] = y_pred
    results['report'] = report
    results['r2'] = report['global']['R2']
    results['rmse'] = report['global']['RMSE']
    results['mae'] = report['global']['MAE']
    results['mape'] = report['global']['MAPE']
    results['mase'] = report['global']['MASE']
    
    # Print all metrics
    print(f"  RMSE: {results['rmse']:.6f}")
    print(f"  RÂ²:   {results['r2']:.4f}")
    print(f"  MAE:  {results['mae']:.6f}")
    print(f"  MAPE: {results['mape']:.2f}%")
    print(f"  MASE: {results['mase']:.4f}")
    
    return model, results


def run_pipeline(
    data_path: str,
    models_to_run: List[str],
    config: Dict,
    force_retrain: bool = False,
    run_anomaly_detection: bool = True
) -> Dict:
    """
    Run the complete pipeline.
    
    Args:
        data_path: Data file path
        models_to_run: List of models to train/use
        config: Configuration
        force_retrain: If True, retrain all models
        run_anomaly_detection: If True, run anomaly detection
        
    Returns:
        Dictionary with all results
    """
    print_separator("FORECASTING + ANOMALY DETECTION PIPELINE")
    print(f"Data: {data_path}")
    print(f"Models: {models_to_run}")
    print(f"Force retrain: {force_retrain}")
    
    # Setup
    set_random_seeds(config.get('random_seed', 42))
    configure_gpu()
    ensure_dir(config['models_dir'])
    ensure_dir(config['results_dir'])
    
    # Data loading
    print_separator("DATA LOADING")
    
    # Check if separate datasets are provided
    train_data_path = config.get('train_data')
    val_data_path = config.get('val_data')
    test_data_path = config.get('test_data')
    
    use_separate_datasets = any([train_data_path, val_data_path, test_data_path])
    
    if use_separate_datasets:
        # Load separate datasets
        print("Loading separate train/val/test datasets...")
        
        if train_data_path:
            print(f"  Train: {train_data_path}")
            df_train = load_csv(train_data_path, verbose=False)
        else:
            df_train = None
            
        if val_data_path:
            print(f"  Val:   {val_data_path}")
            df_val = load_csv(val_data_path, verbose=False)
        else:
            df_val = None
            
        if test_data_path:
            print(f"  Test:  {test_data_path}")
            df_test = load_csv(test_data_path, verbose=False)
        else:
            df_test = None
        
        # Use main data file for any missing datasets
        if not all([df_train is not None, df_test is not None]):
            print(f"  Main:  {data_path} (for missing splits)")
            df_main = load_csv(data_path, verbose=False)
            if df_train is None:
                df_train = df_main
            if df_test is None:
                df_test = df_main
        
        # Save reference for boosting (needs raw DataFrame)
        raw_df_for_boosting = df_train.copy()
    else:
        # Single dataset with automatic split
        df = load_csv(data_path, verbose=True)
        train_split = config.get('train_split', 0.7)
        val_split = config.get('val_split', 0.15)
        test_split = config.get('test_split', 0.15)
        print(f"\nDataset split: train={train_split:.0%}, val={val_split:.0%}, test={test_split:.0%}")
        
        # Save reference for boosting (needs raw DataFrame)
        raw_df_for_boosting = df.copy()
    
    # Data preparation
    print_separator("DATA PREPARATION")
    data_config = DataConfig()
    data_config.DOWNSAMPLE_FACTOR = config.get('downsample_factor', 1)
    
    if use_separate_datasets:
        # Prepare separate datasets
        prepared = prepare_data_for_training(
            df_train,
            config=data_config,
            window_size=config.get('window_size', 30),
            forecast_length=config.get('forecast_length', 10),
            downsample_factor=config.get('downsample_factor', 1),
            exclude_current=config.get('exclude_current', True),
            test_size=0.0,  # No internal split, use external test data
            verbose=True
        )
        
        # Get all training data as X_train, y_train
        X_train = prepared['X_train']
        y_train = prepared['y_train']
        feature_names = prepared['feature_names']
        scaler = prepared['scaler']
        
        # Prepare test data separately using the same scaler
        prepared_test = prepare_data_for_training(
            df_test,
            config=data_config,
            window_size=config.get('window_size', 30),
            forecast_length=config.get('forecast_length', 10),
            downsample_factor=config.get('downsample_factor', 1),
            exclude_current=config.get('exclude_current', True),
            test_size=0.0,
            verbose=False
        )
        X_test = prepared_test['X_train']  # All data goes to "train" when test_size=0
        y_test = prepared_test['y_train']
        
        # Prepare validation data if provided
        if df_val is not None:
            prepared_val = prepare_data_for_training(
                df_val,
                config=data_config,
                window_size=config.get('window_size', 30),
                forecast_length=config.get('forecast_length', 10),
                downsample_factor=config.get('downsample_factor', 1),
                exclude_current=config.get('exclude_current', True),
                test_size=0.0,
                verbose=False
            )
            X_val = prepared_val['X_train']
            y_val = prepared_val['y_train']
        else:
            X_val, y_val = None, None
            
        current_cols = prepared.get('current_cols', [])
        current_data = prepared.get('current_data')
    else:
        # Single dataset with automatic split
        prepared = prepare_data_for_training(
            df,
            config=data_config,
            window_size=config.get('window_size', 30),
            forecast_length=config.get('forecast_length', 10),
            downsample_factor=config.get('downsample_factor', 1),
            exclude_current=config.get('exclude_current', True),
            test_size=config.get('val_split', 0.15) + config.get('test_split', 0.15),
            verbose=True
        )
        
        X_train = prepared['X_train']
        X_test = prepared['X_test']
        y_train = prepared['y_train']
        y_test = prepared['y_test']
        feature_names = prepared['feature_names']
        scaler = prepared['scaler']
        
        # Further split test into val and test if val_split > 0
        val_split = config.get('val_split', 0.15)
        test_split = config.get('test_split', 0.15)
        if val_split > 0 and test_split > 0:
            val_ratio = val_split / (val_split + test_split)
            split_idx = int(len(X_test) * val_ratio)
            X_val = X_test[:split_idx]
            y_val = y_test[:split_idx]
            X_test = X_test[split_idx:]
            y_test = y_test[split_idx:]
        else:
            X_val, y_val = None, None
        
        current_cols = prepared.get('current_cols', [])
        current_data = prepared.get('current_data')
    
    # Print data summary
    if current_cols:
        print(f"\nCurrent columns (excluded from training, used for anomaly detection):")
        print(f"  {current_cols}")
    
    print(f"\nPrepared data (training features):")
    print(f"  X_train: {X_train.shape}")
    if X_val is not None:
        print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    if y_val is not None:
        print(f"  y_val:   {y_val.shape}")
    print(f"  y_test:  {y_test.shape}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Feature names: {feature_names}")
    
    # Hyperparameter optimization (if enabled)
    optimize_method = config.get('optimize', 'none')
    
    if optimize_method == 'optuna':
        print_separator("OPTUNA HYPERPARAMETER OPTIMIZATION")
        print(f"Trials: {config.get('n_trials', 20)}")
        if config.get('timeout'):
            print(f"Timeout: {config['timeout']} seconds")
        
        if not OPTUNA_AVAILABLE:
            print("WARNING: Optuna not installed. Skipping optimization.")
            print("Install with: pip install optuna")
        else:
            param_space = {
                'learning_rate': {'low': 1e-5, 'high': 1e-2, 'log': True},
                'dropout_rate': {'low': 0.1, 'high': 0.5},
            }
            
            optimized_params = {}
            for model_name in models_to_run:
                if is_pytorch_model(model_name):
                    print(f"\n[{model_name}] Skipping (PyTorch not yet supported)")
                    continue
                    
                print(f"\n[{model_name}] Running Optuna optimization...")
                
                opt_result = run_optuna_optimization(
                    model_name=model_name,
                    build_fn=lambda **kwargs: get_model(model_name, **kwargs),
                    X_train=X_train,
                    y_train=y_train,
                    window_size=config['window_size'],
                    forecast_length=config['forecast_length'],
                    num_features=X_train.shape[2],
                    param_space=param_space,
                    n_trials=config.get('n_trials', 20),
                    timeout=config.get('timeout')
                )
                
                if opt_result:
                    optimized_params[model_name] = opt_result['best_params']
                    print(f"  Best params: {opt_result['best_params']}")
                    config[f'{model_name}_learning_rate'] = opt_result['best_params'].get('learning_rate', config['learning_rate'])
                    config[f'{model_name}_dropout_rate'] = opt_result['best_params'].get('dropout_rate', config['dropout_rate'])
    
    elif optimize_method == 'grid':
        print_separator("GRID SEARCH HYPERPARAMETER OPTIMIZATION")
        max_configs = config.get('max_configs', 10)
        print(f"Max configurations: {max_configs}")
        
        # Parameter grid for grid search
        param_grid = {
            'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
            'dropout_rate': [0.2, 0.3, 0.4],
        }
        
        for model_name in models_to_run:
            if is_pytorch_model(model_name):
                print(f"\n[{model_name}] Skipping (PyTorch not yet supported)")
                continue
                
            print(f"\n[{model_name}] Running grid search...")
            
            best_result, all_results = run_grid_search(
                model_name=model_name,
                build_fn=lambda **kwargs: get_model(model_name, **kwargs),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                window_size=config['window_size'],
                forecast_length=config['forecast_length'],
                num_features=X_train.shape[2],
                param_grid=param_grid,
                max_configs=max_configs,
                epochs=config.get('epochs', 20),
                batch_size=config.get('batch_size', 32),
                patience=config.get('patience', 5)
            )
            
            if best_result:
                config[f'{model_name}_learning_rate'] = best_result['learning_rate']
                config[f'{model_name}_dropout_rate'] = best_result['dropout_rate']
    
    # Model training
    print_separator("TRAINING/LOADING MODELS")
    
    all_results = {}
    trained_models = {}
    
    for model_name in models_to_run:
        # Handle boosting model separately (sklearn, not Keras/PyTorch)
        if model_name.lower() == 'boosting':
            print(f"\n[{model_name}] Training HistGradientBoosting...")
            try:
                # Need original DataFrame for boosting (not windowed data)
                # Use feature_names to select numeric columns
                df_for_boosting = raw_df_for_boosting[feature_names].copy()
                
                boosting_result = train_boosting(
                    df_for_boosting,
                    horizon=config['forecast_length'],
                    test_size=0.2,
                    learning_rate=config.get('boosting_learning_rate', 0.06),
                    max_iter=config.get('boosting_max_iter', 100),
                    max_depth=config.get('boosting_max_depth', None),
                    save_path=os.path.join(config['models_dir'], f"boosting_h{config['forecast_length']}")
                )
                
                trained_models[model_name] = boosting_result['model']
                all_results[model_name] = {
                    'model_name': model_name,
                    'rmse': np.sqrt(boosting_result['report']['mse']),
                    'mae': boosting_result['report']['mae'],
                    'r2': boosting_result['report']['r2'],
                    'mape': 0.0,  # TODO: calculate if needed
                    'mase': 0.0,  # TODO: calculate if needed
                    'pretrained': False,
                    'trained_epochs': f"{config.get('boosting_max_iter', 100)} iters"
                }
                print(f"  RÂ² = {boosting_result['report']['r2']:.4f}")
                print(f"  RMSE = {np.sqrt(boosting_result['report']['mse']):.4f}")
            except Exception as e:
                print(f"  ERROR training boosting: {e}")
                import traceback
                traceback.print_exc()
            continue
        
        # Check availability for neural network models
        available = get_available_models()
        if model_name not in available:
            print(f"\n[{model_name}] SKIPPED - Not available")
            print(f"  Available models: {available}")
            if model_name in ['tcn', 'tcn_lstm'] and not is_tcn_available():
                print(f"  (TCN requires: pip install keras-tcn)")
            continue
        
        model, results = train_or_load_model(
            model_name,
            X_train, y_train,
            X_test, y_test,
            config,
            force_retrain=force_retrain
        )
        
        if model is not None:
            trained_models[model_name] = model
            all_results[model_name] = results
    
    # Performance summary
    print_separator("PERFORMANCE SUMMARY")
    
    summary_data = []
    for name, res in all_results.items():
        if 'error' not in res:
            summary_data.append({
                'Model': name,
                'RMSE': res['rmse'],
                'RÂ²': res['r2'],
                'MAE': res['mae'],
                'MAPE (%)': res['mape'],
                'MASE': res['mase'],
                'Pretrained': res.get('pretrained', False),
                'Epochs': res.get('trained_epochs', 'N/A')
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('RÂ²', ascending=False)
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_path = os.path.join(config['results_dir'], 'model_comparison.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved: {summary_path}")
    
    # Anomaly Detection (on best model)
    if run_anomaly_detection and trained_models:
        print_separator("ANOMALY DETECTION")
        
        # Find best model (by RÂ²)
        best_model_name = max(all_results.keys(), key=lambda k: all_results[k].get('r2', -999))
        best_results = all_results[best_model_name]
        
        print(f"Best model: {best_model_name}")
        print(f"  RMSE: {best_results['rmse']:.6f}")
        print(f"  RÂ²:   {best_results['r2']:.4f}")
        print(f"  MAE:  {best_results['mae']:.6f}")
        print(f"  MAPE: {best_results['mape']:.2f}%")
        print(f"  MASE: {best_results['mase']:.4f}")
        
        y_pred = best_results['y_pred']
        
        # Calculate residuals
        print(f"\nCalculating residuals...")
        residuals = compute_forecast_residuals(y_test, y_pred)
        
        # Reshape for anomaly detection
        residuals_flat = residuals.reshape(-1, residuals.shape[-1])
        
        # Residual statistics
        print(f"  Residual shape: {residuals_flat.shape}")
        print(f"  Residual mean: {np.mean(residuals_flat):.6f}")
        print(f"  Residual std:  {np.std(residuals_flat):.6f}")
        
        # Detect anomalies
        print(f"\nDetecting anomalies (3-sigma method)...")
        anomalies, anom_info = detect_anomalies_from_residuals(
            residuals_flat,
            method='sigma',
            threshold=3.0
        )
        
        n_anomalies = anomalies.sum()
        total_points = anomalies.size
        pct_anomalies = n_anomalies / total_points * 100
        
        print(f"\n  Total data points: {total_points}")
        print(f"  Anomalies detected: {n_anomalies}")
        print(f"  Anomaly rate: {pct_anomalies:.2f}%")
        
        # Per-feature anomaly breakdown
        if anomalies.ndim > 1:
            print(f"\n  Per-feature anomalies:")
            for i, fname in enumerate(feature_names):
                n_anom_feat = anomalies[:, i].sum()
                pct_feat = n_anom_feat / anomalies.shape[0] * 100
                print(f"    {fname}: {n_anom_feat} ({pct_feat:.2f}%)")
        
        # Save anomaly detection results
        anom_results = {
            'model_used': best_model_name,
            'model_metrics': {
                'rmse': float(best_results['rmse']),
                'r2': float(best_results['r2']),
                'mae': float(best_results['mae']),
                'mape': float(best_results['mape']),
                'mase': float(best_results['mase']),
            },
            'total_data_points': int(total_points),
            'total_anomalies': int(n_anomalies),
            'anomaly_percentage': float(pct_anomalies),
            'detection_method': anom_info['method'],
            'threshold': float(anom_info.get('k', 3.0)),
            'residual_stats': {
                'mean': float(np.mean(residuals_flat)),
                'std': float(np.std(residuals_flat)),
            }
        }
        
        anom_path = os.path.join(config['results_dir'], 'anomaly_detection_results.json')
        save_json(anom_results, anom_path)
        print(f"\nResults saved: {anom_path}")
        
        # ======================================================================
        # PLOTTING: Generate anomaly visualizations
        # ======================================================================
        print("\nGenerating anomaly plots...")

        # Organize plots by model name
        plots_dir = os.path.join(config['results_dir'], 'anomaly_plots', best_model_name)
        ensure_dir(plots_dir)
        
        # Create DataFrame with anomaly flags for plotting
        # We need to align anomalies with the downsampled dataframe
        try:
            # Get the test portion of the downsampled data
            if use_separate_datasets:
                df_for_plot = prepared_test.get('df_downsampled', pd.DataFrame())
            else:
                df_for_plot = prepared.get('df_downsampled', pd.DataFrame())
                # Take the test portion
                n_total = len(df_for_plot)
                test_start = int(n_total * config.get('train_split', 0.7))
                df_for_plot = df_for_plot.iloc[test_start:].reset_index(drop=True)
            
            # Align with residuals (account for windowing)
            n_residuals = residuals_flat.shape[0]
            if len(df_for_plot) > n_residuals:
                df_for_plot = df_for_plot.iloc[-n_residuals:].reset_index(drop=True)
            elif len(df_for_plot) < n_residuals:
                # Pad or truncate residuals
                anomalies = anomalies[:len(df_for_plot)]
                residuals_flat = residuals_flat[:len(df_for_plot)]
            
            # Add anomaly columns
            for i, fname in enumerate(feature_names):
                if anomalies.ndim > 1:
                    df_for_plot[f'{fname}_anomaly'] = anomalies[:, i].astype(int)
                    df_for_plot[f'{fname}_warning'] = anomalies[:, i].astype(int)
                    df_for_plot[f'{fname}_danger'] = anomalies[:, i].astype(int)
                else:
                    df_for_plot[f'{fname}_anomaly'] = anomalies.astype(int)
                    df_for_plot[f'{fname}_warning'] = anomalies.astype(int)
                    df_for_plot[f'{fname}_danger'] = anomalies.astype(int)
            
            # Determine datetime column
            datetime_col = None
            for col in ['datetime', 'timestamp', 'time', 'date']:
                if col in df_for_plot.columns:
                    datetime_col = col
                    break
            
            if datetime_col is None and len(df_for_plot) > 0:
                # Create a simple index as x-axis
                df_for_plot['datetime'] = range(len(df_for_plot))
                datetime_col = 'datetime'
            
            # Find current column for startup detection
            current_col = None
            for col_name in ['current', 'Current', 'corrente', 'Corrente', 'I', 'i'] + [c for c in df_for_plot.columns if 'current' in c.lower()]:
                if col_name in df_for_plot.columns:
                    current_col = col_name
                    break
            
            if current_col:
                print(f"  Using '{current_col}' for startup detection")
            
            # Plot individual features
            for fname in feature_names:
                if fname in df_for_plot.columns:
                    try:
                        # Clean feature name for filename (replace invalid characters)
                        clean_fname = fname.replace("/", "_").replace("\\", "_").replace(" ", "_")
                        save_path = os.path.join(plots_dir, f'test_data_{clean_fname}.png')
                        plot_anomalies_with_colors(
                            df_for_plot,
                            feature=fname,
                            datetime_col=datetime_col,
                            current_col=current_col,
                            startup_quantile=0.05,
                            figsize=(14, 6),
                            save_path=save_path,
                            show_legend=True,
                            title=f'Anomaly Detection - {fname} (Test Data)'
                        )
                    except Exception as e:
                        print(f"  Warning: Could not plot {fname}: {e}")

            # Plot all features together
            try:
                all_features_path = os.path.join(plots_dir, 'test_data_all_features.png')
                plot_all_features_anomalies(
                    df_for_plot,
                    feature_names,
                    datetime_col=datetime_col,
                    current_col=current_col,
                    startup_quantile=0.05,
                    save_path=all_features_path,
                    title_prefix='Test Data - '
                )
            except Exception as e:
                print(f"  Warning: Could not create multi-feature plot: {e}")

            # Plot summary bar chart
            try:
                summary_plot_path = os.path.join(plots_dir, 'test_data_summary.png')
                plot_anomaly_summary(
                    df_for_plot,
                    feature_names,
                    current_col=current_col,
                    startup_quantile=0.05,
                    figsize=(12, 6),
                    save_path=summary_plot_path
                )
            except Exception as e:
                print(f"  Warning: Could not create summary plot: {e}")
            
            print(f"  Plots saved in: {plots_dir}")
            anom_results['plots_dir'] = plots_dir
            
        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")
        
        all_results['anomaly_detection'] = anom_results
    
    print_separator("PIPELINE COMPLETED")
    
    return {
        'models': all_results,
        'feature_names': feature_names,
        'config': config
    }


# =============================================================================
# Anomaly Detection Pipeline (Forecast-based)
# =============================================================================

def run_anomaly_pipeline(
    model_path: str,
    config: Dict,
    calib_data_path: Optional[str] = None,
    fatica_data_path: Optional[str] = None,
    apply_data_path: Optional[str] = None,
    thresholds_path: Optional[str] = None,
    mode: str = 'full',
    calib_weight: float = 1.0,
    fatica_weight: float = 1.0,
    anomaly_config: Optional[ForecastAnomalyConfig] = None
) -> Dict:
    """
    Run forecast-based anomaly detection pipeline.
    
    This pipeline:
    1. Calibrate: Learn thresholds from characterization data residuals
    2. Validate: Test thresholds on fatigue data (with appropriate weighting)
    3. Apply: Detect anomalies on new data (e.g., tampieri)
    
    The anomaly detection charts will show:
    - YELLOW: Anomalies during machine startup (low current)
    - ORANGE (CUSUM warning): Early signs of degradation (CUSUM > 75% of threshold h)
    - RED (danger): Confirmed anomaly/failure (sigma or CUSUM > threshold h)
    
    Args:
        model_path: Path to trained forecasting model
        config: Pipeline configuration
        calib_data_path: Path to characterization data for calibration
        fatica_data_path: Path to fatigue data for validation
        apply_data_path: Path to data for anomaly detection
        thresholds_path: Path to save/load thresholds JSON
        mode: 'calibrate', 'validate', 'apply', or 'full'
        calib_weight: Weight for characterization thresholds
        fatica_weight: Weight for fatigue thresholds
        anomaly_config: Configuration for anomaly detection
        
    Returns:
        Dictionary with results
    """
    print_separator("FORECAST-BASED ANOMALY DETECTION PIPELINE")
    
    if anomaly_config is None:
        anomaly_config = ForecastAnomalyConfig()
    
    results = {'mode': mode, 'config': anomaly_config.__dict__}
    
    # Load model
    print(f"\nLoading model: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Extract model name from path for organizing plots
    model_name = os.path.basename(model_path).split('_')[0]  # e.g., 'cnn' from 'cnn_W2880_F2880.keras'

    # Determine model type
    is_transformer = 'transformer' in model_path.lower()
    
    if is_transformer:
        model = torch.load(model_path)
        model.eval()
    else:
        from keras.models import load_model
        model = load_model(model_path)
    
    # Get configuration
    window_size = config.get('window_size', 2880)
    forecast_length = config.get('forecast_length', 2880)
    downsample_factor = config.get('downsample_factor', 15)
    exclude_current = config.get('exclude_current', True)
    
    # Helper function to prepare data and get residuals
    def get_residuals_from_data(data_path: str, data_name: str) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        """Load data, run forecast, compute residuals."""
        print(f"\nProcessing {data_name}: {data_path}")
        
        df = load_csv(data_path, verbose=True)
        
        data_result = prepare_data_for_training(
            df,
            window_size=window_size,
            forecast_length=forecast_length,
            downsample_factor=downsample_factor,
            exclude_current=exclude_current,
            test_size=0.0,  # Use all data
            verbose=True
        )
        
        X = data_result['X_train']
        y = data_result['y_train']
        feature_names = data_result['feature_names']
        df_down = data_result['df_downsampled']
        
        print(f"  Samples: {X.shape[0]:,}, Features: {len(feature_names)}")
        
        # Run prediction
        print(f"  Running forecast...")
        if is_transformer:
            device = next(model.parameters()).device
            X_tensor = torch.from_numpy(X).float().to(device)
            with torch.no_grad():
                y_pred = model(X_tensor).cpu().numpy()
        else:
            y_pred = model.predict(X, verbose=0)
        
        # Compute residuals
        residuals = compute_forecast_residuals(y, y_pred)
        print(f"  Residuals shape: {residuals.shape}")
        
        return residuals, feature_names, df_down
    
    # ==========================================================================
    # CALIBRATE: Learn thresholds from characterization data
    # ==========================================================================
    thresholds = None
    
    if mode in ['calibrate', 'full'] and calib_data_path:
        print_separator("CALIBRATION (Characterization Data)")
        
        residuals_calib, feature_names, df_calib = get_residuals_from_data(
            calib_data_path, "Characterization"
        )
        
        thresholds_calib = calibrate_thresholds_from_residuals(
            residuals_calib,
            feature_names,
            anomaly_config,
            verbose=True
        )
        
        thresholds_calib['source'] = 'characterization'
        thresholds_calib['data_path'] = calib_data_path
        thresholds_calib['n_samples'] = residuals_calib.shape[0] * residuals_calib.shape[1] if residuals_calib.ndim > 1 else len(residuals_calib)
        
        thresholds = thresholds_calib
        results['calibration'] = {
            'data': calib_data_path,
            'n_samples': int(thresholds_calib['n_samples']),
            'features': list(feature_names)
        }
    
    # ==========================================================================
    # VALIDATE: Test and combine with fatigue data
    # ==========================================================================
    if mode in ['validate', 'full'] and fatica_data_path:
        print_separator("VALIDATION (Fatigue Data)")
        
        residuals_fatica, feature_names_fatica, df_fatica = get_residuals_from_data(
            fatica_data_path, "Fatigue"
        )
        
        # If we have calibration thresholds, validate against them
        if thresholds is not None:
            validation = validate_thresholds_on_test(
                residuals_fatica,
                feature_names_fatica,
                thresholds,
                weight=fatica_weight,
                verbose=True
            )
            results['validation'] = validation
        
        # Also calibrate on fatigue data for weighted combination
        thresholds_fatica = calibrate_thresholds_from_residuals(
            residuals_fatica,
            feature_names_fatica,
            anomaly_config,
            verbose=True
        )
        
        thresholds_fatica['source'] = 'fatigue'
        thresholds_fatica['data_path'] = fatica_data_path
        thresholds_fatica['n_samples'] = residuals_fatica.shape[0] * residuals_fatica.shape[1] if residuals_fatica.ndim > 1 else len(residuals_fatica)
        
        # Combine thresholds with weighting
        if thresholds is not None and thresholds_calib is not None:
            print(f"\nCombining thresholds (calib_weight={calib_weight}, fatica_weight={fatica_weight})...")
            
            thresholds = combine_thresholds_weighted(
                [thresholds_calib, thresholds_fatica],
                [calib_weight, fatica_weight]
            )
            thresholds['source'] = 'combined'
            thresholds['weights'] = {'characterization': calib_weight, 'fatigue': fatica_weight}
            
            print("  Thresholds combined successfully")
        else:
            thresholds = thresholds_fatica
    
    # Load thresholds from file if not calibrated
    if thresholds is None and thresholds_path and os.path.exists(thresholds_path):
        print(f"\nLoading thresholds from: {thresholds_path}")
        thresholds = load_thresholds(thresholds_path)
        results['thresholds_loaded_from'] = thresholds_path
    
    # Save thresholds
    if thresholds is not None and thresholds_path:
        print(f"\nSaving thresholds to: {thresholds_path}")
        save_thresholds(thresholds, thresholds_path)
        results['thresholds_saved_to'] = thresholds_path
    
    # ==========================================================================
    # APPLY: Detect anomalies on new data
    # ==========================================================================
    if mode in ['apply', 'full'] and apply_data_path and thresholds is not None:
        print_separator("ANOMALY DETECTION (Application Data)")
        
        residuals_apply, feature_names_apply, df_apply = get_residuals_from_data(
            apply_data_path, "Application"
        )
        
        # Apply thresholds
        print("\nApplying thresholds...")
        anomaly_results = apply_thresholds_to_residuals(
            residuals_apply,
            feature_names_apply,
            thresholds,
            return_cusum=True
        )
        
        # Summary
        summary = anomaly_results['summary']
        print(f"\n  Total samples: {summary['total_samples']:,}")
        print(f"  Warning (yellow): {summary['warning_samples']:,} ({summary['warning_pct']:.2f}%)")
        print(f"  Danger (red):     {summary['danger_samples']:,} ({summary['danger_pct']:.2f}%)")
        
        # Per-feature breakdown
        print("\n  Per-feature breakdown:")
        for fname in feature_names_apply:
            if fname in anomaly_results.get('features', {}):
                feat = anomaly_results['features'][fname]
                n_warn = feat['warning'].sum()
                n_dang = feat['danger'].sum()
                print(f"    {fname}: warning={n_warn}, danger={n_dang}")
        
        # Create annotated DataFrame
        df_annotated = create_anomaly_dataframe(
            df_apply,
            anomaly_results,
            feature_names_apply,
            include_cusum=True
        )
        
        # Save results
        results_dir = config.get('results_dir', './results')
        ensure_dir(results_dir)
        
        # Save annotated data
        output_csv = os.path.join(results_dir, 'anomaly_detection_output.csv')
        df_annotated.to_csv(output_csv, index=False)
        print(f"\nAnnotated data saved: {output_csv}")
        
        # Save summary
        results['detection'] = {
            'data': apply_data_path,
            'summary': summary,
            'output_file': output_csv
        }
        
        # Save detection results as JSON
        detection_json = os.path.join(results_dir, 'anomaly_detection_summary.json')
        save_json(results, detection_json)
        print(f"Summary saved: {detection_json}")
        
        # ======================================================================
        # PLOTTING: Generate color-coded anomaly visualizations
        # ======================================================================
        print("\nGenerating anomaly plots...")

        # Organize plots by model name
        plots_dir = os.path.join(results_dir, 'anomaly_plots', model_name)
        ensure_dir(plots_dir)
        
        # Determine current column for startup detection
        # Try common names for current column
        current_col = None
        for col_name in ['current', 'Current', 'corrente', 'Corrente', 'I', 'i']:
            if col_name in df_annotated.columns:
                current_col = col_name
                break
        
        if current_col:
            print(f"  Using '{current_col}' for startup detection")
        else:
            print("  Warning: No current column found - startup detection disabled")
        
        # Plot individual features
        for fname in feature_names_apply:
            if fname in df_annotated.columns:
                try:
                    # Clean feature name for filename (replace invalid characters)
                    clean_fname = fname.replace("/", "_").replace("\\", "_").replace(" ", "_")
                    save_path = os.path.join(plots_dir, f'production_{clean_fname}.png')
                    plot_anomalies_with_colors(
                        df_annotated,
                        feature=fname,
                        datetime_col='datetime',
                        current_col=current_col,
                        startup_quantile=0.05,
                        figsize=(14, 6),
                        save_path=save_path,
                        show_legend=True,
                        title=f'Anomaly Detection - {fname} (Production Data)'
                    )
                except Exception as e:
                    print(f"  Warning: Could not plot {fname}: {e}")

        # Plot all features together
        try:
            all_features_path = os.path.join(plots_dir, 'production_all_features.png')
            plot_all_features_anomalies(
                df_annotated,
                feature_names_apply,
                datetime_col='datetime',
                current_col=current_col,
                startup_quantile=0.05,
                save_path=all_features_path,
                title_prefix='Production Data - '
            )
        except Exception as e:
            print(f"  Warning: Could not create multi-feature plot: {e}")

        # Plot summary bar chart
        try:
            summary_path = os.path.join(plots_dir, 'production_summary.png')
            plot_anomaly_summary(
                df_annotated,
                feature_names_apply,
                current_col=current_col,
                startup_quantile=0.05,
                figsize=(12, 6),
                save_path=summary_path
            )
        except Exception as e:
            print(f"  Warning: Could not create summary plot: {e}")
        
        print(f"  Plots saved in: {plots_dir}")
        results['detection']['plots_dir'] = plots_dir
    
    print_separator("ANOMALY PIPELINE COMPLETED")
    
    return results


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Training and Forecasting Pipeline for Anomaly Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        EXAMPLES
        ========
        Training:
            python main.py --data ./data/dataset.csv --models cnn lstm
            python main.py --data ./data/dataset.csv --models cnn --optimize optuna --n-trials 30
            python main.py --data ./data/dataset.csv --models cnn --optimize grid --max-configs 10

        Anomaly Detection:
            python main.py --anomaly-mode full --models cnn
            python main.py --anomaly-mode full --models cnn --sigma-warning 2.0 --sigma-danger 3.0
        """
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Path to the CSV file with data'
    )
    
    parser.add_argument(
        '--models', '-m',
        nargs='+',
        type=str,
        help='List of models to train/use'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all available models'
    )
    
    parser.add_argument(
        '--list-models', '-l',
        action='store_true',
        help='Show available models and exit'
    )
    
    parser.add_argument(
        '--force-retrain', '-f',
        action='store_true',
        help='Force retraining even if pretrained models exist'
    )
    
    parser.add_argument(
        '--no-anomaly',
        action='store_true',
        help='Skip anomaly detection phase'
    )
    
    parser.add_argument(
        '--window-size', '-w',
        type=int,
        default=DEFAULT_CONFIG['window_size'],
        help=f"Input window size in samples. Default {DEFAULT_CONFIG['window_size']} = 24h at 30s intervals"
    )
    
    parser.add_argument(
        '--forecast-length', '-F',
        type=int,
        default=DEFAULT_CONFIG['forecast_length'],
        help=f"Forecast length in samples. Default {DEFAULT_CONFIG['forecast_length']} = 24h at 30s intervals"
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=DEFAULT_CONFIG['epochs'],
        help=f"Maximum number of epochs (default: {DEFAULT_CONFIG['epochs']})"
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=DEFAULT_CONFIG['batch_size'],
        help=f"Batch size (default: {DEFAULT_CONFIG['batch_size']})"
    )
    
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=DEFAULT_CONFIG['learning_rate'],
        help=f"Learning rate (default: {DEFAULT_CONFIG['learning_rate']})"
    )
    
    parser.add_argument(
        '--dropout-rate',
        type=float,
        default=DEFAULT_CONFIG['dropout_rate'],
        help=f"Dropout rate (default: {DEFAULT_CONFIG['dropout_rate']})"
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=DEFAULT_CONFIG['patience'],
        help=f"Early stopping patience (default: {DEFAULT_CONFIG['patience']})"
    )
    
    # Data filtering arguments
    parser.add_argument(
        '--include-current',
        action='store_true',
        help='Include current column in training features (default: excluded)'
    )
    
    # Hyperparameter optimization arguments
    parser.add_argument(
        '--optimize',
        type=str,
        choices=['optuna', 'grid', 'none'],
        default='none',
        help='Hyperparameter optimization method: optuna (Bayesian), grid (grid search), none'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=20,
        help='Number of Optuna trials (default: 20)'
    )
    
    parser.add_argument(
        '--max-configs',
        type=int,
        default=10,
        help='Max configurations for grid search (default: 10)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Optimization timeout in seconds (optional)'
    )
    
    parser.add_argument(
        '--downsample',
        type=int,
        default=DEFAULT_CONFIG['downsample_factor'],
        help=f"Downsampling factor. Default {DEFAULT_CONFIG['downsample_factor']} converts 2sâ†’30s sampling"
    )
    
    # Dataset arguments
    parser.add_argument(
        '--train-data',
        type=str,
        default=None,
        help='Path to training dataset (if separate from --data)'
    )
    
    parser.add_argument(
        '--val-data',
        type=str,
        default=None,
        help='Path to validation dataset (if separate from --data)'
    )
    
    parser.add_argument(
        '--test-data',
        type=str,
        default=None,
        help='Path to test dataset (if separate from --data)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        help='Train/val/test split ratios (e.g., "0.7/0.15/0.15" or "70/15/15"). '
             'Used when --data is provided without separate datasets.'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default=DEFAULT_CONFIG['models_dir'],
        help=f"Directory for pretrained models (default: {DEFAULT_CONFIG['models_dir']})"
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default=DEFAULT_CONFIG['results_dir'],
        help=f"Directory for results (default: {DEFAULT_CONFIG['results_dir']})"
    )
    
    # ==========================================================================
    # Anomaly Detection Pipeline Arguments
    # ==========================================================================
    parser.add_argument(
        '--anomaly-mode',
        type=str,
        choices=['calibrate', 'validate', 'apply', 'full'],
        default=None,
        help='Anomaly detection mode: calibrate (learn thresholds), validate (test thresholds), '
             'apply (detect anomalies), full (calibrate+validate+apply)'
    )
    
    parser.add_argument(
        '--calib-data',
        type=str,
        default=DEFAULT_DATASETS['caratterizzazione'],
        help=f"Path to LAB characterization data (normal operation) for threshold calibration. "
             f"(default: {DEFAULT_DATASETS['caratterizzazione']})"
    )
    
    parser.add_argument(
        '--fatica-data',
        type=str,
        default=DEFAULT_DATASETS['fatica'],
        help=f"Path to LAB fatigue/stress test data for threshold validation. "
             f"(default: {DEFAULT_DATASETS['fatica']})"
    )
    
    parser.add_argument(
        '--apply-data',
        type=str,
        default=DEFAULT_DATASETS['real_data'],
        help=f"Path to REAL MACHINE data (Tampieri) for anomaly detection in production. "
             f"(default: {DEFAULT_DATASETS['real_data']})"
    )
    
    parser.add_argument(
        '--thresholds-file',
        type=str,
        default=DEFAULT_CONFIG['thresholds_file'],
        help=f"Path to thresholds JSON file for saving/loading "
             f"(default: {DEFAULT_CONFIG['thresholds_file']})"
    )
    
    parser.add_argument(
        '--calib-weight',
        type=float,
        default=1.0,
        help='Weight for characterization thresholds in combination (default: 1.0)'
    )
    
    parser.add_argument(
        '--fatica-weight',
        type=float,
        default=1.0,
        help='Weight for fatigue thresholds in combination (default: 1.0)'
    )
    
    parser.add_argument(
        '--sigma-warning',
        type=float,
        default=2.0,
        help='Sigma threshold for warnings (yellow) (default: 2.0)'
    )
    
    parser.add_argument(
        '--sigma-danger',
        type=float,
        default=3.0,
        help='Sigma threshold for danger (red) (default: 3.0)'
    )
    
    parser.add_argument(
        '--cusum-strategy',
        type=str,
        choices=['min_baseline_alarms', 'balanced', 'max_sensitivity'],
        default='min_baseline_alarms',
        help='Strategy for selecting best CUSUM parameters: '
             'min_baseline_alarms (most conservative), '
             'balanced (balance baseline/test), '
             'max_sensitivity (most sensitive). Default: min_baseline_alarms'
    )

    return parser.parse_args()

def cleanup_tf():
    # CLEANUP FINALE (FONDAMENTALE SU SLURM)
    import tensorflow as tf
    import gc
    tf.keras.backend.clear_session()
    gc.collect()

def main():
    """Main entry point."""
    args = parse_args()
    
    # List models
    if args.list_models:
        print("\nAvailable models:")
        print("-" * 40)
        print("\n  Neural Network models (Keras/PyTorch):")
        for model in get_available_models():
            print(f"    - {model}")
        print("\n  Traditional ML models:")
        print(f"    - boosting (HistGradientBoosting)")
        if not is_tcn_available():
            print("\nNote: TCN not available (install with: pip install keras-tcn)")
        return
    
    # ==========================================================================
    # Check if running anomaly pipeline mode
    # ==========================================================================
    if args.anomaly_mode:
        # Anomaly detection pipeline mode
        print("\n" + "="*60)
        print("  ANOMALY DETECTION PIPELINE MODE")
        print("="*60)
        
        # Show default datasets being used
        print(f"\nDefault datasets:")
        print(f"  LAB - Normal (caratterizzazione): {DEFAULT_DATASETS['caratterizzazione']}")
        print(f"  LAB - Stress (fatica):            {DEFAULT_DATASETS['fatica']}")
        print(f"  REAL MACHINE (tampieri):          {DEFAULT_DATASETS['real_data']}")
        print(f"  Thresholds file:                 {DEFAULT_CONFIG['thresholds_file']}")
        
        # Validate arguments for anomaly mode - check files exist
        if args.anomaly_mode in ['calibrate', 'full']:
            if not args.calib_data or not os.path.exists(args.calib_data):
                print(f"ERROR: Calibration data not found: {args.calib_data}")
                print("       Use --calib-data to specify a different path")
                sys.exit(1)
            print(f"\n  Using calibration data: {args.calib_data}")
        
        if args.anomaly_mode in ['validate', 'full']:
            if args.fatica_data and not os.path.exists(args.fatica_data):
                print(f"ERROR: Fatigue data not found: {args.fatica_data}")
                print("       Use --fatica-data to specify a different path")
                sys.exit(1)
            if args.fatica_data:
                print(f"  Using validation data:  {args.fatica_data}")
        
        if args.anomaly_mode in ['apply', 'full']:
            if not args.apply_data or not os.path.exists(args.apply_data):
                print(f"ERROR: Application data not found: {args.apply_data}")
                print("       Use --apply-data to specify a different path")
                sys.exit(1)
            print(f"  Using application data: {args.apply_data}")
        
        # Find model to use
        model_path = None
        model_name = args.models[0] if args.models and len(args.models) == 1 else 'cnn'
        
        # Use find_best_model to search for compatible pretrained models
        model_path = find_best_model(
            models_dir=args.models_dir,
            model_name=model_name,
            window_size=args.window_size,
            forecast_length=args.forecast_length
        )
        
        if not model_path:
            # Fallback: list any available models
            keras_models = list(Path(args.models_dir).glob('*.keras'))
            h5_models = list(Path(args.models_dir).glob('*.h5'))
            pt_models = list(Path(args.models_dir).glob('*.pt'))
            all_models = keras_models + h5_models + pt_models
            
            if all_models:
                model_path = str(all_models[0])  # Use first available
                print(f"Using fallback model: {model_path}")
            else:
                print(f"ERROR: No pretrained models found in {args.models_dir}")
                print("       Train a model first or specify model with --models")
                sys.exit(1)
        else:
            print(f"Using pretrained model: {model_path}")
        
        # Configure
        config = DEFAULT_CONFIG.copy()
        config.update({
            'window_size': args.window_size,
            'forecast_length': args.forecast_length,
            'downsample_factor': args.downsample,
            'models_dir': args.models_dir,
            'results_dir': args.results_dir,
            'exclude_current': not args.include_current,
        })
        
        # Anomaly config
        anomaly_config = ForecastAnomalyConfig(
            sigma_warning=args.sigma_warning,
            sigma_danger=args.sigma_danger,
        )
        
        # Run anomaly pipeline
        results = run_anomaly_pipeline(
            model_path=model_path,
            config=config,
            calib_data_path=args.calib_data,
            fatica_data_path=args.fatica_data,
            apply_data_path=args.apply_data,
            thresholds_path=args.thresholds_file,
            mode=args.anomaly_mode,
            calib_weight=args.calib_weight,
            fatica_weight=args.fatica_weight,
            anomaly_config=anomaly_config
        )
        
        return results
    
    # ==========================================================================
    # Standard training pipeline
    # ==========================================================================
    
    # Verify arguments
    if not args.data:
        print("ERROR: Specify data path with --data")
        print("Use --help to see available options")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        print(f"ERROR: File not found: {args.data}")
        sys.exit(1)
    
    # Determine models to use
    if args.all:
        models_to_run = get_all_available_models()
    elif args.models:
        models_to_run = args.models
    else:
        print("ERROR: Specify models with --models or use --all")
        print(f"Available models: {get_all_available_models()}")
        sys.exit(1)
    
    # Configure
    config = DEFAULT_CONFIG.copy()
    config.update({
        'window_size': args.window_size,
        'forecast_length': args.forecast_length,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'dropout_rate': args.dropout_rate,
        'patience': args.patience,
        'downsample_factor': args.downsample,
        'models_dir': args.models_dir,
        'results_dir': args.results_dir,
        'optimize': args.optimize,
        'n_trials': args.n_trials,
        'max_configs': args.max_configs,
        'timeout': args.timeout,
        'exclude_current': not args.include_current,
    })
    
    # Parse split ratios if provided
    if args.split:
        try:
            parts = args.split.replace('/', ' ').replace(',', ' ').split()
            splits = [float(p) for p in parts]
            if len(splits) != 3:
                raise ValueError("Split must have 3 values")
            # Normalize if percentages (sum > 1)
            if sum(splits) > 1:
                splits = [s / 100 for s in splits]
            if abs(sum(splits) - 1.0) > 0.01:
                raise ValueError(f"Split ratios must sum to 1.0, got {sum(splits)}")
            config['train_split'] = splits[0]
            config['val_split'] = splits[1]
            config['test_split'] = splits[2]
        except Exception as e:
            print(f"ERROR: Invalid split format '{args.split}': {e}")
            print("       Use format like '0.7/0.1/0.2' or '70/20/10'")
            sys.exit(1)
    
    # Dataset configuration
    config['train_data'] = args.train_data
    config['val_data'] = args.val_data
    config['test_data'] = args.test_data
    
    # Check Optuna availability if optimization requested
    if args.optimize == 'optuna' and not OPTUNA_AVAILABLE:
        print("WARNING: Optuna not available. Install with: pip install optuna")
        print("         Proceeding without optimization...")
        config['optimize'] = 'none'
    
    # Run pipeline
    results = run_pipeline(
        data_path=args.data,
        models_to_run=models_to_run,
        config=config,
        force_retrain=args.force_retrain,
        run_anomaly_detection=not args.no_anomaly
    )
    cleanup_tf()
    return results


if __name__ == "__main__":
    main()