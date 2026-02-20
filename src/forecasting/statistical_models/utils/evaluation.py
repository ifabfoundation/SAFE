"""
Evaluation metrics for time series forecasting.

Provides:
- RMSE, MAE, R2, MAPE, MASE computation
- Aggregate metrics (averaged over forecast horizon)
- Per-horizon metrics (at specific timesteps: h=1, 120, 720, 1440, 2880)
- Per-feature metrics (separate evaluation for each sensor)
- JSON serialization helpers
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_to_native_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object to convert (dict, list, numpy type, etc.)
        
    Returns:
        Object with all numpy types converted to native Python types
    """
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute Mean Absolute Percentage Error.

    Handles zeros by adding small epsilon to denominator.
    Alternative: Use sMAPE for better handling of zeros.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE value (as percentage, 0-100)
    """
    # Skip samples where true value is exactly zero
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        logger.warning("All true values near zero, MAPE undefined. Returning NaN.")
        return np.nan

    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


def compute_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """
    Compute Mean Absolute Scaled Error.

    Uses lag-1 persistence baseline from training data:
    MAE(model) / MAE(naive lag-1 on training data)

    Args:
        y_true: Ground truth values for test set
        y_pred: Model predictions for test set
        y_train: Training data for computing naive baseline

    Returns:
        MASE value (< 1 means better than naive, > 1 means worse)
    """
    # Model MAE
    mae_model = np.mean(np.abs(y_true - y_pred))

    # Naive baseline MAE (lag-1 persistence on training data)
    naive_errors = np.abs(y_train[1:] - y_train[:-1])
    mae_naive = np.mean(naive_errors)

    if mae_naive == 0:
        logger.warning("Naive MAE is zero, MASE undefined. Returning NaN.")
        return np.nan

    mase = mae_model / mae_naive
    return mase


def compute_metrics(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   y_train: np.ndarray = None) -> Dict[str, float]:
    """
    Compute all metrics for given predictions.

    Args:
        y_true: Ground truth values (any shape)
        y_pred: Predicted values (same shape as y_true)
        y_train: Training data for MASE computation (optional)

    Returns:
        Dictionary with RMSE, MAE, R2, MAPE, MASE (if y_train provided)
    """
    # Flatten arrays for metric computation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)),
        'mae': mean_absolute_error(y_true_flat, y_pred_flat),
        'r2': r2_score(y_true_flat, y_pred_flat),
        'mape': compute_mape(y_true_flat, y_pred_flat)
    }

    # Add MASE if training data provided
    if y_train is not None:
        y_train_flat = y_train.flatten()
        metrics['mase'] = compute_mase(y_true_flat, y_pred_flat, y_train_flat)

    return metrics


def compute_aggregate_metrics(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_train: np.ndarray = None) -> Dict[str, float]:
    """
    Compute aggregate metrics averaged over entire forecast horizon.

    For multi-sample predictions, computes metrics across all samples and timesteps.

    Args:
        y_true: Ground truth [n_samples, horizon] or [n_samples, horizon, n_features]
        y_pred: Predictions (same shape as y_true)
        y_train: Training data for MASE (same feature structure)

    Returns:
        Dictionary with aggregate RMSE, MAE, R2, MAPE, MASE
    """
    logger.info(f"Computing aggregate metrics for shape {y_true.shape}")
    return compute_metrics(y_true, y_pred, y_train)


def compute_per_horizon_metrics(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_train: np.ndarray = None,
                               horizons: List[int] = [1, 120, 720, 1440, 2880]) -> Dict[int, Dict[str, float]]:
    """
    Compute metrics at specific forecast horizons.

    Evaluates accuracy degradation over time by measuring performance
    at h=1 (30s), h=120 (1h), h=720 (6h), h=1440 (12h), h=2880 (24h).

    Args:
        y_true: Ground truth [n_samples, horizon] or [n_samples, horizon, n_features]
        y_pred: Predictions (same shape as y_true)
        y_train: Training data for MASE
        horizons: List of horizon indices to evaluate (1-indexed, will subtract 1 for 0-indexing)

    Returns:
        Dictionary mapping horizon -> metrics dict
    """
    logger.info(f"Computing per-horizon metrics for horizons {horizons}")

    if y_true.ndim == 2:
        # Shape: [n_samples, horizon]
        max_horizon = y_true.shape[1]
    elif y_true.ndim == 3:
        # Shape: [n_samples, horizon, n_features]
        max_horizon = y_true.shape[1]
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {y_true.shape}")

    per_horizon_metrics = {}

    for h in horizons:
        if h > max_horizon:
            logger.warning(f"Horizon {h} exceeds max horizon {max_horizon}, skipping")
            continue

        # Extract predictions at this horizon (convert to 0-indexed)
        h_idx = h - 1
        if y_true.ndim == 2:
            y_true_h = y_true[:, h_idx]
            y_pred_h = y_pred[:, h_idx]
        else:  # 3D
            y_true_h = y_true[:, h_idx, :]
            y_pred_h = y_pred[:, h_idx, :]

        # Compute metrics for this horizon
        metrics_h = compute_metrics(y_true_h, y_pred_h, y_train)
        per_horizon_metrics[h] = metrics_h

        logger.info(f"  h={h}: RMSE={metrics_h['rmse']:.4f}, MAE={metrics_h['mae']:.4f}, R2={metrics_h['r2']:.4f}")

    return per_horizon_metrics


def compute_per_feature_metrics(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_train: np.ndarray = None,
                               feature_names: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for each feature.

    Useful for multivariate forecasting to see which sensors are easier/harder to predict.

    Args:
        y_true: Ground truth [n_samples, horizon, n_features]
        y_pred: Predictions (same shape)
        y_train: Training data [n_train_samples, n_features] or [n_train_samples, horizon, n_features]
        feature_names: Names of features (default: ['feature_0', 'feature_1', ...])

    Returns:
        Dictionary mapping feature_name -> metrics dict
    """
    if y_true.ndim != 3:
        raise ValueError(f"Expected 3D array [n_samples, horizon, n_features], got shape {y_true.shape}")

    n_features = y_true.shape[2]

    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]

    if len(feature_names) != n_features:
        raise ValueError(f"Expected {n_features} feature names, got {len(feature_names)}")

    logger.info(f"Computing per-feature metrics for {n_features} features")

    per_feature_metrics = {}

    for i, feature_name in enumerate(feature_names):
        # Extract this feature across all samples and timesteps
        y_true_feat = y_true[:, :, i]  # [n_samples, horizon]
        y_pred_feat = y_pred[:, :, i]

        # Extract training data for this feature
        y_train_feat = None
        if y_train is not None:
            if y_train.ndim == 2:
                # Training data is [n_samples, n_features]
                y_train_feat = y_train[:, i]
            elif y_train.ndim == 3:
                # Training data is [n_samples, horizon, n_features]
                y_train_feat = y_train[:, :, i]
            else:
                raise ValueError(f"Expected y_train to be 2D or 3D, got shape {y_train.shape}")

        # Compute metrics for this feature
        metrics_feat = compute_metrics(y_true_feat, y_pred_feat, y_train_feat)
        per_feature_metrics[feature_name] = metrics_feat

        logger.info(f"  {feature_name}: RMSE={metrics_feat['rmse']:.4f}, MAE={metrics_feat['mae']:.4f}, R2={metrics_feat['r2']:.4f}")

    return per_feature_metrics


def evaluate_forecast(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     y_train: np.ndarray = None,
                     feature_names: List[str] = None,
                     horizons: List[int] = [1, 120, 720, 1440, 2880]) -> Dict:
    """
    Complete evaluation with aggregate, per-horizon, and per-feature metrics.

    Args:
        y_true: Ground truth [n_samples, horizon, n_features]
        y_pred: Predictions (same shape)
        y_train: Training data for MASE baseline
        feature_names: Names of features
        horizons: Specific horizons to evaluate

    Returns:
        Dictionary with:
        {
            'aggregate': {...},
            'per_horizon': {h: {...}, ...},
            'per_feature': {feat: {...}, ...}
        }
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating forecast with shape {y_true.shape}")
    logger.info(f"{'='*60}")

    results = {}

    # Aggregate metrics
    logger.info("\n--- Aggregate Metrics ---")
    results['aggregate'] = compute_aggregate_metrics(y_true, y_pred, y_train)

    # Per-horizon metrics
    logger.info("\n--- Per-Horizon Metrics ---")
    results['per_horizon'] = compute_per_horizon_metrics(y_true, y_pred, y_train, horizons)

    # Per-feature metrics (if multivariate)
    if y_true.ndim == 3 and y_true.shape[2] > 1:
        logger.info("\n--- Per-Feature Metrics ---")
        results['per_feature'] = compute_per_feature_metrics(y_true, y_pred, y_train, feature_names)

    logger.info(f"\n{'='*60}\n")

    return results


if __name__ == "__main__":
    # Test evaluation functions
    logger.info("Testing evaluation metrics...")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    horizon = 2880
    n_features = 4

    # Generate test predictions
    y_true = np.random.randn(n_samples, horizon, n_features)
    y_pred = y_true + np.random.randn(n_samples, horizon, n_features) * 0.1  # Add small noise

    # Generate training data
    y_train = np.random.randn(1000, n_features)

    feature_names = ['acc_p3', 'acc_p4', 'temp_p3', 'temp_p4']

    # Test complete evaluation
    results = evaluate_forecast(
        y_true=y_true,
        y_pred=y_pred,
        y_train=y_train,
        feature_names=feature_names,
        horizons=[1, 120, 720, 1440, 2880]
    )

    # Print results
    logger.info("\nResults:")
    logger.info(f"Aggregate RMSE: {results['aggregate']['rmse']:.4f}")
    logger.info(f"Aggregate MAE: {results['aggregate']['mae']:.4f}")
    logger.info(f"Aggregate R2: {results['aggregate']['r2']:.4f}")
    logger.info(f"Aggregate MAPE: {results['aggregate']['mape']:.4f}%")
    logger.info(f"Aggregate MASE: {results['aggregate']['mase']:.4f}")

    logger.info("\nPer-horizon RMSE:")
    for h, metrics in results['per_horizon'].items():
        logger.info(f"  h={h}: {metrics['rmse']:.4f}")

    logger.info("\nPer-feature RMSE:")
    for feat, metrics in results['per_feature'].items():
        logger.info(f"  {feat}: {metrics['rmse']:.4f}")
