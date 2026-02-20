"""
Visualization utilities for time series forecasting results.

Provides:
- Forecast vs actual plots
- Residual analysis plots
- Per-horizon metric plots
- Per-feature metric comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10


def plot_forecast_single_sample(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                feature_names: List[str],
                                sample_idx: int = 0,
                                save_path: Optional[str] = None):
    """
    Plot forecast vs actual for a single sample across all features.

    Args:
        y_true: Ground truth [n_samples, horizon, n_features]
        y_pred: Predictions (same shape)
        feature_names: Names of features
        sample_idx: Which sample to plot
        save_path: Path to save figure (optional)
    """
    n_features = len(feature_names)
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 3 * n_features))

    if n_features == 1:
        axes = [axes]

    timesteps = np.arange(y_true.shape[1])

    for i, (ax, feat_name) in enumerate(zip(axes, feature_names)):
        ax.plot(timesteps, y_true[sample_idx, :, i], label='Actual', linewidth=2, alpha=0.7)
        ax.plot(timesteps, y_pred[sample_idx, :, i], label='Forecast', linewidth=2, alpha=0.7)
        ax.set_xlabel('Timestep (30s intervals)')
        ax.set_ylabel('Normalized Value')
        ax.set_title(f'{feat_name} - Sample {sample_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    plt.close()


def plot_forecast_multiple_samples(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   feature_names: List[str],
                                   n_samples: int = 5,
                                   save_path: Optional[str] = None):
    """
    Plot forecasts for multiple random samples.

    Args:
        y_true: Ground truth [n_samples, horizon, n_features]
        y_pred: Predictions (same shape)
        feature_names: Names of features
        n_samples: Number of random samples to plot
        save_path: Path to save figure (optional)
    """
    n_total_samples = y_true.shape[0]
    sample_indices = np.random.choice(n_total_samples, size=min(n_samples, n_total_samples), replace=False)

    for sample_idx in sample_indices:
        save_path_i = None
        if save_path:
            # Append sample index to filename
            base, ext = save_path.rsplit('.', 1)
            save_path_i = f"{base}_sample{sample_idx}.{ext}"

        plot_forecast_single_sample(y_true, y_pred, feature_names, sample_idx, save_path_i)


def plot_residuals(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  feature_names: List[str],
                  save_path: Optional[str] = None):
    """
    Plot residual distributions for each feature.

    Args:
        y_true: Ground truth [n_samples, horizon, n_features]
        y_pred: Predictions (same shape)
        feature_names: Names of features
        save_path: Path to save figure (optional)
    """
    n_features = len(feature_names)
    fig, axes = plt.subplots(2, n_features, figsize=(4 * n_features, 8))

    if n_features == 1:
        axes = axes.reshape(2, 1)

    for i, feat_name in enumerate(feature_names):
        residuals = y_true[:, :, i].flatten() - y_pred[:, :, i].flatten()

        # Histogram
        axes[0, i].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[0, i].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, i].set_xlabel('Residual')
        axes[0, i].set_ylabel('Frequency')
        axes[0, i].set_title(f'{feat_name} - Residual Distribution')
        axes[0, i].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, i])
        axes[1, i].set_title(f'{feat_name} - Q-Q Plot')
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    plt.close()


def plot_per_horizon_metrics(per_horizon_metrics: Dict[int, Dict[str, float]],
                            metric_names: List[str] = ['rmse', 'mae', 'r2'],
                            save_path: Optional[str] = None):
    """
    Plot how metrics evolve across forecast horizon.

    Args:
        per_horizon_metrics: Dict mapping horizon -> metrics dict
        metric_names: Which metrics to plot
        save_path: Path to save figure (optional)
    """
    horizons = sorted(per_horizon_metrics.keys())
    n_metrics = len(metric_names)

    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    for i, metric_name in enumerate(metric_names):
        metric_values = [per_horizon_metrics[h][metric_name] for h in horizons]

        axes[i].plot(horizons, metric_values, marker='o', linewidth=2, markersize=8)
        axes[i].set_xlabel('Forecast Horizon (timesteps)')
        axes[i].set_ylabel(metric_name.upper())
        axes[i].set_title(f'{metric_name.upper()} vs Forecast Horizon')
        axes[i].grid(True, alpha=0.3)

        # Add horizon labels
        axes[i].set_xticks(horizons)
        axes[i].set_xticklabels([f'{h}\n({h*30}s)' for h in horizons])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    plt.close()


def plot_per_feature_metrics(per_feature_metrics: Dict[str, Dict[str, float]],
                            metric_names: List[str] = ['rmse', 'mae', 'r2'],
                            save_path: Optional[str] = None):
    """
    Compare metrics across different features.

    Args:
        per_feature_metrics: Dict mapping feature_name -> metrics dict
        metric_names: Which metrics to plot
        save_path: Path to save figure (optional)
    """
    feature_names = list(per_feature_metrics.keys())
    n_metrics = len(metric_names)

    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    for i, metric_name in enumerate(metric_names):
        metric_values = [per_feature_metrics[feat][metric_name] for feat in feature_names]

        axes[i].bar(feature_names, metric_values, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel('Feature')
        axes[i].set_ylabel(metric_name.upper())
        axes[i].set_title(f'{metric_name.upper()} by Feature')
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    plt.close()


def plot_model_comparison(results_dict: Dict[str, Dict],
                         metric_name: str = 'rmse',
                         save_path: Optional[str] = None):
    """
    Compare multiple models side-by-side.

    Args:
        results_dict: Dict mapping model_name -> evaluation results
        metric_name: Which metric to compare
        save_path: Path to save figure (optional)
    """
    model_names = list(results_dict.keys())
    metric_values = [results_dict[model]['aggregate'][metric_name] for model in model_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(model_names, metric_values, alpha=0.7, edgecolor='black')

    # Color bars by performance (lower is better for most metrics)
    if metric_name.lower() != 'r2':
        # For error metrics, lower is better
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(bars)))
        sorted_indices = np.argsort(metric_values)
        for i, idx in enumerate(sorted_indices):
            bars[idx].set_color(colors[i])
    else:
        # For R2, higher is better
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(bars)))
        sorted_indices = np.argsort(metric_values)[::-1]
        for i, idx in enumerate(sorted_indices):
            bars[idx].set_color(colors[i])

    ax.set_xlabel('Model')
    ax.set_ylabel(metric_name.upper())
    ax.set_title(f'Model Comparison - {metric_name.upper()}')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    plt.close()


def create_evaluation_report(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            evaluation_results: Dict,
                            feature_names: List[str],
                            model_name: str,
                            save_dir: str):
    """
    Create a complete visualization report for a model.

    Generates:
    - Forecast plots for random samples
    - Residual analysis
    - Per-horizon metrics
    - Per-feature metrics

    Args:
        y_true: Ground truth
        y_pred: Predictions
        evaluation_results: Results from evaluate_forecast()
        feature_names: Names of features
        model_name: Name of model for plot titles
        save_dir: Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Creating evaluation report for {model_name} in {save_dir}")

    # Forecast plots
    plot_forecast_multiple_samples(
        y_true, y_pred, feature_names, n_samples=3,
        save_path=f"{save_dir}/{model_name}_forecast.png"
    )

    # Residuals
    plot_residuals(
        y_true, y_pred, feature_names,
        save_path=f"{save_dir}/{model_name}_residuals.png"
    )

    # Per-horizon metrics
    if 'per_horizon' in evaluation_results:
        plot_per_horizon_metrics(
            evaluation_results['per_horizon'],
            metric_names=['rmse', 'mae', 'r2'],
            save_path=f"{save_dir}/{model_name}_per_horizon.png"
        )

    # Per-feature metrics
    if 'per_feature' in evaluation_results:
        plot_per_feature_metrics(
            evaluation_results['per_feature'],
            metric_names=['rmse', 'mae', 'r2'],
            save_path=f"{save_dir}/{model_name}_per_feature.png"
        )

    logger.info(f"Evaluation report complete for {model_name}")


if __name__ == "__main__":
    # Test visualization functions
    logger.info("Testing visualization utilities...")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 50
    horizon = 2880
    n_features = 4

    y_true = np.random.randn(n_samples, horizon, n_features)
    y_pred = y_true + np.random.randn(n_samples, horizon, n_features) * 0.1

    feature_names = ['acc_p3', 'acc_p4', 'temp_p3', 'temp_p4']

    # Test single sample plot
    logger.info("Testing single sample plot...")
    plot_forecast_single_sample(y_true, y_pred, feature_names, sample_idx=0)

    # Test residuals
    logger.info("Testing residuals plot...")
    plot_residuals(y_true, y_pred, feature_names)

    # Test per-horizon metrics
    logger.info("Testing per-horizon metrics plot...")
    per_horizon_metrics = {
        1: {'rmse': 0.10, 'mae': 0.08, 'r2': 0.95},
        120: {'rmse': 0.15, 'mae': 0.12, 'r2': 0.90},
        720: {'rmse': 0.25, 'mae': 0.20, 'r2': 0.80},
        1440: {'rmse': 0.35, 'mae': 0.28, 'r2': 0.70},
        2880: {'rmse': 0.45, 'mae': 0.36, 'r2': 0.60}
    }
    plot_per_horizon_metrics(per_horizon_metrics)

    # Test per-feature metrics
    logger.info("Testing per-feature metrics plot...")
    per_feature_metrics = {
        'acc_p3': {'rmse': 0.20, 'mae': 0.15, 'r2': 0.85},
        'acc_p4': {'rmse': 0.22, 'mae': 0.17, 'r2': 0.83},
        'temp_p3': {'rmse': 0.18, 'mae': 0.14, 'r2': 0.87},
        'temp_p4': {'rmse': 0.19, 'mae': 0.15, 'r2': 0.86}
    }
    plot_per_feature_metrics(per_feature_metrics)

    # Test model comparison
    logger.info("Testing model comparison plot...")
    results_dict = {
        'Prophet': {'aggregate': {'rmse': 0.45}},
        'SARIMA': {'aggregate': {'rmse': 0.42}},
        'VAR': {'aggregate': {'rmse': 0.38}},
        'RandomForest': {'aggregate': {'rmse': 0.35}},
        'XGBoost': {'aggregate': {'rmse': 0.32}},
        'LightGBM': {'aggregate': {'rmse': 0.30}}
    }
    plot_model_comparison(results_dict, metric_name='rmse')

    logger.info("All visualization tests completed!")
