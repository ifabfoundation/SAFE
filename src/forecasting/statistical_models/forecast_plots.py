"""
Forecast visualization utilities.

Provides functions to plot forecasted vs actual time series for each feature.
Can be called after model training to visualize prediction quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_forecast_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    sample_idx: int = 0,
    model_name: str = "Model",
    scenario: str = "A",
    save_dir: str = "plots",
    timestep_seconds: int = 30,
    show_plot: bool = False
) -> str:
    """
    Plot forecasted vs actual time series for each feature.
    
    Args:
        y_true: Ground truth [n_samples, horizon, n_features]
        y_pred: Predictions [n_samples, horizon, n_features]
        feature_names: List of feature names
        sample_idx: Which sample to plot (default: 0)
        model_name: Name of the model for title
        scenario: Scenario identifier (A, B, C)
        save_dir: Directory to save plots
        timestep_seconds: Seconds per timestep (default 30)
        show_plot: Whether to display the plot
        
    Returns:
        Path to saved figure
    """
    os.makedirs(save_dir, exist_ok=True)
    
    horizon = y_true.shape[1]
    n_features = len(feature_names)
    
    # Convert timesteps to hours for x-axis
    hours = np.arange(horizon) * timestep_seconds / 3600
    
    # Create figure with subplot for each feature
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 4 * n_features), sharex=True)
    if n_features == 1:
        axes = [axes]
    
    fig.suptitle(f'{model_name} - Scenario {scenario} - Sample {sample_idx}\nForecast vs Actual', 
                 fontsize=14, fontweight='bold')
    
    for feat_idx, (ax, feat_name) in enumerate(zip(axes, feature_names)):
        actual = y_true[sample_idx, :, feat_idx]
        predicted = y_pred[sample_idx, :, feat_idx]
        
        # Plot actual and predicted
        ax.plot(hours, actual, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
        ax.plot(hours, predicted, 'r--', label='Predicted', linewidth=1.5, alpha=0.8)
        
        # Fill between for error visualization
        ax.fill_between(hours, actual, predicted, alpha=0.2, color='gray')
        
        # Compute sample metrics
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mae = np.mean(np.abs(actual - predicted))
        
        ax.set_ylabel(feat_name, fontsize=11)
        ax.set_title(f'{feat_name} (RMSE: {rmse:.4f}, MAE: {mae:.4f})', fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add vertical lines at key horizons
        key_hours = [1, 3, 6, 12, 18, 24]
        for h in key_hours:
            if h <= hours[-1]:
                ax.axvline(x=h, color='gray', linestyle=':', alpha=0.5)
    
    axes[-1].set_xlabel('Forecast Horizon (hours)', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(save_dir, f'{model_name.lower()}_scenario_{scenario}_sample_{sample_idx}_forecast.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    logger.info(f"Saved forecast plot to {save_path}")
    return save_path


def plot_forecast_multiple_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    n_samples: int = 5,
    model_name: str = "Model",
    scenario: str = "A",
    save_dir: str = "plots",
    timestep_seconds: int = 30,
    show_plot: bool = False
) -> List[str]:
    """
    Plot forecasts for multiple samples.
    
    Args:
        y_true: Ground truth [n_samples, horizon, n_features]
        y_pred: Predictions [n_samples, horizon, n_features]
        feature_names: List of feature names
        n_samples: Number of samples to plot
        model_name: Name of the model
        scenario: Scenario identifier
        save_dir: Directory to save plots
        timestep_seconds: Seconds per timestep
        show_plot: Whether to display plots
        
    Returns:
        List of paths to saved figures
    """
    total_samples = y_true.shape[0]
    
    # Select evenly spaced sample indices
    sample_indices = np.linspace(0, total_samples - 1, n_samples, dtype=int)
    
    saved_paths = []
    for idx in sample_indices:
        path = plot_forecast_vs_actual(
            y_true=y_true,
            y_pred=y_pred,
            feature_names=feature_names,
            sample_idx=idx,
            model_name=model_name,
            scenario=scenario,
            save_dir=save_dir,
            timestep_seconds=timestep_seconds,
            show_plot=show_plot
        )
        saved_paths.append(path)
    
    return saved_paths


def plot_forecast_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    model_name: str = "Model",
    scenario: str = "A",
    save_dir: str = "plots",
    timestep_seconds: int = 30,
    show_plot: bool = False
) -> str:
    """
    Plot summary of forecast performance across all samples.
    Shows mean prediction with confidence intervals.
    
    Args:
        y_true: Ground truth [n_samples, horizon, n_features]
        y_pred: Predictions [n_samples, horizon, n_features]
        feature_names: List of feature names
        model_name: Name of the model
        scenario: Scenario identifier
        save_dir: Directory to save plots
        timestep_seconds: Seconds per timestep
        show_plot: Whether to display plot
        
    Returns:
        Path to saved figure
    """
    os.makedirs(save_dir, exist_ok=True)
    
    horizon = y_true.shape[1]
    n_features = len(feature_names)
    
    # Convert timesteps to hours
    hours = np.arange(horizon) * timestep_seconds / 3600
    
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 4 * n_features), sharex=True)
    if n_features == 1:
        axes = [axes]
    
    fig.suptitle(f'{model_name} - Scenario {scenario}\nMean Forecast Error Over Time', 
                 fontsize=14, fontweight='bold')
    
    for feat_idx, (ax, feat_name) in enumerate(zip(axes, feature_names)):
        # Compute error at each timestep
        errors = y_pred[:, :, feat_idx] - y_true[:, :, feat_idx]
        
        mean_error = np.mean(errors, axis=0)
        std_error = np.std(errors, axis=0)
        
        # Plot mean error with confidence band
        ax.plot(hours, mean_error, 'b-', label='Mean Error', linewidth=1.5)
        ax.fill_between(hours, mean_error - std_error, mean_error + std_error, 
                        alpha=0.3, color='blue', label='±1 Std')
        ax.fill_between(hours, mean_error - 2*std_error, mean_error + 2*std_error, 
                        alpha=0.15, color='blue', label='±2 Std')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylabel(f'{feat_name}\nError', fontsize=10)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Compute RMSE at key horizons
        key_horizons = [1, 120, 720, 1440, 2880]
        rmse_text = []
        for h in key_horizons:
            if h <= horizon:
                rmse_h = np.sqrt(np.mean(errors[:, h-1] ** 2))
                hours_h = h * timestep_seconds / 3600
                rmse_text.append(f'h={hours_h:.1f}h: {rmse_h:.3f}')
        
        ax.set_title(f'{feat_name} - RMSE: {", ".join(rmse_text)}', fontsize=10)
    
    axes[-1].set_xlabel('Forecast Horizon (hours)', fontsize=11)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{model_name.lower()}_scenario_{scenario}_error_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    logger.info(f"Saved error summary plot to {save_path}")
    return save_path


def plot_per_horizon_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    model_name: str = "Model",
    scenario: str = "A",
    save_dir: str = "plots",
    timestep_seconds: int = 30,
    show_plot: bool = False
) -> str:
    """
    Plot RMSE as a function of forecast horizon.
    
    Args:
        y_true: Ground truth [n_samples, horizon, n_features]
        y_pred: Predictions [n_samples, horizon, n_features]
        feature_names: List of feature names
        model_name: Name of the model
        scenario: Scenario identifier
        save_dir: Directory to save plots
        timestep_seconds: Seconds per timestep
        show_plot: Whether to display plot
        
    Returns:
        Path to saved figure
    """
    os.makedirs(save_dir, exist_ok=True)
    
    horizon = y_true.shape[1]
    hours = np.arange(horizon) * timestep_seconds / 3600
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for feat_idx, feat_name in enumerate(feature_names):
        # Compute RMSE at each horizon
        errors = (y_pred[:, :, feat_idx] - y_true[:, :, feat_idx]) ** 2
        rmse_per_horizon = np.sqrt(np.mean(errors, axis=0))
        
        ax.plot(hours, rmse_per_horizon, label=feat_name, linewidth=2, 
                color=colors[feat_idx % len(colors)])
    
    ax.set_xlabel('Forecast Horizon (hours)', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'{model_name} - Scenario {scenario}\nRMSE vs Forecast Horizon', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Mark key horizons
    key_hours = [1, 3, 6, 12, 18, 24]
    for h in key_hours:
        if h <= hours[-1]:
            ax.axvline(x=h, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{model_name.lower()}_scenario_{scenario}_rmse_horizon.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    logger.info(f"Saved RMSE vs horizon plot to {save_path}")
    return save_path


def generate_all_forecast_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    model_name: str = "Model",
    scenario: str = "A",
    save_dir: str = "plots",
    n_sample_plots: int = 3,
    timestep_seconds: int = 30,
    show_plots: bool = False
) -> Dict[str, List[str]]:
    """
    Generate all forecast visualization plots.
    
    Args:
        y_true: Ground truth [n_samples, horizon, n_features]
        y_pred: Predictions [n_samples, horizon, n_features]
        feature_names: List of feature names
        model_name: Name of the model
        scenario: Scenario identifier
        save_dir: Directory to save plots
        n_sample_plots: Number of individual sample plots
        timestep_seconds: Seconds per timestep
        show_plots: Whether to display plots
        
    Returns:
        Dictionary with paths to all generated plots
    """
    model_save_dir = os.path.join(save_dir, f'{model_name.lower()}_scenario_{scenario}')
    os.makedirs(model_save_dir, exist_ok=True)
    
    logger.info(f"Generating forecast plots for {model_name} - Scenario {scenario}...")
    
    results = {
        'sample_plots': [],
        'summary_plot': None,
        'rmse_horizon_plot': None
    }
    
    # Generate individual sample plots
    results['sample_plots'] = plot_forecast_multiple_samples(
        y_true=y_true,
        y_pred=y_pred,
        feature_names=feature_names,
        n_samples=n_sample_plots,
        model_name=model_name,
        scenario=scenario,
        save_dir=model_save_dir,
        timestep_seconds=timestep_seconds,
        show_plot=show_plots
    )
    
    # Generate error summary plot
    results['summary_plot'] = plot_forecast_summary(
        y_true=y_true,
        y_pred=y_pred,
        feature_names=feature_names,
        model_name=model_name,
        scenario=scenario,
        save_dir=model_save_dir,
        timestep_seconds=timestep_seconds,
        show_plot=show_plots
    )
    
    # Generate RMSE vs horizon plot
    results['rmse_horizon_plot'] = plot_per_horizon_rmse(
        y_true=y_true,
        y_pred=y_pred,
        feature_names=feature_names,
        model_name=model_name,
        scenario=scenario,
        save_dir=model_save_dir,
        timestep_seconds=timestep_seconds,
        show_plot=show_plots
    )
    
    logger.info(f"Generated {len(results['sample_plots']) + 2} plots in {model_save_dir}")
    
    return results


if __name__ == "__main__":
    # Example usage with dummy data
    np.random.seed(42)
    
    n_samples = 100
    horizon = 2880
    n_features = 4
    feature_names = ['acc_p3', 'acc_p4', 'temp_p3', 'temp_p4']
    
    # Generate dummy data
    y_true = np.random.randn(n_samples, horizon, n_features)
    y_pred = y_true + np.random.randn(n_samples, horizon, n_features) * 0.5
    
    # Generate all plots
    results = generate_all_forecast_plots(
        y_true=y_true,
        y_pred=y_pred,
        feature_names=feature_names,
        model_name="Test",
        scenario="A",
        save_dir="plots",
        n_sample_plots=3,
        show_plots=False
    )
    
    print("Generated plots:")
    for key, value in results.items():
        print(f"  {key}: {value}")
