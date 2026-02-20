"""
Prophet model for time series forecasting.

Implements DIRECT forecasting using 4 separate univariate Prophet models
(one for each feature: acc_p3, acc_p4, temp_p3, temp_p4).

Prophet is a procedure for forecasting time series data based on an additive model
where non-linear trends are fit with yearly, weekly, and daily seasonality.

NOTE: Prophet can predict entire forecast horizon at once (no recursion needed).
This makes it ~1000x faster than recursive approaches.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import json
import os
import pickle
from typing import Dict, Tuple
import logging
import warnings

# Suppress Prophet's verbose output
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

from models.utils.data_loader import load_data_for_scenario
from models.utils.evaluation import evaluate_forecast, convert_to_native_types
from models.utils.forecast_plots import generate_all_forecast_plots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProphetForecaster:
    """Prophet-based forecaster with direct (non-recursive) prediction."""

    def __init__(self, feature_names: list):
        """
        Initialize Prophet forecaster.

        Args:
            feature_names: List of feature names to forecast
        """
        self.feature_names = feature_names
        self.models = {}  # Will store one model per feature

    def fit(self, train_df: pd.DataFrame):
        """
        Fit separate Prophet models for each feature.

        Args:
            train_df: Training data with DatetimeIndex and feature columns
        """
        logger.info("Fitting Prophet models...")

        for feat in self.feature_names:
            logger.info(f"  Fitting Prophet for {feat}...")

            # Prepare data for Prophet (requires 'ds' and 'y' columns)
            prophet_df = pd.DataFrame({
                'ds': train_df.index,
                'y': train_df[feat].values
            })

            # Create and fit Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=False,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,  # Default: 0.05
                seasonality_prior_scale=10.0,  # Default: 10.0
                interval_width=0.95
            )

            model.fit(prophet_df)
            self.models[feat] = model

        logger.info(f"Fitted {len(self.models)} Prophet models")

    def predict_direct(self,
                       start_time: pd.Timestamp,
                       forecast_horizon: int = 2880) -> np.ndarray:
        """
        Direct forecasting: predict entire horizon at once.
        
        Prophet's additive model (trend + seasonality) can extrapolate
        any future horizon without needing recursive prediction.

        Args:
            start_time: Last timestamp of history (forecast starts after this)
            forecast_horizon: Number of timesteps to forecast

        Returns:
            predictions: [forecast_horizon, n_features]
        """
        predictions = np.zeros((forecast_horizon, len(self.feature_names)))

        # Create future dataframe with ALL forecast timesteps at once
        future_dates = pd.date_range(
            start=start_time + pd.Timedelta(seconds=30),
            periods=forecast_horizon,
            freq='30S'
        )
        future_df = pd.DataFrame({'ds': future_dates})

        for feat_idx, feat in enumerate(self.feature_names):
            # Predict entire horizon at once (NOT recursively!)
            forecast = self.models[feat].predict(future_df)
            predictions[:, feat_idx] = forecast['yhat'].values

        return predictions

    def forecast(self, test_df: pd.DataFrame, forecast_horizon: int = 2880,
                 max_samples: int = 100, sample_step: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate forecasts for test set with sampling.
        
        Instead of forecasting at every possible position (which could be
        hundreds of thousands), we sample evenly spaced positions.

        Args:
            test_df: Test data with DatetimeIndex
            forecast_horizon: Number of timesteps to forecast
            max_samples: Maximum number of forecast samples to generate
            sample_step: Step between samples (if None, computed from max_samples)

        Returns:
            predictions: [n_samples, forecast_horizon, n_features]
            sample_indices: Array of starting indices for each sample
        """
        # Use minimum 1 day of history (2880 timesteps at 30s intervals)
        min_history = 2880
        total_possible = len(test_df) - min_history - forecast_horizon + 1  # +1 to include the boundary case

        if total_possible < 1:
            raise ValueError(f"Test set too small. Need at least {min_history + forecast_horizon} timesteps, got {len(test_df)}")

        # Determine sampling strategy
        if sample_step is None:
            sample_step = max(1, total_possible // max_samples)

        sample_indices = list(range(0, total_possible, sample_step))[:max_samples]
        n_samples = len(sample_indices)

        predictions = np.zeros((n_samples, forecast_horizon, len(self.feature_names)))

        logger.info(f"Generating {n_samples} forecasts (step={sample_step}, horizon={forecast_horizon})...")

        for i, sample_idx in enumerate(sample_indices):
            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{n_samples} samples")

            # Get the timestamp to start forecasting from
            history_end_idx = min_history + sample_idx
            start_time = test_df.index[history_end_idx - 1]

            # Direct prediction (entire horizon at once)
            predictions[i] = self.predict_direct(start_time, forecast_horizon)

        logger.info(f"Generated {n_samples} forecasts")
        return predictions, np.array(sample_indices)

    def save(self, save_dir: str):
        """Save models to disk."""
        os.makedirs(save_dir, exist_ok=True)

        for feat, model in self.models.items():
            save_path = os.path.join(save_dir, f'prophet_{feat}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(model, f)

        logger.info(f"Saved Prophet models to {save_dir}")

    def load(self, save_dir: str):
        """Load models from disk."""
        for feat in self.feature_names:
            load_path = os.path.join(save_dir, f'prophet_{feat}.pkl')
            with open(load_path, 'rb') as f:
                self.models[feat] = pickle.load(f)

        logger.info(f"Loaded Prophet models from {save_dir}")


def run_prophet_experiment(scenario: str = 'A', forecast_horizon: int = 2880,
                           max_samples: int = 100):
    """
    Run complete Prophet experiment for a scenario.

    Args:
        scenario: 'A', 'B', or 'C'
        forecast_horizon: Number of timesteps to forecast
        max_samples: Maximum number of forecast samples for evaluation
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running Prophet experiment - Scenario {scenario}")
    logger.info(f"{'='*80}\n")

    # Load data
    logger.info("Loading data...")
    data = load_data_for_scenario(scenario, forecast_horizon=forecast_horizon, for_ml_model=False)

    train_df = data['train']
    val_df = data['val']
    test_df = data['test']
    feature_names = data['feature_columns']

    logger.info(f"Train shape: {train_df.shape}")
    logger.info(f"Val shape: {val_df.shape}")
    logger.info(f"Test shape: {test_df.shape}")

    # Initialize and fit model
    logger.info("\n--- Training ---")
    forecaster = ProphetForecaster(feature_names)
    forecaster.fit(train_df)

    # Save model
    save_dir = f"weights/prophet_scenario_{scenario}"
    forecaster.save(save_dir)

    # Evaluate on validation set
    logger.info("\n--- Validation ---")
    min_history = 2880
    val_results = None
    n_val_samples = 0
    
    try:
        y_val_pred, val_sample_indices = forecaster.forecast(val_df, forecast_horizon, max_samples=max_samples)
        n_val_samples = len(val_sample_indices)

        # Create ground truth for validation (matching sampled positions)
        y_val_true = np.zeros((n_val_samples, forecast_horizon, len(feature_names)))

        for i, sample_idx in enumerate(val_sample_indices):
            for feat_idx, feat in enumerate(feature_names):
                start_idx = min_history + sample_idx
                end_idx = start_idx + forecast_horizon
                y_val_true[i, :, feat_idx] = val_df[feat].values[start_idx:end_idx]

        # Compute validation metrics
        val_results = evaluate_forecast(
            y_true=y_val_true,
            y_pred=y_val_pred,
            y_train=train_df[feature_names].values,
            feature_names=feature_names,
            horizons=[1, 120, 720, 1440, 2880]
        )
    except ValueError as e:
        logger.warning(f"Skipping validation - {e}")

    # Evaluate on test set
    logger.info("\n--- Testing ---")
    test_results = None
    n_test_samples = 0
    
    try:
        y_test_pred, test_sample_indices = forecaster.forecast(test_df, forecast_horizon, max_samples=max_samples)
        n_test_samples = len(test_sample_indices)

        # Create ground truth for test (matching sampled positions)
        y_test_true = np.zeros((n_test_samples, forecast_horizon, len(feature_names)))

        for i, sample_idx in enumerate(test_sample_indices):
            for feat_idx, feat in enumerate(feature_names):
                start_idx = min_history + sample_idx
                end_idx = start_idx + forecast_horizon
                y_test_true[i, :, feat_idx] = test_df[feat].values[start_idx:end_idx]

        # Compute test metrics
        test_results = evaluate_forecast(
            y_true=y_test_true,
            y_pred=y_test_pred,
            y_train=train_df[feature_names].values,
            feature_names=feature_names,
            horizons=[1, 120, 720, 1440, 2880]
        )
    except ValueError as e:
        logger.warning(f"Skipping testing - {e}")
    
    # Generate forecast plots (only if test results available)
    if test_results is not None:
        logger.info("\n--- Generating Forecast Plots ---")
        generate_all_forecast_plots(
            y_true=y_test_true,
            y_pred=y_test_pred,
            feature_names=feature_names,
            model_name="Prophet",
            scenario=scenario,
            save_dir="plots",
            n_sample_plots=3
        )

    # Save results (with type conversion for JSON)
    results = {
        'model': 'Prophet',
        'scenario': scenario,
        'forecast_horizon': forecast_horizon,
        'approach': 'direct',
        'n_val_samples': n_val_samples,
        'n_test_samples': n_test_samples,
        'validation': convert_to_native_types({
            'aggregate': val_results['aggregate'],
            'per_horizon': val_results['per_horizon'],
            'per_feature': val_results['per_feature']
        }) if val_results is not None else None,
        'test': convert_to_native_types({
            'aggregate': test_results['aggregate'],
            'per_horizon': test_results['per_horizon'],
            'per_feature': test_results['per_feature']
        }) if test_results is not None else None
    }

    # Save results JSON
    os.makedirs('results', exist_ok=True)
    results_path = f'results/prophet_scenario_{scenario}_results.json'

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")

    # Print summary
    if test_results is not None:
        logger.info(f"\n{'='*80}")
        logger.info("SUMMARY - Test Set Performance")
        logger.info(f"{'='*80}")
        logger.info(f"Samples evaluated: {n_test_samples}")
        logger.info(f"Aggregate RMSE: {test_results['aggregate']['rmse']:.4f}")
        logger.info(f"Aggregate MAE:  {test_results['aggregate']['mae']:.4f}")
        logger.info(f"Aggregate R2:   {test_results['aggregate']['r2']:.4f}")
        logger.info(f"Aggregate MAPE: {test_results['aggregate']['mape']:.2f}%")
        logger.info(f"Aggregate MASE: {test_results['aggregate']['mase']:.4f}")

        logger.info("\nPer-feature RMSE:")
        for feat, metrics in test_results['per_feature'].items():
            logger.info(f"  {feat}: {metrics['rmse']:.4f}")

        logger.info(f"\n{'='*80}\n")
    elif val_results is not None:
        logger.info(f"\n{'='*80}")
        logger.info("SUMMARY - Validation Set Performance (test set unavailable)")
        logger.info(f"{'='*80}")
        logger.info(f"Samples evaluated: {n_val_samples}")
        logger.info(f"Aggregate RMSE: {val_results['aggregate']['rmse']:.4f}")
        logger.info(f"Aggregate MAE:  {val_results['aggregate']['mae']:.4f}")
        logger.info(f"Aggregate R2:   {val_results['aggregate']['r2']:.4f}")
        logger.info(f"Aggregate MAPE: {val_results['aggregate']['mape']:.2f}%")
        logger.info(f"Aggregate MASE: {val_results['aggregate']['mase']:.4f}")
        logger.info(f"\n{'='*80}\n")
    else:
        logger.warning("No evaluation results available (both validation and test sets too small)")

    return results


if __name__ == "__main__":
    import sys

    # Default to scenario A if not specified
    scenario = sys.argv[1] if len(sys.argv) > 1 else 'A'

    run_prophet_experiment(scenario=scenario, forecast_horizon=2880)
