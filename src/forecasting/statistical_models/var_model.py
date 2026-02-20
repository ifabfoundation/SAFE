"""
VAR (Vector AutoRegression) model for multivariate time series forecasting.

Implements direct forecasting using a single multivariate VAR model
that captures cross-correlations between all features (acc_p3, acc_p4, temp_p3, temp_p4).

VAR models are useful when features are interdependent.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import json
import os
import pickle
from typing import Dict, Tuple
import logging
import warnings

warnings.filterwarnings('ignore')

from models.utils.data_loader import load_data_for_scenario
from models.utils.evaluation import evaluate_forecast, convert_to_native_types
from models.utils.forecast_plots import generate_all_forecast_plots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VARForecaster:
    """VAR-based multivariate forecaster with recursive prediction."""

    def __init__(self, feature_names: list, maxlags: int = None):
        """
        Initialize VAR forecaster.

        Args:
            feature_names: List of feature names to forecast
            maxlags: Maximum number of lags to consider (None = auto-select with AIC)
        """
        self.feature_names = feature_names
        self.maxlags = maxlags
        self.model = None
        self.model_fit = None

    def fit(self, train_df: pd.DataFrame):
        """
        Fit VAR model on all features jointly.

        Args:
            train_df: Training data with DatetimeIndex and feature columns
        """
        logger.info(f"Fitting VAR model with maxlags={self.maxlags}...")

        # Prepare data (all features as columns)
        train_data = train_df[self.feature_names].values

        try:
            # Create and fit VAR model
            model = VAR(train_data)

            # Select optimal lag order if maxlags not specified
            if self.maxlags is None:
                # Use AIC to select lag order (limit search to reasonable range)
                lag_order_results = model.select_order(maxlags=15)
                self.maxlags = lag_order_results.aic
                logger.info(f"  Selected lag order: {self.maxlags} (via AIC)")

            self.model_fit = model.fit(maxlags=self.maxlags)
            self.model = model

            logger.info(f"  AIC: {self.model_fit.aic:.2f}, BIC: {self.model_fit.bic:.2f}")
            logger.info(f"  Number of parameters: {self.model_fit.k_ar * self.model_fit.neqs**2}")

        except Exception as e:
            logger.error(f"  Failed to fit VAR: {e}")
            raise

        logger.info("VAR model fitted successfully")

    def predict_direct(self, history_df: pd.DataFrame, steps: int) -> np.ndarray:
        """
        Direct multi-step forecast without refitting.
        Uses VAR's native forecast() method which is much faster than recursive refitting.
        
        Args:
            history_df: Historical data DataFrame with feature columns
            steps: Number of steps to forecast
            
        Returns:
            predictions: Array of shape (steps, n_features)
        """
        if self.model_fit is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get recent history as numpy array
        history = history_df[self.feature_names].values
        
        # VAR forecast requires history of at least lag_order
        lag_order = self.model_fit.k_ar
        if len(history) < lag_order:
            raise ValueError(f"History length {len(history)} < lag order {lag_order}")
        
        # Use only the most recent lag_order observations
        recent_history = history[-lag_order:]
        
        # Direct forecast for all steps at once (much faster!)
        forecast = self.model_fit.forecast(y=recent_history, steps=steps)
        
        return forecast

    def predict_recursive(self,
                         initial_history: pd.DataFrame,
                         forecast_horizon: int = 2880) -> np.ndarray:
        """
        Recursive forecasting: predict next point, add to history, repeat.

        Args:
            initial_history: Historical data to start from
            forecast_horizon: Number of timesteps to forecast

        Returns:
            predictions: [forecast_horizon, n_features]
        """
        # Get history for all features
        history = initial_history[self.feature_names].values.copy()

        # Fit VAR on this specific history
        model = VAR(history)
        model_fit = model.fit(maxlags=self.maxlags)

        predictions = np.zeros((forecast_horizon, len(self.feature_names)))

        # Recursive forecasting
        for h in range(forecast_horizon):
            # Predict next timestep for all features
            forecast = model_fit.forecast(history[-model_fit.k_ar:], steps=1)
            pred_value = forecast[0]  # [n_features]

            predictions[h] = pred_value

            # Add prediction to history for next iteration
            history = np.vstack([history, pred_value])

            # Refit model with updated history
            model = VAR(history)
            model_fit = model.fit(maxlags=self.maxlags)

        return predictions

    def forecast(self, test_df: pd.DataFrame, forecast_horizon: int = 2880, max_samples: int = 100) -> np.ndarray:
        """
        Generate forecasts for sampled positions in test set using direct forecasting.
        
        Args:
            test_df: Test data with DatetimeIndex
            forecast_horizon: Number of timesteps to forecast
            max_samples: Maximum number of samples to evaluate (for efficiency)

        Returns:
            predictions: [n_samples, forecast_horizon, n_features]
        """
        # Determine how many forecasts we can make
        # For VAR, use minimum 1 day of history (2880 timesteps at 30s intervals)
        min_history = 2880
        n_possible = len(test_df) - min_history - forecast_horizon + 1  # +1 to include the boundary case

        if n_possible < 1:
            raise ValueError(f"Test set too small for forecasting. Need at least {min_history + forecast_horizon} timesteps, got {len(test_df)}")

        # Sample positions instead of using all (for efficiency)
        n_samples = min(max_samples, n_possible)
        sample_indices = np.linspace(0, n_possible - 1, n_samples, dtype=int)
        
        predictions = np.zeros((n_samples, forecast_horizon, len(self.feature_names)))

        logger.info(f"Generating {n_samples} forecasts (sampled from {n_possible} possible) with horizon {forecast_horizon}...")

        for idx, i in enumerate(sample_indices):
            if idx % 20 == 0:
                logger.info(f"  Progress: {idx}/{n_samples} samples")

            # Extract history up to this point
            history_end_idx = min_history + i
            history = test_df.iloc[:history_end_idx]

            # Direct prediction (much faster than recursive!)
            try:
                predictions[idx] = self.predict_direct(history, forecast_horizon)
            except Exception as e:
                logger.warning(f"Failed to predict for sample {idx}: {str(e)}")
                # Fill with NaN on failure
                predictions[idx] = np.nan

        logger.info(f"Generated {n_samples} forecasts")
        return predictions, sample_indices

    def save(self, save_dir: str):
        """Save model to disk."""
        os.makedirs(save_dir, exist_ok=True)

        # Save metadata (convert numpy types to native Python types for JSON)
        metadata = {
            'maxlags': int(self.maxlags) if self.maxlags is not None else None,
            'feature_names': self.feature_names
        }

        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save fitted model
        save_path = os.path.join(save_dir, 'var_model.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self.model_fit, f)

        logger.info(f"Saved VAR model to {save_dir}")

    def load(self, save_dir: str):
        """Load model from disk."""
        # Load metadata
        with open(os.path.join(save_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        self.maxlags = metadata['maxlags']
        self.feature_names = metadata['feature_names']

        # Load fitted model
        load_path = os.path.join(save_dir, 'var_model.pkl')
        with open(load_path, 'rb') as f:
            self.model_fit = pickle.load(f)

        logger.info(f"Loaded VAR model from {save_dir}")


def run_var_experiment(scenario: str = 'A', forecast_horizon: int = 2880):
    """
    Run complete VAR experiment for a scenario.

    Args:
        scenario: 'A', 'B', or 'C'
        forecast_horizon: Number of timesteps to forecast
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running VAR experiment - Scenario {scenario}")
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
    forecaster = VARForecaster(feature_names, maxlags=None)  # Auto-select lag order
    forecaster.fit(train_df)

    # Save model
    save_dir = f"weights/var_scenario_{scenario}"
    forecaster.save(save_dir)

    # Evaluate on validation set
    logger.info("\n--- Validation ---")
    min_history = 2880
    val_results = None
    n_val_samples = 0
    
    try:
        y_val_pred, val_sample_indices = forecaster.forecast(val_df, forecast_horizon)
        n_val_samples = len(y_val_pred)

        # Create ground truth for validation using sample indices
        y_val_true = np.zeros((n_val_samples, forecast_horizon, len(feature_names)))

        for idx, i in enumerate(val_sample_indices):
            for feat_idx, feat in enumerate(feature_names):
                start_idx = min_history + i
                end_idx = start_idx + forecast_horizon
                y_val_true[idx, :, feat_idx] = val_df[feat].values[start_idx:end_idx]

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
        y_test_pred, test_sample_indices = forecaster.forecast(test_df, forecast_horizon)
        n_test_samples = len(y_test_pred)

        # Create ground truth for test using sample indices
        y_test_true = np.zeros((n_test_samples, forecast_horizon, len(feature_names)))

        for idx, i in enumerate(test_sample_indices):
            for feat_idx, feat in enumerate(feature_names):
                start_idx = min_history + i
                end_idx = start_idx + forecast_horizon
                y_test_true[idx, :, feat_idx] = test_df[feat].values[start_idx:end_idx]

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
            model_name="VAR",
            scenario=scenario,
            save_dir="plots",
            n_sample_plots=3
        )

    # Save results
    results = {
        'model': 'VAR',
        'scenario': scenario,
        'forecast_horizon': forecast_horizon,
        'approach': 'direct',  # Changed from recursive to direct forecasting
        'maxlags': int(forecaster.maxlags) if forecaster.maxlags is not None else None,
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
    results_path = f'results/var_scenario_{scenario}_results.json'

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")

    # Print summary
    if test_results is not None:
        logger.info(f"\n{'='*80}")
        logger.info("SUMMARY - Test Set Performance")
        logger.info(f"{'='*80}")
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

    run_var_experiment(scenario=scenario, forecast_horizon=2880)
