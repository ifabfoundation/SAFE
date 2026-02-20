"""
ARIMA model for time series forecasting.

Implements DIRECT forecasting using 4 separate univariate ARIMA models
(one for each feature: acc_p3, acc_p4, temp_p3, temp_p4).

Supports:
- Manual order specification (p, d, q)
- Auto-ARIMA for automatic order selection (uses pmdarima)

Note: Using ARIMA instead of full SARIMA due to very large seasonal period
(2880 timesteps = 24h at 30s intervals), which makes seasonal fitting impractical.

NOTE: ARIMA can forecast multiple steps directly using model.forecast(steps=n),
no need for recursive refitting!
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import json
import os
import pickle
from typing import Dict, Tuple, Optional
import logging
import warnings

warnings.filterwarnings('ignore')

from models.utils.data_loader import load_data_for_scenario
from models.utils.evaluation import evaluate_forecast, convert_to_native_types
from models.utils.forecast_plots import generate_all_forecast_plots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ARIMAForecaster:
    """ARIMA-based forecaster with direct multi-step prediction."""

    def __init__(self, feature_names: list, order: Optional[Tuple[int, int, int]] = None,
                 use_auto_arima: bool = False):
        """
        Initialize ARIMA forecaster.

        Args:
            feature_names: List of feature names to forecast
            order: (p, d, q) order for ARIMA model. If None and use_auto_arima=True,
                   will be determined automatically.
            use_auto_arima: If True, use pmdarima to find optimal order
        """
        self.feature_names = feature_names
        self.order = order if order else (5, 1, 2)  # Default AR(5), d=1, MA(2)
        self.use_auto_arima = use_auto_arima
        self.models = {}
        self.model_fits = {}
        self.orders = {}  # Store order per feature (may differ with auto_arima)

    def fit(self, train_df: pd.DataFrame):
        """
        Fit separate ARIMA models for each feature.

        Args:
            train_df: Training data with DatetimeIndex and feature columns
        """
        if self.use_auto_arima:
            logger.info("Fitting Auto-ARIMA models (finding optimal order)...")
            try:
                import pmdarima as pm
            except ImportError:
                logger.error("pmdarima not installed. Run: pip install pmdarima")
                raise
        else:
            logger.info(f"Fitting ARIMA{self.order} models...")

        for feat in self.feature_names:
            logger.info(f"  Fitting ARIMA for {feat}...")

            try:
                if self.use_auto_arima:
                    import pmdarima as pm
                    # Auto-ARIMA with reasonable constraints
                    auto_model = pm.auto_arima(
                        train_df[feat].values,
                        start_p=1, max_p=5,
                        start_q=0, max_q=2,
                        d=None,  # Auto-determine differencing
                        max_d=2,
                        seasonal=False,  # No seasonality (period too large)
                        stepwise=True,  # Faster
                        suppress_warnings=True,
                        error_action='ignore',
                        trace=False,
                        n_fits=20  # Limit search
                    )
                    
                    feat_order = auto_model.order
                    logger.info(f"    Auto-ARIMA selected order: {feat_order}")
                    
                    # Store and refit with statsmodels for consistency
                    self.orders[feat] = feat_order
                    model = ARIMA(train_df[feat].values, order=feat_order)
                    model_fit = model.fit()
                else:
                    feat_order = self.order
                    self.orders[feat] = feat_order
                    model = ARIMA(train_df[feat].values, order=feat_order)
                    model_fit = model.fit()

                self.models[feat] = model
                self.model_fits[feat] = model_fit

                logger.info(f"    Order: {feat_order}, AIC: {model_fit.aic:.2f}, BIC: {model_fit.bic:.2f}")

            except Exception as e:
                logger.error(f"    Failed to fit ARIMA for {feat}: {e}")
                raise

        logger.info(f"Fitted {len(self.models)} ARIMA models")

    def predict_direct(self,
                       history: np.ndarray,
                       feat: str,
                       forecast_horizon: int = 2880) -> np.ndarray:
        """
        Direct multi-step forecasting (no refitting!).

        Args:
            history: Historical data for this feature
            feat: Feature name
            forecast_horizon: Number of timesteps to forecast

        Returns:
            predictions: [forecast_horizon]
        """
        # Fit ARIMA on this specific history window
        feat_order = self.orders.get(feat, self.order)
        model = ARIMA(history, order=feat_order)
        model_fit = model.fit()
        
        # Direct forecast all steps at once!
        forecast = model_fit.forecast(steps=forecast_horizon)
        return forecast

    def forecast(self, test_df: pd.DataFrame, forecast_horizon: int = 2880,
                 max_samples: int = 100, sample_step: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate forecasts for test set with sampling.

        Args:
            test_df: Test data with DatetimeIndex
            forecast_horizon: Number of timesteps to forecast
            max_samples: Maximum number of forecast samples
            sample_step: Step between samples (if None, computed from max_samples)

        Returns:
            predictions: [n_samples, forecast_horizon, n_features]
            sample_indices: Array of starting indices for each sample
        """
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

            # Extract history up to this point
            history_end_idx = min_history + sample_idx

            for feat_idx, feat in enumerate(self.feature_names):
                history = test_df[feat].values[:history_end_idx]
                predictions[i, :, feat_idx] = self.predict_direct(history, feat, forecast_horizon)

        logger.info(f"Generated {n_samples} forecasts")
        return predictions, np.array(sample_indices)

    def save(self, save_dir: str):
        """Save models to disk."""
        os.makedirs(save_dir, exist_ok=True)

        # Save metadata including per-feature orders
        metadata = {
            'default_order': self.order,
            'orders': {feat: list(order) for feat, order in self.orders.items()},
            'feature_names': self.feature_names,
            'use_auto_arima': self.use_auto_arima
        }

        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save fitted models
        for feat, model_fit in self.model_fits.items():
            save_path = os.path.join(save_dir, f'arima_{feat}.pkl')
            model_fit.save(save_path)

        logger.info(f"Saved ARIMA models to {save_dir}")

    def load(self, save_dir: str):
        """Load models from disk."""
        # Load metadata
        with open(os.path.join(save_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        self.order = tuple(metadata['default_order'])
        self.orders = {feat: tuple(order) for feat, order in metadata['orders'].items()}
        self.feature_names = metadata['feature_names']
        self.use_auto_arima = metadata.get('use_auto_arima', False)

        # Load fitted models
        from statsmodels.tsa.arima.model import ARIMAResults

        for feat in self.feature_names:
            load_path = os.path.join(save_dir, f'arima_{feat}.pkl')
            self.model_fits[feat] = ARIMAResults.load(load_path)

        logger.info(f"Loaded ARIMA models from {save_dir}")


def run_arima_experiment(scenario: str = 'A', forecast_horizon: int = 2880,
                         use_auto_arima: bool = False, max_samples: int = 100):
    """
    Run complete ARIMA experiment for a scenario.

    Args:
        scenario: 'A', 'B', or 'C'
        forecast_horizon: Number of timesteps to forecast
        use_auto_arima: If True, use auto_arima to find optimal order
        max_samples: Maximum samples for evaluation
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running ARIMA experiment - Scenario {scenario}")
    if use_auto_arima:
        logger.info("Using Auto-ARIMA for order selection")
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
    forecaster = ARIMAForecaster(
        feature_names,
        order=(5, 1, 2),  # Default if not using auto_arima
        use_auto_arima=use_auto_arima
    )
    forecaster.fit(train_df)

    # Save model
    save_dir = f"weights/arima_scenario_{scenario}"
    forecaster.save(save_dir)

    min_history = 2880

    # Evaluate on validation set (if large enough)
    val_results = None
    if len(val_df) >= min_history + forecast_horizon:
        logger.info("\n--- Validation ---")
        y_val_pred, val_sample_indices = forecaster.forecast(val_df, forecast_horizon, max_samples=max_samples)
        n_val_samples = len(val_sample_indices)

        # Create ground truth for validation
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
    else:
        logger.warning(f"Validation set too small ({len(val_df)} rows), skipping validation")

    # Evaluate on test set (if large enough)
    test_results = None
    if len(test_df) >= min_history + forecast_horizon:
        logger.info("\n--- Testing ---")
        y_test_pred, test_sample_indices = forecaster.forecast(test_df, forecast_horizon, max_samples=max_samples)
        n_test_samples = len(test_sample_indices)

        # Create ground truth for test
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
    else:
        logger.warning(f"Test set too small ({len(test_df)} rows), skipping test evaluation")
    
    # Generate forecast plots if we have test results
    if test_results is not None:
        logger.info("\n--- Generating Forecast Plots ---")
        generate_all_forecast_plots(
            y_true=y_test_true,
            y_pred=y_test_pred,
            feature_names=feature_names,
            model_name="ARIMA",
            scenario=scenario,
            save_dir="plots",
            n_sample_plots=3
        )

    # Save results (with type conversion for JSON)
    results = {
        'model': 'ARIMA',
        'scenario': scenario,
        'forecast_horizon': forecast_horizon,
        'approach': 'direct',
        'use_auto_arima': use_auto_arima,
        'orders': {feat: list(order) for feat, order in forecaster.orders.items()},
    }
    
    if val_results is not None:
        results['validation'] = convert_to_native_types({
            'n_samples': n_val_samples,
            'aggregate': val_results['aggregate'],
            'per_horizon': val_results['per_horizon'],
            'per_feature': val_results['per_feature']
        })
    
    if test_results is not None:
        results['test'] = convert_to_native_types({
            'n_samples': n_test_samples,
            'aggregate': test_results['aggregate'],
            'per_horizon': test_results['per_horizon'],
            'per_feature': test_results['per_feature']
        })

    # Save results JSON
    os.makedirs('results', exist_ok=True)
    results_path = f'results/arima_scenario_{scenario}_results.json'

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

    return results


if __name__ == "__main__":
    import sys

    # Parse arguments: scenario [--auto]
    scenario = sys.argv[1] if len(sys.argv) > 1 else 'A'
    use_auto = '--auto' in sys.argv

    run_arima_experiment(scenario=scenario, forecast_horizon=2880, use_auto_arima=use_auto)
