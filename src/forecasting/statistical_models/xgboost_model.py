"""
XGBoost model for time series forecasting.

Implements DIRECT multi-horizon forecasting:
- Trains separate models for key horizons (1h, 6h, 12h, 24h)
- Interpolates between horizons for full 2880-step prediction
- Much faster than recursive (48 model calls vs 2880) or 
  multi-output (which would train 2880 separate models)

Uses lag features created by data_loader.py.
"""

import numpy as np
import xgboost as xgb
import json
import os
import pickle
from typing import Dict, List
import logging

from models.utils.data_loader import load_data_for_scenario
from models.utils.evaluation import evaluate_forecast, convert_to_native_types
from models.utils.forecast_plots import generate_all_forecast_plots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sampling constant for fair comparison with statistical models
MAX_EVAL_SAMPLES = 100


class XGBoostForecaster:
    """
    XGBoost forecaster with direct multi-horizon approach.
    
    Strategy: Train models at key horizons and interpolate between them.
    This is much more practical than:
    - Recursive: 2880 iterations with feature drift
    - MultiOutput: Training 2880 separate XGBoost models
    """

    # Key horizons to model (timesteps at 30s intervals)
    # 1=30s, 60=30min, 120=1h, 360=3h, 720=6h, 1440=12h, 2160=18h, 2880=24h
    KEY_HORIZONS = [1, 60, 120, 360, 720, 1440, 2160, 2880]

    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: int = 8,
                 learning_rate: float = 0.05,
                 n_jobs: int = -1,
                 early_stopping_rounds: int = 20):
        """
        Initialize XGBoost forecaster.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate
            n_jobs: Number of parallel jobs (-1 = all cores)
            early_stopping_rounds: Early stopping patience
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        
        # Models: {feature_name: {horizon: model}}
        self.models: Dict[str, Dict[int, xgb.XGBRegressor]] = {}
        self.feature_names: List[str] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list,
            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Fit XGBoost models for each feature at each key horizon.

        Args:
            X_train: Training features [n_samples, n_features]
            y_train: Training targets [n_samples, horizon, n_targets]
            feature_names: Names of target features
            X_val: Validation features (optional, for early stopping)
            y_val: Validation targets (optional)
        """
        self.feature_names = feature_names
        n_targets = len(feature_names)
        
        # Total models to train: n_features * n_horizons
        total_models = n_targets * len(self.KEY_HORIZONS)
        logger.info(f"Training {total_models} XGBoost models ({n_targets} features × {len(self.KEY_HORIZONS)} horizons)...")

        model_count = 0
        for feat_idx, feat_name in enumerate(feature_names):
            self.models[feat_name] = {}
            
            for horizon in self.KEY_HORIZONS:
                model_count += 1
                logger.info(f"  [{model_count}/{total_models}] Training {feat_name} @ h={horizon}...")
                
                # Target: value at specific horizon (horizon-1 for 0-indexing)
                y_train_h = y_train[:, horizon - 1, feat_idx]
                
                model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    n_jobs=self.n_jobs,
                    random_state=42,
                    tree_method='hist',
                    early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None
                )
                
                if X_val is not None and y_val is not None:
                    y_val_h = y_val[:, horizon - 1, feat_idx]
                    model.fit(X_train, y_train_h, 
                             eval_set=[(X_val, y_val_h)],
                             verbose=False)
                else:
                    model.fit(X_train, y_train_h)
                
                self.models[feat_name][horizon] = model

        logger.info(f"Fitted {model_count} XGBoost models")

    def predict(self, X: np.ndarray, forecast_horizon: int = 2880) -> np.ndarray:
        """
        Generate predictions for all timesteps using interpolation.

        Args:
            X: Features [n_samples, n_features]
            forecast_horizon: Number of timesteps to predict (default 2880)

        Returns:
            predictions: [n_samples, horizon, n_features]
        """
        n_samples = X.shape[0]
        n_features = len(self.feature_names)
        predictions = np.zeros((n_samples, forecast_horizon, n_features))

        logger.info(f"Generating forecasts for {n_samples} samples...")

        for feat_idx, feat_name in enumerate(self.feature_names):
            logger.info(f"  Predicting {feat_name}...")
            
            # Predict at all key horizons
            key_preds = {}
            for horizon in self.KEY_HORIZONS:
                if horizon <= forecast_horizon:
                    key_preds[horizon] = self.models[feat_name][horizon].predict(X)
            
            # Interpolate to fill all timesteps
            horizons_used = sorted([h for h in key_preds.keys() if h <= forecast_horizon])
            
            for t in range(forecast_horizon):
                h = t + 1  # 1-indexed horizon
                
                if h in key_preds:
                    # Exact match - use model prediction
                    predictions[:, t, feat_idx] = key_preds[h]
                else:
                    # Interpolate between nearest key horizons
                    h_lower = max([hk for hk in horizons_used if hk < h], default=horizons_used[0])
                    h_upper = min([hk for hk in horizons_used if hk > h], default=horizons_used[-1])
                    
                    if h_lower == h_upper:
                        predictions[:, t, feat_idx] = key_preds[h_lower]
                    else:
                        # Linear interpolation
                        alpha = (h - h_lower) / (h_upper - h_lower)
                        predictions[:, t, feat_idx] = (1 - alpha) * key_preds[h_lower] + alpha * key_preds[h_upper]

        logger.info(f"Generated {n_samples} forecasts")
        return predictions

    def save(self, save_dir: str):
        """Save models to disk."""
        os.makedirs(save_dir, exist_ok=True)

        # Save metadata
        metadata = {
            'approach': 'direct_multi_horizon',
            'key_horizons': self.KEY_HORIZONS,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'feature_names': self.feature_names
        }

        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save models
        for feat_name in self.feature_names:
            for horizon, model in self.models[feat_name].items():
                save_path = os.path.join(save_dir, f'xgb_{feat_name}_h{horizon}.pkl')
                with open(save_path, 'wb') as f:
                    pickle.dump(model, f)

        logger.info(f"Saved XGBoost models to {save_dir}")

    def load(self, save_dir: str):
        """Load models from disk."""
        # Load metadata
        with open(os.path.join(save_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        self.n_estimators = metadata['n_estimators']
        self.max_depth = metadata['max_depth']
        self.learning_rate = metadata['learning_rate']
        self.feature_names = metadata['feature_names']

        # Load models
        for feat_name in self.feature_names:
            self.models[feat_name] = {}
            for horizon in self.KEY_HORIZONS:
                load_path = os.path.join(save_dir, f'xgb_{feat_name}_h{horizon}.pkl')
                with open(load_path, 'rb') as f:
                    self.models[feat_name][horizon] = pickle.load(f)

        logger.info(f"Loaded XGBoost models from {save_dir}")


def run_xgboost_experiment(scenario: str = 'A', forecast_horizon: int = 2880):
    """
    Run complete XGBoost experiment for a scenario.

    Args:
        scenario: 'A', 'B', or 'C'
        forecast_horizon: Number of timesteps to forecast
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running XGBoost experiment - Scenario {scenario}")
    logger.info(f"{'='*80}\n")

    # Load data - but handle small datasets gracefully
    logger.info("Loading data...")
    
    try:
        data = load_data_for_scenario(scenario, forecast_horizon=forecast_horizon, for_ml_model=True)
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
        feature_names = data['feature_columns']
        
    except (ValueError, Exception) as e:
        # Catch any error from lag feature creation (including "negative dimensions")
        error_msg = str(e)
        if "negative dimensions" in error_msg or "Dataset too small" in error_msg or "Not enough data" in error_msg:
            logger.warning(f"Validation/test set too small for lag features: {e}")
            logger.warning("Attempting to load with adjusted strategy...")
            
            # Load raw DataFrames instead
            data_raw = load_data_for_scenario(scenario, forecast_horizon=forecast_horizon, for_ml_model=False)
            
            train_df = data_raw['train']
            val_df = data_raw['val']
            test_df = data_raw['test']
            feature_names = data_raw['feature_columns']
            
            logger.info(f"Raw data sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            
            from models.utils.data_loader import TimeSeriesDataLoader
            loader = TimeSeriesDataLoader()
            
            # First, determine the max_lag based on the SMALLEST dataset we need to use
            # This ensures all datasets use the same feature set
            min_dataset_size = min(len(train_df), len(val_df), len(test_df))
            available_for_lags = min_dataset_size - forecast_horizon - 1
            
            # Determine appropriate max_lag for all datasets
            if available_for_lags >= 2880:
                force_max_lag = 2880
            elif available_for_lags >= 720:
                force_max_lag = 720
            elif available_for_lags >= 360:
                force_max_lag = 360
            else:
                force_max_lag = 120
                
            logger.info(f"Using consistent max_lag={force_max_lag} for all datasets (based on min size {min_dataset_size})")
            
            # Create features with consistent lag structure
            X_train, y_train = loader.create_lag_features(train_df, forecast_horizon, force_max_lag=force_max_lag)
            
            # Use val for validation only if large enough
            try:
                X_val, y_val = loader.create_lag_features(val_df, forecast_horizon, force_max_lag=force_max_lag)
                logger.info(f"Created validation features: {X_val.shape}")
            except (ValueError, Exception) as val_e:
                logger.warning(f"Failed to create validation features: {val_e}")
                X_val, y_val = None, None
            
            # Use test for testing only if large enough
            try:
                X_test, y_test = loader.create_lag_features(test_df, forecast_horizon, force_max_lag=force_max_lag)
                logger.info(f"Created test features: {X_test.shape}")
            except (ValueError, Exception) as test_e:
                logger.warning(f"Failed to create test features: {test_e}")
                X_test, y_test = None, None
        else:
            raise

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    if X_val is not None:
        logger.info(f"X_val shape: {X_val.shape}")
        logger.info(f"y_val shape: {y_val.shape}")
    if X_test is not None:
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"y_test shape: {y_test.shape}")

    # Initialize and fit model with early stopping
    logger.info("\n--- Training ---")
    forecaster = XGBoostForecaster(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        n_jobs=-1,
        early_stopping_rounds=20
    )
    
    # Fit with optional validation
    if X_val is not None:
        forecaster.fit(X_train, y_train, feature_names, X_val, y_val)
    else:
        forecaster.fit(X_train, y_train, feature_names)

    # Save model
    save_dir = f"weights/xgboost_scenario_{scenario}"
    forecaster.save(save_dir)

    # Get raw training values for MASE calculation
    y_train_raw = y_train[:, 0, :]  # [n_samples, n_features] - just first timestep as proxy

    # Evaluate on validation set (if available)
    val_results = None
    y_val_pred = None
    if X_val is not None:
        logger.info("\n--- Validation ---")
        
        # Sample validation set for fair comparison with statistical models
        if len(X_val) > MAX_EVAL_SAMPLES:
            sample_indices = np.linspace(0, len(X_val) - 1, MAX_EVAL_SAMPLES, dtype=int)
            X_val_sample = X_val[sample_indices]
            y_val_sample = y_val[sample_indices]
            logger.info(f"Sampled {MAX_EVAL_SAMPLES} from {len(X_val)} validation samples")
        else:
            X_val_sample = X_val
            y_val_sample = y_val
        
        y_val_pred = forecaster.predict(X_val_sample, forecast_horizon)

        val_results = evaluate_forecast(
            y_true=y_val_sample,
            y_pred=y_val_pred,
            y_train=y_train_raw,
            feature_names=feature_names,
            horizons=[1, 120, 720, 1440, 2880]
        )

    # Evaluate on test set (if available)
    test_results = None
    y_test_pred = None
    y_test_sample = None
    if X_test is not None:
        logger.info("\n--- Testing ---")
        
        # Sample test set for fair comparison with statistical models
        if len(X_test) > MAX_EVAL_SAMPLES:
            sample_indices = np.linspace(0, len(X_test) - 1, MAX_EVAL_SAMPLES, dtype=int)
            X_test_sample = X_test[sample_indices]
            y_test_sample = y_test[sample_indices]
            logger.info(f"Sampled {MAX_EVAL_SAMPLES} from {len(X_test)} test samples")
        else:
            X_test_sample = X_test
            y_test_sample = y_test
        
        y_test_pred = forecaster.predict(X_test_sample, forecast_horizon)

        test_results = evaluate_forecast(
            y_true=y_test_sample,
            y_pred=y_test_pred,
            y_train=y_train_raw,
            feature_names=feature_names,
            horizons=[1, 120, 720, 1440, 2880]
        )
        
        # Generate forecast plots
        logger.info("\n--- Generating Forecast Plots ---")
        generate_all_forecast_plots(
            y_true=y_test_sample,
            y_pred=y_test_pred,
            feature_names=feature_names,
            model_name="XGBoost",
            scenario=scenario,
            save_dir="plots",
            n_sample_plots=3
        )

    # Save results (with type conversion for JSON)
    results = {
        'model': 'XGBoost',
        'scenario': scenario,
        'forecast_horizon': forecast_horizon,
        'approach': 'direct_multi_horizon',
        'key_horizons': XGBoostForecaster.KEY_HORIZONS,
        'n_estimators': forecaster.n_estimators,
        'max_depth': forecaster.max_depth,
        'learning_rate': forecaster.learning_rate,
    }
    
    if val_results is not None:
        results['validation'] = convert_to_native_types({
            'aggregate': val_results['aggregate'],
            'per_horizon': val_results['per_horizon'],
            'per_feature': val_results['per_feature']
        })
    
    if test_results is not None:
        results['test'] = convert_to_native_types({
            'aggregate': test_results['aggregate'],
            'per_horizon': test_results['per_horizon'],
            'per_feature': test_results['per_feature']
        })

    # Save results JSON
    os.makedirs('results', exist_ok=True)
    results_path = f'results/xgboost_scenario_{scenario}_results.json'

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
        logger.info("SUMMARY - Validation Set Performance (no test set available)")
        logger.info(f"{'='*80}")
        logger.info(f"Aggregate RMSE: {val_results['aggregate']['rmse']:.4f}")
        logger.info(f"Aggregate MAE:  {val_results['aggregate']['mae']:.4f}")
        logger.info(f"Aggregate R2:   {val_results['aggregate']['r2']:.4f}")
        logger.info(f"Aggregate MAPE: {val_results['aggregate']['mape']:.2f}%")
        logger.info(f"Aggregate MASE: {val_results['aggregate']['mase']:.4f}")

        logger.info("\nPer-feature RMSE:")
        for feat, metrics in val_results['per_feature'].items():
            logger.info(f"  {feat}: {metrics['rmse']:.4f}")

        logger.info(f"\n{'='*80}\n")

    return results


if __name__ == "__main__":
    import sys

    # Parse arguments: scenario
    scenario = sys.argv[1] if len(sys.argv) > 1 else 'A'

    run_xgboost_experiment(scenario=scenario, forecast_horizon=2880)
