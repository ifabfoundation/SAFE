"""
Data loading and preprocessing utilities for time series forecasting.

Handles:
- Loading tampieri, caratterizzazione, fatica datasets
- Resampling to 30s intervals
- Column alignment and cleaning
- Train/val/test splitting for 3 scenarios
- StandardScaler normalization (fit on train, transform on val/test)
- Lag feature creation for ML models (after split to avoid leakage)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimeSeriesDataLoader:
    """Load and preprocess time series data for forecasting."""

    def __init__(self, data_dir: str = "data/final"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.feature_columns = ['acc_p3', 'acc_p4', 'temp_p3', 'temp_p4']

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load all three datasets."""
        logger.info("Loading datasets...")

        # Load tampieri
        tampieri = pd.read_csv(f"{self.data_dir}/tampieri_continuous.csv")
        tampieri['timestamp'] = pd.to_datetime(tampieri['timestamp'])
        tampieri = tampieri.set_index('timestamp')

        # Load caratterizzazione
        caratterizzazione = pd.read_csv(f"{self.data_dir}/bonfiglioli_caratterizzazione_continuous.csv")
        caratterizzazione['datetime'] = pd.to_datetime(caratterizzazione['datetime'])
        caratterizzazione = caratterizzazione.set_index('datetime')

        # Load fatica
        fatica = pd.read_csv(f"{self.data_dir}/bonfiglioli_fatica_continuous.csv")
        fatica['datetime'] = pd.to_datetime(fatica['datetime'])
        fatica = fatica.set_index('datetime')

        logger.info(f"Tampieri: {len(tampieri)} rows")
        logger.info(f"Caratterizzazione: {len(caratterizzazione)} rows")
        logger.info(f"Fatica: {len(fatica)} rows")

        return {
            'tampieri': tampieri,
            'caratterizzazione': caratterizzazione,
            'fatica': fatica
        }

    def preprocess_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Preprocess a single dataset:
        1. Resample to 30s intervals (mean aggregation)
        2. Drop 'current' column
        3. Align column names to [acc_p3, acc_p4, temp_p3, temp_p4]
        4. Handle NaNs (ffill, bfill, dropna)
        """
        logger.info(f"Preprocessing {dataset_name}...")

        # Resample to 30s intervals with mean aggregation
        df_resampled = df.resample('30s').mean()
        logger.info(f"  After resampling: {len(df_resampled)} rows")

        # Drop current column
        if 'current' in df_resampled.columns:
            df_resampled = df_resampled.drop(columns=['current'])

        # Ensure columns are in correct order
        df_resampled = df_resampled[self.feature_columns]

        # Handle NaNs
        df_resampled = df_resampled.fillna(method='ffill')  # Forward fill
        df_resampled = df_resampled.fillna(method='bfill')  # Backward fill
        df_resampled = df_resampled.dropna()  # Drop any remaining NaNs

        logger.info(f"  After cleaning: {len(df_resampled)} rows")
        logger.info(f"  Columns: {list(df_resampled.columns)}")

        return df_resampled

    def prepare_scenarios(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Prepare data for all 3 scenarios.

        Returns:
            Dict with keys 'A', 'B', 'C', each containing:
            {
                'train': DataFrame,
                'val': DataFrame,
                'test': DataFrame,
                'scaler': fitted StandardScaler
            }
        """
        # Load and preprocess all datasets
        raw_data = self.load_raw_data()

        tampieri = self.preprocess_dataset(raw_data['tampieri'], 'tampieri')
        caratterizzazione = self.preprocess_dataset(raw_data['caratterizzazione'], 'caratterizzazione')
        fatica = self.preprocess_dataset(raw_data['fatica'], 'fatica')

        scenarios = {}

        # Scenario A: Train 70% / Val 10% / Test 20% on tampieri only
        logger.info("\n=== Scenario A: Tampieri only ===")
        n_train = int(len(tampieri) * 0.7)
        n_val = int(len(tampieri) * 0.1)

        train_a = tampieri.iloc[:n_train].copy()
        val_a = tampieri.iloc[n_train:n_train+n_val].copy()
        test_a = tampieri.iloc[n_train+n_val:].copy()

        # Fit scaler on train, transform all
        scaler_a = StandardScaler()
        train_a[self.feature_columns] = scaler_a.fit_transform(train_a[self.feature_columns])
        val_a[self.feature_columns] = scaler_a.transform(val_a[self.feature_columns])
        test_a[self.feature_columns] = scaler_a.transform(test_a[self.feature_columns])

        scenarios['A'] = {
            'train': train_a,
            'val': val_a,
            'test': test_a,
            'scaler': scaler_a
        }

        logger.info(f"Train: {len(train_a)} rows, Val: {len(val_a)} rows, Test: {len(test_a)} rows")

        # Scenario B: Train on 100% caratterizzazione, Val on fatica (first 191k rows), Test on 100% tampieri
        logger.info("\n=== Scenario B: Train caratterizzazione, Val fatica (truncated), Test tampieri ===")

        train_b = caratterizzazione.copy()
        val_b = fatica.iloc[:len(caratterizzazione)].copy()  # Truncate fatica to match caratterizzazione size
        test_b = tampieri.copy()

        # Fit scaler on train (caratterizzazione), transform all
        scaler_b = StandardScaler()
        train_b[self.feature_columns] = scaler_b.fit_transform(train_b[self.feature_columns])
        val_b[self.feature_columns] = scaler_b.transform(val_b[self.feature_columns])
        test_b[self.feature_columns] = scaler_b.transform(test_b[self.feature_columns])

        scenarios['B'] = {
            'train': train_b,
            'val': val_b,
            'test': test_b,
            'scaler': scaler_b
        }

        logger.info(f"Train: {len(train_b)} rows, Val: {len(val_b)} rows, Test: {len(test_b)} rows")

        # Scenario C: Train 80% / Val 20% on caratterizzazione, Test on 100% tampieri
        logger.info("\n=== Scenario C: Train/Val on caratterizzazione, Test tampieri ===")

        n_train_c = int(len(caratterizzazione) * 0.8)

        train_c = caratterizzazione.iloc[:n_train_c].copy()
        val_c = caratterizzazione.iloc[n_train_c:].copy()
        test_c = tampieri.copy()

        # Fit scaler on train, transform all
        scaler_c = StandardScaler()
        train_c[self.feature_columns] = scaler_c.fit_transform(train_c[self.feature_columns])
        val_c[self.feature_columns] = scaler_c.transform(val_c[self.feature_columns])
        test_c[self.feature_columns] = scaler_c.transform(test_c[self.feature_columns])

        scenarios['C'] = {
            'train': train_c,
            'val': val_c,
            'test': test_c,
            'scaler': scaler_c
        }

        logger.info(f"Train: {len(train_c)} rows, Val: {len(val_c)} rows, Test: {len(test_c)} rows")

        return scenarios

    def create_lag_features(self,
                          df: pd.DataFrame,
                          forecast_horizon: int = 2880,
                          force_max_lag: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create lag features for ML models.

        Creates features from:
        - Recent lags: 1, 2, 4, 8, 16, 32, 64
        - Hourly lags: 120, 240, 360, 480, 600, 720
        - Daily lags: 2880 (or less if data is limited)
        - Rolling statistics: mean/std over adaptive windows

        Args:
            df: DataFrame with feature columns (already normalized)
            forecast_horizon: Number of timesteps to forecast (default 2880 = 24h)
            force_max_lag: If specified, force this max lag configuration (for consistency between train/test)

        Returns:
            X: Feature matrix [n_samples, n_features]
            y: Target matrix [n_samples, forecast_horizon, n_targets]
        """
        # Start with full lag structure
        recent_lags = [1, 2, 4, 8, 16, 32, 64]
        hourly_lags = [120, 240, 360, 480, 600, 720]
        daily_lags = [2880]
        rolling_windows = [120, 720, 2880]

        # Calculate available data budget
        available_for_lags = len(df) - forecast_horizon - 1  # Need at least 1 sample
        
        # If force_max_lag is specified, use that configuration
        if force_max_lag is not None:
            if force_max_lag >= 2880:
                pass  # Use full lags
            elif force_max_lag >= 720:
                daily_lags = []
                rolling_windows = [120, 720]
            elif force_max_lag >= 360:
                daily_lags = []
                hourly_lags = [120, 240, 360]
                rolling_windows = [120]
            else:
                daily_lags = []
                hourly_lags = [120]
                rolling_windows = [60]
        else:
            # Progressively reduce lag requirements if dataset is too small
            if available_for_lags < 2880:
                # Remove daily lags and reduce rolling windows
                daily_lags = []
                rolling_windows = [120, 720]
                logger.warning(f"Dataset too small for daily lags. Using hourly lags only (max_lag={max(hourly_lags)})")
                
                if available_for_lags < 720:
                    # Also reduce hourly lags
                    hourly_lags = [120, 240, 360]
                    rolling_windows = [120]
                    logger.warning(f"Dataset limited. Using reduced hourly lags (max_lag={max(hourly_lags)})")
                    
                    if available_for_lags < 360:
                        # Minimal lags only
                        hourly_lags = [120]
                        rolling_windows = [60]
                        logger.warning(f"Very small dataset. Using minimal lags (max_lag={max(hourly_lags)})")

        all_lags = recent_lags + hourly_lags + daily_lags
        max_lag = max(all_lags)
        
        n_samples = len(df) - max_lag - forecast_horizon
        
        if n_samples < 1:
            raise ValueError(f"Dataset too small for ML model. Need at least {max_lag + forecast_horizon + 1} rows, got {len(df)}")
        
        n_lag_features = len(all_lags) + len(rolling_windows) * 2  # mean + std
        n_total_features = n_lag_features * len(self.feature_columns)

        X = np.zeros((n_samples, n_total_features))
        y = np.zeros((n_samples, forecast_horizon, len(self.feature_columns)))

        logger.info(f"Creating lag features: {n_samples} samples, {n_total_features} features (max_lag={max_lag})")

        for i in range(n_samples):
            feature_idx = 0
            actual_idx = max_lag + i

            for col_idx, col in enumerate(self.feature_columns):
                series = df[col].values

                # Lag features
                for lag in all_lags:
                    X[i, feature_idx] = series[actual_idx - lag]
                    feature_idx += 1

                # Rolling statistics
                for window in rolling_windows:
                    window_data = series[actual_idx - window:actual_idx]
                    X[i, feature_idx] = np.mean(window_data)
                    feature_idx += 1
                    X[i, feature_idx] = np.std(window_data)
                    feature_idx += 1

                # Target: full forecast horizon
                y[i, :, col_idx] = series[actual_idx + 1:actual_idx + 1 + forecast_horizon]

        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Store the max_lag used for this feature set (useful for consistency)
        self._last_max_lag = max_lag

        return X, y


def load_data_for_scenario(scenario: str,
                           forecast_horizon: int = 2880,
                           for_ml_model: bool = False) -> Dict:
    """
    Convenience function to load data for a specific scenario.

    Args:
        scenario: 'A', 'B', or 'C'
        forecast_horizon: Number of timesteps to forecast (default 2880 = 24h)
        for_ml_model: If True, create lag features for ML models

    Returns:
        Dict containing train/val/test data (either DataFrames or numpy arrays)
    """
    loader = TimeSeriesDataLoader()
    scenarios = loader.prepare_scenarios()

    if scenario not in scenarios:
        raise ValueError(f"Invalid scenario: {scenario}. Must be 'A', 'B', or 'C'")

    scenario_data = scenarios[scenario]

    if for_ml_model:
        # Determine appropriate max_lag based on SMALLEST dataset to ensure consistency
        min_dataset_size = min(len(scenario_data['train']), 
                               len(scenario_data['val']), 
                               len(scenario_data['test']))
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
            
        logger.info(f"Using consistent max_lag={force_max_lag} for all datasets (min size: {min_dataset_size})")
        
        # Create lag features for ML models with consistent lag structure
        X_train, y_train = loader.create_lag_features(scenario_data['train'], forecast_horizon, force_max_lag=force_max_lag)
        X_val, y_val = loader.create_lag_features(scenario_data['val'], forecast_horizon, force_max_lag=force_max_lag)
        X_test, y_test = loader.create_lag_features(scenario_data['test'], forecast_horizon, force_max_lag=force_max_lag)

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scenario_data['scaler'],
            'feature_columns': loader.feature_columns,
            'max_lag': force_max_lag  # Include for reference
        }
    else:
        # Return raw DataFrames for statistical models
        return {
            'train': scenario_data['train'],
            'val': scenario_data['val'],
            'test': scenario_data['test'],
            'scaler': scenario_data['scaler'],
            'feature_columns': loader.feature_columns
        }


if __name__ == "__main__":
    # Test data loading
    logger.info("Testing data loader...")

    # Test statistical model data
    data_stat = load_data_for_scenario('A', for_ml_model=False)
    logger.info(f"\nStatistical model data:")
    logger.info(f"  Train shape: {data_stat['train'].shape}")
    logger.info(f"  Val shape: {data_stat['val'].shape}")
    logger.info(f"  Test shape: {data_stat['test'].shape}")

    # Test ML model data
    data_ml = load_data_for_scenario('A', for_ml_model=True)
    logger.info(f"\nML model data:")
    logger.info(f"  X_train shape: {data_ml['X_train'].shape}")
    logger.info(f"  y_train shape: {data_ml['y_train'].shape}")
    logger.info(f"  X_val shape: {data_ml['X_val'].shape}")
    logger.info(f"  y_val shape: {data_ml['y_val'].shape}")
    logger.info(f"  X_test shape: {data_ml['X_test'].shape}")
    logger.info(f"  y_test shape: {data_ml['y_test'].shape}")
