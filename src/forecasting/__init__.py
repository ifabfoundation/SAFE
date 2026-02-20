"""
Forecasting Module
==================
Deep learning and statistical models for multivariate time series forecasting.
Includes CNN, LSTM, CNN-LSTM, TCN, Transformer architectures (Keras/PyTorch),
plus statistical models (VAR, ARIMA, Prophet) and gradient boosting (XGBoost, LightGBM).

Contributors: Lucia Gasperini, Giacomo Piergentili (Università di Bologna)
Coordinators: Orso Peruzzi, Benedetta Baldini

Modules:
- models: Definition of forecasting models (CNN, LSTM, TCN, Transformer)
- data_loader: Data reading, preprocessing, feature engineering
- train_optimize: Training, validation, hyperparameter optimization
- utils: Shared utility functions
- statistical_models/: VAR, ARIMA, Prophet, XGBoost, LightGBM, Random Forest

Note: Heavy dependencies (TensorFlow, PyTorch) are imported lazily.
      Use `from src.forecasting.models import ...` for model classes.
"""

__version__ = "1.0.0"

__all__ = [
    # === data_loader ===
    'DataConfig', 
    'load_csv', 
    'load_multiple_csv', 
    'get_numeric_columns',
    'handle_missing_values', 
    'downsample_data', 
    'standardize_data',
    'create_sequences', 
    'split_sequences', 
    'prepare_data_for_training',
    'make_supervised_features', 
    'identify_temperature_columns',
    'identify_vibration_columns', 
    'identify_current_columns', 
    'get_time_series',
    
    # === models ===
    'ModelConfig', 
    'build_cnn_model', 
    'build_cnn_lstm_model', 
    'build_lstm_model',
    'build_tcn_model', 
    'build_tcn_lstm_model', 
    'build_deepant_model',
    'build_seq2seq_model', 
    'build_rnn_micro_model', 
    'build_transformer_model',
    'build_transformer_wrapper',
    'TransformerForecaster', 
    'SequenceDataset', 
    'get_model', 
    'get_available_models',
    'get_all_available_models',
    'is_tcn_available',
    # Boosting
    'make_supervised',
    'train_boosting',
    'load_boosting_model',
    'predict_boosting',
    
    # === train_optimize ===
    'TrainingConfig', 
    'regression_report', 
    'regression_report_df', 
    'print_metrics',
    'train_keras_model', 
    'train_transformer', 
    'KerasGridSearch',
    'run_grid_search',
    'create_optuna_objective',
    'run_optuna_optimization', 
    'set_random_seeds', 
    'save_training_results',
    
    # === anomaly_detection ===
    # Config
    'AnomalyConfig',
    'ForecastAnomalyConfig',
    # Statistical methods
    'compute_zscore_anomalies', 
    'compute_mad_anomalies',
    'compute_peak_anomalies', 
    'compute_trend_anomalies', 
    # CUSUM
    'compute_cusum',
    'compute_ewma_cusum', 
    'compute_cusum_multi_params',
    'select_best_cusum_params',
    'compute_ewma_control_limits',
    # ML-based
    'compute_isolation_forest_anomalies', 
    'compute_lof_anomalies',
    # Forecast-based
    'compute_forecast_residuals', 
    'detect_anomalies_from_residuals',
    # Analysis
    'analyze_signal_anomalies', 
    'analyze_dataframe_anomalies', 
    # Thresholds
    'apply_thresholds',
    'save_thresholds',
    'load_thresholds',
    # Calibration workflow
    'calibrate_thresholds_from_residuals',
    'validate_thresholds_on_test',
    'apply_thresholds_to_residuals',
    'create_anomaly_dataframe',
    'combine_thresholds_weighted',
    # Health & classification
    'compute_health_index', 
    'classify_anomaly_type',
    'get_anomaly_summary',
    # Plotting
    'plot_anomalies_with_colors',
    'plot_all_features_anomalies',
    'plot_anomaly_summary',
    
    # === utils ===
    'configure_gpu', 
    'get_device', 
    'ensure_dir', 
    'save_json', 
    'load_json',
    'print_separator', 
    'format_time', 
    'get_memory_usage',
    'estimate_memory_for_sequences', 
    'validate_dataframe', 
    'chunk_array',
    'moving_average', 
    'normalize_array', 
    'safe_divide', 
    'ProgressTracker',
]