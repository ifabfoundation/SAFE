"""
Utilities for forecasting models.

Provides:
- data_loader: Load and preprocess time series data, create lag features
- evaluation: Compute forecast metrics (RMSE, MAE, R2, MAPE, MASE)
- forecast_plots: Visualize forecast vs actual time series
- visualization: Additional visualization utilities
"""

from models.utils.data_loader import (
    load_data_for_scenario,
    TimeSeriesDataLoader
)

from models.utils.evaluation import (
    evaluate_forecast,
    convert_to_native_types,
    compute_mape,
    compute_mase,
    compute_metrics
)

from models.utils.forecast_plots import (
    plot_forecast_vs_actual,
    plot_forecast_multiple_samples,
    plot_forecast_summary,
    plot_per_horizon_rmse,
    generate_all_forecast_plots
)

__all__ = [
    # Data loading
    'load_data_for_scenario',
    'TimeSeriesDataLoader',
    
    # Evaluation
    'evaluate_forecast',
    'convert_to_native_types',
    'compute_mape',
    'compute_mase',
    'compute_metrics',
    
    # Plotting
    'plot_forecast_vs_actual',
    'plot_forecast_multiple_samples',
    'plot_forecast_summary',
    'plot_per_horizon_rmse',
    'generate_all_forecast_plots',
]
