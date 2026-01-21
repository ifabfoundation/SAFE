"""
Anomaly Detection Module
========================
Anomaly detection based on forecast.
Comparison between observed and predicted values.
Use of residuals and statistical methods.

Implemented methods:
- Z-score (rolling window)
- MAD (Median Absolute Deviation)
- Peak Detection
- Trend Analysis (slope)
- CUSUM (Cumulative Sum Control Chart)
- EWMA (Exponentially Weighted Moving Average)
- Isolation Forest
- Local Outlier Factor
- Health Index
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Optional import for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""
    z_window: int = 5000
    z_threshold: float = 3.0
    mad_threshold: float = 3.5
    trend_window: int = 20000
    trend_sigma: float = 3.0
    ewma_alpha: float = 0.01
    cusum_baseline_samples: int = 50000
    peak_height_sigma: float = 2.0
    peak_prominence_factor: float = 0.5
    warning_threshold: int = 1
    danger_threshold: int = 3


def compute_zscore_anomalies(
    signal: np.ndarray,
    window: int = 5000,
    threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect anomalies using Z-score on rolling window."""
    sig = pd.Series(signal)
    roll_mean = sig.rolling(window).mean()
    roll_std = sig.rolling(window).std()
    zscore = (sig - roll_mean) / roll_std
    anomalies = (zscore.abs() > threshold).fillna(False).values
    return zscore.values, anomalies


def compute_mad_anomalies(
    signal: np.ndarray,
    threshold: float = 3.5
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Detect anomalies using MAD (Median Absolute Deviation)."""
    sig = pd.Series(signal)
    median = sig.median()
    mad = (sig - median).abs().median()
    
    if mad > 0:
        mad_score = 0.6745 * (sig - median).abs() / mad
        anomalies = (mad_score > threshold).fillna(False).values
    else:
        mad_score = pd.Series(0.0, index=sig.index)
        anomalies = np.zeros(len(signal), dtype=bool)
    
    return mad_score.values, median, mad, anomalies


def compute_peak_anomalies(
    signal: np.ndarray,
    height_sigma: float = 2.0,
    prominence_factor: float = 0.5
) -> Tuple[np.ndarray, Dict]:
    """Detect anomalous peaks using scipy.signal.find_peaks."""
    sig_mean = np.nanmean(signal)
    sig_std = np.nanstd(signal)
    
    height_threshold = sig_mean + height_sigma * sig_std
    prominence_threshold = sig_std * prominence_factor
    
    peaks, props = find_peaks(signal, prominence=prominence_threshold, height=height_threshold)
    
    peaks_mask = np.zeros(len(signal), dtype=bool)
    peaks_mask[peaks] = True
    
    info = {
        'height_threshold': height_threshold,
        'prominence_threshold': prominence_threshold,
        'n_peaks': len(peaks),
        'peak_indices': peaks
    }
    return peaks_mask, info


def compute_trend_anomalies(
    signal: np.ndarray,
    window: int = 20000,
    sigma: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], np.ndarray]:
    """Detect trend anomalies using slope of the rolling mean.
    
    Detects both abnormally INCREASING and DECREASING trends.
    A trend is anomalous if slope > upper_threshold OR slope < lower_threshold.
    
    Returns:
        roll_avg: Rolling average of the signal
        slope: First derivative of rolling average
        thresholds: Dict with 'upper', 'lower', 'mean', 'std' keys
        anomalies: Boolean array where True = anomalous trend
    """
    sig = pd.Series(signal)
    roll_avg = sig.rolling(window).mean()
    slope = roll_avg.diff()
    
    slope_mean = slope.mean()
    slope_std = slope.std()
    
    # Detect both positive AND negative anomalous trends
    threshold_upper = slope_mean + sigma * slope_std
    threshold_lower = slope_mean - sigma * slope_std
    
    anomalies = ((slope > threshold_upper) | (slope < threshold_lower)).fillna(False).values
    
    thresholds_dict = {
        'upper': float(threshold_upper),
        'lower': float(threshold_lower),
        'mean': float(slope_mean),
        'std': float(slope_std)
    }
    return roll_avg.values, slope.values, thresholds_dict, anomalies


def compute_cusum(
    signal: np.ndarray,
    mu0: float,  
    h: float,
    k: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute CUSUM (Cumulative Sum Control Chart)."""
    n = len(signal)
    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)
    alarms = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        cusum_pos[i] = max(0, cusum_pos[i-1] + (signal[i] - mu0 - k))
        cusum_neg[i] = max(0, cusum_neg[i-1] - (signal[i] - mu0 + k))
        alarms[i] = (cusum_pos[i] > h) or (cusum_neg[i] > h)
    
    return cusum_pos, cusum_neg, alarms


def compute_ewma_cusum(
    signal: np.ndarray,
    alpha: float = 0.01,
    baseline_samples: int = 50000,
    h_factor: float = 4.0,
    k_factor: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Compute CUSUM on EWMA of the signal."""
    sig = pd.Series(signal)
    ewma = sig.ewm(alpha=alpha).mean()
    
    n_baseline = min(baseline_samples, len(ewma))
    mu0 = ewma.iloc[:n_baseline].mean()
    sigma0 = ewma.iloc[:n_baseline].std()
    
    h = h_factor * sigma0
    k = k_factor * sigma0
    
    cusum_pos, cusum_neg, alarms = compute_cusum(ewma.values, mu0, h, k)
    
    info = {'mu0': mu0, 'sigma0': sigma0, 'h': h, 'k': k, 'alpha': alpha}
    return ewma.values, cusum_pos, cusum_neg, alarms, info


# Default CUSUM parameter combinations to test
DEFAULT_CUSUM_COMBINATIONS = [
    {'h_factor': 4.0, 'k_factor': 0.5, 'name': 'h=4_k=0.5'},
    {'h_factor': 5.0, 'k_factor': 0.5, 'name': 'h=5_k=0.5'},
    {'h_factor': 8.0, 'k_factor': 0.25, 'name': 'h=8_k=0.25'},
    {'h_factor': 2.5, 'k_factor': 1.0, 'name': 'h=2.5_k=1'},
]


def compute_cusum_multi_params(
    signal: np.ndarray,
    alpha: float = 0.01,
    baseline_samples: int = 50000,
    combinations: Optional[List[Dict]] = None,
    verbose: bool = False
) -> Tuple[Dict, str, Dict]:
    """
    Compute CUSUM with multiple parameter combinations and select the best.
    
    The best combination is selected based on:
    1. For baseline/characterization data (expected few anomalies): 
       Choose the one with lowest alarm rate (most conservative)
    2. The selection criteria can be customized
    
    Args:
        signal: Input signal
        alpha: EWMA smoothing factor
        baseline_samples: Number of samples for baseline statistics (N0)
        combinations: List of parameter combinations to test.
                     Each dict must have 'h_factor', 'k_factor', 'name'.
                     If None, uses DEFAULT_CUSUM_COMBINATIONS.
        verbose: If True, print results for each combination
        
    Returns:
        Tuple of:
        - results_dict: Dictionary with results for each combination
        - best_combo_name: Name of the best combination
        - best_info: Info dict for the best combination
    """
    if combinations is None:
        combinations = DEFAULT_CUSUM_COMBINATIONS
    
    sig = pd.Series(signal)
    ewma = sig.ewm(alpha=alpha).mean()
    
    # Calculate baseline statistics
    n_baseline = min(baseline_samples, len(ewma))
    mu0 = float(ewma.iloc[:n_baseline].mean())
    sigma0 = float(ewma.iloc[:n_baseline].std())
    
    if sigma0 == 0:
        sigma0 = 1e-10  # Avoid division by zero
    
    results = {}
    
    if verbose:
        print(f"  Baseline μ0 (EWMA): {mu0:.6f}, σ0: {sigma0:.6f}")
    
    for combo in combinations:
        h_factor = combo['h_factor']
        k_factor = combo['k_factor']
        name = combo['name']
        
        h = h_factor * sigma0
        k = k_factor * sigma0
        
        cusum_pos, cusum_neg, alarms = compute_cusum(ewma.values, mu0, h, k)
        
        n_alarms = int(alarms.sum())
        alarm_rate = n_alarms / len(signal) * 100
        
        results[name] = {
            'h_factor': h_factor,
            'k_factor': k_factor,
            'h': float(h),
            'k': float(k),
            'n_alarms': n_alarms,
            'alarm_rate': alarm_rate,
            'cusum_pos': cusum_pos,
            'cusum_neg': cusum_neg,
            'alarms': alarms,
        }
        
        if verbose:
            print(f"    [{name}] Alarms: {n_alarms} ({alarm_rate:.2f}%) | h={h:.4f}, k={k:.4f}")
    
    # Select best combination
    # For baseline data: prefer lowest alarm rate (most conservative for clean data)
    # But avoid combinations with 0 alarms if others have some (could be too insensitive)
    best_combo_name = min(results.keys(), key=lambda x: results[x]['alarm_rate'])
    
    best_info = {
        'alpha': alpha,
        'N0': baseline_samples,
        'mu0': mu0,
        'sigma0': sigma0,
        'best_combo': best_combo_name,
        'h': results[best_combo_name]['h'],
        'k': results[best_combo_name]['k'],
        'h_factor': results[best_combo_name]['h_factor'],
        'k_factor': results[best_combo_name]['k_factor'],
        'all_combinations': [
            {
                'name': name,
                'h_factor': results[name]['h_factor'],
                'k_factor': results[name]['k_factor'],
                'h': results[name]['h'],
                'k': results[name]['k'],
                'n_alarms': results[name]['n_alarms'],
                'alarm_rate': results[name]['alarm_rate'],
            }
            for name in results
        ]
    }
    
    if verbose:
        print(f"  → Best combination: {best_combo_name}")
    
    return results, best_combo_name, best_info


def select_best_cusum_params(
    calib_results: Dict,
    test_results: Optional[Dict] = None,
    strategy: str = 'min_baseline_alarms'
) -> str:
    """
    Select the best CUSUM parameter combination based on calibration and test results.
    
    Strategies:
    - 'min_baseline_alarms': Choose combination with lowest alarms on baseline (conservative)
    - 'balanced': Balance between low baseline alarms and detection capability on test
    - 'max_sensitivity': Choose combination with highest sensitivity (more alarms)
    
    Args:
        calib_results: Results dict from compute_cusum_multi_params on calibration data
        test_results: Optional results dict from test/fatigue data
        strategy: Selection strategy
        
    Returns:
        Name of the best combination
    """
    if strategy == 'min_baseline_alarms':
        # Most conservative: lowest alarm rate on baseline
        return min(calib_results.keys(), key=lambda x: calib_results[x]['alarm_rate'])
    
    elif strategy == 'max_sensitivity':
        # Most sensitive: highest alarm rate
        return max(calib_results.keys(), key=lambda x: calib_results[x]['alarm_rate'])
    
    elif strategy == 'balanced' and test_results is not None:
        # Balance: low baseline alarms but reasonable test detection
        scores = {}
        for name in calib_results:
            if name in test_results:
                # Penalize high baseline alarms, reward test detection
                baseline_penalty = calib_results[name]['alarm_rate']
                test_bonus = test_results[name]['alarm_rate'] * 0.1  # Weight test detection less
                scores[name] = test_bonus - baseline_penalty
        return max(scores.keys(), key=lambda x: scores[x])
    
    else:
        # Default: min baseline alarms
        return min(calib_results.keys(), key=lambda x: calib_results[x]['alarm_rate'])


def compute_ewma_control_limits(
    signal: np.ndarray,
    lam: float = 0.2,
    L: float = 3.0
) -> np.ndarray:
    """Detect anomalies using EWMA Control Chart."""
    mu = np.mean(signal)
    sigma = np.std(signal)
    n = len(signal)
    
    mask = np.zeros(n, dtype=bool)
    ewma = 0
    
    for i in range(n):
        ewma = lam * signal[i] + (1 - lam) * ewma
        sigma_t = sigma * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam) ** (2 * (i + 1))))
        ucl = mu + L * sigma_t
        lcl = mu - L * sigma_t
        if ewma > ucl or ewma < lcl:
            mask[i] = True
    
    return mask


def compute_isolation_forest_anomalies(
    data: np.ndarray,
    contamination: float = 0.01,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect anomalies using Isolation Forest."""
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    predictions = iso_forest.fit_predict(data)
    anomalies = predictions == -1
    scores = -iso_forest.score_samples(data)
    return anomalies, scores


def compute_lof_anomalies(
    data: np.ndarray,
    n_neighbors: int = 40,
    contamination: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect anomalies using Local Outlier Factor."""
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)
    predictions = lof.fit_predict(data)
    anomalies = predictions == -1
    scores = -lof.negative_outlier_factor_
    return anomalies, scores


def compute_forecast_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute residuals between actual and predicted values."""
    return y_true - y_pred


def detect_anomalies_from_residuals(
    residuals: np.ndarray,
    method: str = 'zscore',
    threshold: float = 3.0,
    **kwargs
) -> Tuple[np.ndarray, Dict]:
    """Detect anomalies from forecast residuals."""
    if residuals.ndim == 1:
        residuals = residuals.reshape(-1, 1)
    
    n_samples, n_features = residuals.shape
    
    if method == 'zscore':
        window = kwargs.get('window', 100)
        anomalies = np.zeros((n_samples, n_features), dtype=bool)
        for j in range(n_features):
            _, anom = compute_zscore_anomalies(residuals[:, j], window, threshold)
            anomalies[:, j] = anom
        info = {'method': 'zscore', 'window': window, 'threshold': threshold}
    
    elif method == 'mad':
        anomalies = np.zeros((n_samples, n_features), dtype=bool)
        for j in range(n_features):
            _, _, _, anom = compute_mad_anomalies(residuals[:, j], threshold)
            anomalies[:, j] = anom
        info = {'method': 'mad', 'threshold': threshold}
    
    elif method == 'sigma':
        mean_res = np.mean(residuals, axis=0)
        std_res = np.std(residuals, axis=0)
        lower = mean_res - threshold * std_res
        upper = mean_res + threshold * std_res
        anomalies = (residuals < lower) | (residuals > upper)
        info = {'method': 'sigma', 'mean': mean_res, 'std': std_res, 'k': threshold}
    
    elif method == 'isolation_forest':
        contamination = kwargs.get('contamination', 0.01)
        anomalies_flat, scores = compute_isolation_forest_anomalies(residuals, contamination=contamination)
        anomalies = np.tile(anomalies_flat.reshape(-1, 1), (1, n_features))
        info = {'method': 'isolation_forest', 'contamination': contamination, 'scores': scores}
    
    elif method == 'lof':
        n_neighbors = kwargs.get('n_neighbors', 40)
        contamination = kwargs.get('contamination', 0.01)
        anomalies_flat, scores = compute_lof_anomalies(residuals, n_neighbors=n_neighbors, contamination=contamination)
        anomalies = np.tile(anomalies_flat.reshape(-1, 1), (1, n_features))
        info = {'method': 'lof', 'n_neighbors': n_neighbors, 'scores': scores}
    
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return anomalies, info


def analyze_signal_anomalies(
    signal: np.ndarray,
    signal_name: str,
    config: Optional[AnomalyConfig] = None
) -> Dict:
    """Complete anomaly analysis for a single signal."""
    if config is None:
        config = AnomalyConfig()
    
    results = {'name': signal_name}
    
    zscore, z_anom = compute_zscore_anomalies(signal, config.z_window, config.z_threshold)
    results['zscore'] = zscore
    results['z_anomalies'] = z_anom
    
    mad_score, median, mad, mad_anom = compute_mad_anomalies(signal, config.mad_threshold)
    results['mad_score'] = mad_score
    results['mad_anomalies'] = mad_anom
    results['mad_params'] = {'median': median, 'mad': mad}
    
    peaks_mask, peaks_info = compute_peak_anomalies(signal, config.peak_height_sigma, config.peak_prominence_factor)
    results['peaks'] = peaks_mask
    results['peaks_info'] = peaks_info
    
    roll_avg, slope, slope_thresholds, trend_anom = compute_trend_anomalies(signal, config.trend_window, config.trend_sigma)
    results['trend'] = roll_avg
    results['slope'] = slope
    results['trend_anomalies'] = trend_anom
    results['slope_thresholds'] = slope_thresholds  # Dict with 'upper', 'lower', 'mean', 'std'
    
    ewma, cusum_pos, cusum_neg, cusum_alarm, cusum_info = compute_ewma_cusum(
        signal, config.ewma_alpha, config.cusum_baseline_samples
    )
    results['ewma'] = ewma
    results['cusum_pos'] = cusum_pos
    results['cusum_neg'] = cusum_neg
    results['cusum_alarm'] = cusum_alarm
    results['cusum_info'] = cusum_info
    
    any_anomaly = z_anom | mad_anom | peaks_mask | trend_anom | cusum_alarm
    results['any_anomaly'] = any_anomaly
    
    return results


def analyze_dataframe_anomalies(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    config: Optional[AnomalyConfig] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """Analyze anomalies for all numeric columns of a DataFrame."""
    if config is None:
        config = AnomalyConfig()
    
    if numeric_cols is None:
        exclude = ['time', 'anomaly', 'anomaly_score', 'source_file', 'dataset_type']
        numeric_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64', 'float32']]
    
    df_result = df.copy()
    thresholds = {'meta': config.__dict__, 'signals': {}}
    
    for col in numeric_cols:
        if verbose:
            print(f"Analysis: {col}")
        
        signal = df[col].values.astype(float)
        results = analyze_signal_anomalies(signal, col, config)
        
        df_result[f'{col}_z_anom'] = results['z_anomalies'].astype(int)
        df_result[f'{col}_mad_anom'] = results['mad_anomalies'].astype(int)
        df_result[f'{col}_peaks'] = results['peaks'].astype(int)
        df_result[f'{col}_deg_anom'] = results['trend_anomalies'].astype(int)
        df_result[f'{col}_cusum_alarm'] = results['cusum_alarm'].astype(int)
        df_result[f'{col}_any_anom'] = results['any_anomaly'].astype(int)
        
        thresholds['signals'][col] = {
            'trend': results['slope_thresholds'],  # Dict with 'upper', 'lower', 'mean', 'std'
            'peaks': results['peaks_info'],
            'z_mad': {'z_k': config.z_threshold, 'mad_k': config.mad_threshold,
                      'median': float(results['mad_params']['median']),
                      'mad': float(results['mad_params']['mad'])},
            'cusum': results['cusum_info']
        }
    
    return df_result, thresholds


def apply_thresholds(
    df: pd.DataFrame,
    thresholds: Dict,
    numeric_cols: List[str],
    config: Optional[AnomalyConfig] = None
) -> pd.DataFrame:
    """Apply pre-calculated thresholds to a new dataset."""
    if config is None:
        config = AnomalyConfig()
    
    df_result = df.copy()
    
    for col in numeric_cols:
        if col not in thresholds.get('signals', {}):
            continue
        
        signal = df[col].values.astype(float)
        info = thresholds['signals'][col]
        
        # Trend (bidirectional: both increasing and decreasing anomalies)
        if 'trend' in info:
            roll_avg = pd.Series(signal).rolling(config.trend_window).mean()
            slope = roll_avg.diff()
            trend_info = info['trend']
            # Support both old format (slope_threshold) and new format (upper/lower)
            if 'upper' in trend_info and 'lower' in trend_info:
                is_anom = (slope > trend_info['upper']) | (slope < trend_info['lower'])
            else:
                # Fallback for old threshold format (only upper bound)
                is_anom = slope > trend_info.get('slope_threshold', trend_info.get('upper', 0))
            df_result[f'{col}_deg_anom'] = is_anom.fillna(False).astype(int)
        
        # Peaks
        if 'peaks' in info:
            peaks, _ = find_peaks(signal, height=info['peaks']['height_threshold'],
                                  prominence=info['peaks']['prominence_threshold'])
            df_result[f'{col}_peaks'] = 0
            if len(peaks) > 0:
                df_result.loc[df_result.index[peaks], f'{col}_peaks'] = 1
        
        # Z-score e MAD
        if 'z_mad' in info:
            _, z_anom = compute_zscore_anomalies(signal, config.z_window, info['z_mad']['z_k'])
            df_result[f'{col}_z_anom'] = z_anom.astype(int)
            
            mad = info['z_mad']['mad']
            if mad > 0:
                mad_score = 0.6745 * np.abs(signal - info['z_mad']['median']) / mad
                df_result[f'{col}_mad_anom'] = (mad_score > info['z_mad']['mad_k']).astype(int)
            else:
                df_result[f'{col}_mad_anom'] = 0
        
        # CUSUM
        if 'cusum' in info and info['cusum'].get('sigma0', 0) > 0:
            ci = info['cusum']
            ewma = pd.Series(signal).ewm(alpha=ci['alpha']).mean()
            _, _, alarms = compute_cusum(ewma.values, ci['mu0'], ci['h'], ci['k'])
            df_result[f'{col}_cusum_alarm'] = alarms.astype(int)
        else:
            df_result[f'{col}_cusum_alarm'] = 0
        
        # Flag complessivo
        flags = [f'{col}_deg_anom', f'{col}_peaks', f'{col}_z_anom', f'{col}_mad_anom', f'{col}_cusum_alarm']
        existing = [c for c in flags if c in df_result.columns]
        df_result[f'{col}_any_anom'] = (df_result[existing].sum(axis=1) > 0).astype(int)
    
    return df_result


def compute_health_index(
    df: pd.DataFrame,
    critical_cols: List[str],
    warning_threshold: int = 1,
    danger_threshold: int = 3
) -> pd.DataFrame:
    """Compute Health Index based on aggregated anomalies."""
    df_result = df.copy()
    health_score = np.zeros(len(df), dtype=int)
    
    for col in critical_cols:
        any_col = f"{col}_any_anom"
        if any_col in df.columns:
            health_score += df[any_col].values.astype(int)
    
    health_level = np.zeros(len(df), dtype=int)
    health_level[health_score >= warning_threshold] = 1
    health_level[health_score >= danger_threshold] = 2
    
    health_label = np.where(health_level == 2, "Danger",
                            np.where(health_level == 1, "Warning", "OK"))
    
    df_result['health_score'] = health_score
    df_result['health_level'] = health_level
    df_result['health_label'] = health_label
    
    return df_result


def classify_anomaly_type(
    row: pd.Series,
    current_col: str,
    startup_threshold: float
) -> str:
    """Classify anomalies as 'yellow' (startup) or 'red' (true anomaly)."""
    if current_col not in row.index:
        return 'red'
    val = row[current_col]
    if pd.isna(val):
        return 'red'
    return 'yellow' if val < startup_threshold else 'red'


def save_thresholds(thresholds: Dict, filepath: str):
    """Save thresholds in JSON format."""
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    thresholds_clean = json.loads(json.dumps(thresholds, default=convert))
    with open(filepath, 'w') as f:
        json.dump(thresholds_clean, f, indent=2)


def load_thresholds(filepath: str) -> Dict:
    """Load thresholds from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_anomaly_summary(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Generate anomaly summary by column."""
    summary = []
    
    for col in numeric_cols:
        row = {'column': col}
        for anom_type in ['z_anom', 'mad_anom', 'peaks', 'deg_anom', 'cusum_alarm', 'any_anom']:
            col_name = f'{col}_{anom_type}'
            if col_name in df.columns:
                n = df[col_name].sum()
                pct = n / len(df) * 100
                row[anom_type] = n
                row[f'{anom_type}_pct'] = round(pct, 2)
        summary.append(row)
    
    return pd.DataFrame(summary)


# =============================================================================
# Forecast-based Anomaly Detection with Threshold Calibration
# 
# Workflow:
#   1. CALIBRATE: Learn thresholds on LAB Normal Operation (caratterizzazione)
#   2. VALIDATE: Test thresholds on LAB Stress Test (fatica)
#   3. APPLY: Detect anomalies on REAL MACHINE (tampieri)
# =============================================================================

@dataclass
class ForecastAnomalyConfig:
    """Configuration for forecast-based anomaly detection."""
    # Residual thresholds (calibrated on characterization)
    sigma_warning: float = 2.0     # Yellow threshold (warning)
    sigma_danger: float = 3.0      # Red threshold (danger/alarm)
    
    # CUSUM parameters
    cusum_h_warning: float = 3.0   # CUSUM threshold for warning
    cusum_h_danger: float = 5.0    # CUSUM threshold for danger
    cusum_k: float = 0.5           # CUSUM slack parameter (multiplier of sigma)
    
    # EWMA parameters
    ewma_alpha: float = 0.1        # EWMA smoothing factor
    
    # Rolling window for statistics
    rolling_window: int = 100
    
    # Minimum samples for baseline calculation
    min_baseline_samples: int = 1000


def calibrate_thresholds_from_residuals(
    residuals: np.ndarray,
    feature_names: List[str],
    config: Optional[ForecastAnomalyConfig] = None,
    cusum_combinations: Optional[List[Dict]] = None,
    verbose: bool = True
) -> Dict:
    """
    Calibrate anomaly detection thresholds from forecast residuals.
    
    This function should be run on LAB CHARACTERIZATION data (normal operation)
    to learn the baseline statistical parameters. These thresholds define
    what "normal" looks like for the machine.
    
    Tests multiple CUSUM parameter combinations and selects the best one
    for each feature based on the characterization data.
    
    Args:
        residuals: Residuals array [n_samples, n_features] or [n_samples, forecast_len, n_features]
        feature_names: List of feature names
        config: Configuration for thresholds
        cusum_combinations: List of CUSUM parameter combinations to test.
                           If None, uses DEFAULT_CUSUM_COMBINATIONS.
        verbose: If True, print calibration info
        
    Returns:
        Dictionary with calibrated thresholds for each feature
    """
    if config is None:
        config = ForecastAnomalyConfig()
    
    if cusum_combinations is None:
        cusum_combinations = DEFAULT_CUSUM_COMBINATIONS
    
    # Flatten if 3D (from multi-step forecast)
    if residuals.ndim == 3:
        # [n_samples, forecast_len, n_features] -> [n_samples * forecast_len, n_features]
        residuals = residuals.reshape(-1, residuals.shape[-1])
    
    n_samples, n_features = residuals.shape
    
    if verbose:
        print(f"Calibrating thresholds from {n_samples:,} residual samples...")
        print(f"Testing {len(cusum_combinations)} CUSUM combinations per feature")
    
    thresholds = {
        'meta': {
            'sigma_warning': config.sigma_warning,
            'sigma_danger': config.sigma_danger,
            'ewma_alpha': config.ewma_alpha,
            'cusum_combinations_tested': [c['name'] for c in cusum_combinations],
        },
        'features': {}
    }
    
    for i, fname in enumerate(feature_names):
        if i >= n_features:
            break
            
        res = residuals[:, i]
        
        if verbose:
            print(f"\n{fname}:")
        
        # Basic statistics
        mean_res = float(np.mean(res))
        std_res = float(np.std(res))
        median_res = float(np.median(res))
        mad_res = float(np.median(np.abs(res - median_res)))
        
        # Percentiles for robust estimation
        p1, p5, p25, p75, p95, p99 = np.percentile(res, [1, 5, 25, 75, 95, 99])
        iqr = p75 - p25
        
        # Sigma-based thresholds
        warning_lower = mean_res - config.sigma_warning * std_res
        warning_upper = mean_res + config.sigma_warning * std_res
        danger_lower = mean_res - config.sigma_danger * std_res
        danger_upper = mean_res + config.sigma_danger * std_res
        
        if verbose:
            print(f"  Statistics: mean={mean_res:.6f}, std={std_res:.6f}")
            print(f"  Sigma thresholds: warning=[{warning_lower:.4f}, {warning_upper:.4f}]")
        
        # Test multiple CUSUM combinations and select best
        cusum_results, best_cusum_name, cusum_info = compute_cusum_multi_params(
            res,
            alpha=config.ewma_alpha,
            baseline_samples=n_samples,  # Use all samples as baseline for characterization
            combinations=cusum_combinations,
            verbose=verbose
        )
        
        thresholds['features'][fname] = {
            'statistics': {
                'mean': mean_res,
                'std': std_res,
                'median': median_res,
                'mad': mad_res,
                'iqr': iqr,
                'percentiles': {
                    'p1': float(p1), 'p5': float(p5), 'p25': float(p25),
                    'p75': float(p75), 'p95': float(p95), 'p99': float(p99)
                }
            },
            'sigma_thresholds': {
                'warning_lower': warning_lower,
                'warning_upper': warning_upper,
                'danger_lower': danger_lower,
                'danger_upper': danger_upper,
            },
            'cusum': cusum_info  # Contains best combo and all tested combinations
        }
    
    return thresholds


def validate_thresholds_on_test(
    residuals: np.ndarray,
    feature_names: List[str],
    thresholds: Dict,
    weight: float = 1.0,
    verbose: bool = True
) -> Dict:
    """
    Validate calibrated thresholds on LAB STRESS TEST (fatigue) data.
    
    This function tests how well the thresholds from calibration detect
    anomalies in the fatigue/stress test data. Since the machine is under
    stress in lab conditions, we expect more anomalies than in normal operation.
    
    Args:
        residuals: Test residuals [n_samples, n_features]
        feature_names: List of feature names
        thresholds: Calibrated thresholds from characterization (step 1)
        weight: Weight for this dataset (larger datasets may have higher weight)
        verbose: If True, print validation stats
        
    Returns:
        Dictionary with validation metrics
    """
    # Flatten if 3D
    if residuals.ndim == 3:
        residuals = residuals.reshape(-1, residuals.shape[-1])
    
    n_samples, n_features = residuals.shape
    
    if verbose:
        print(f"Validating thresholds on {n_samples:,} test samples (weight={weight:.2f})...")
    
    validation = {
        'n_samples': n_samples,
        'weight': weight,
        'features': {}
    }
    
    for i, fname in enumerate(feature_names):
        if i >= n_features or fname not in thresholds.get('features', {}):
            continue
        
        res = residuals[:, i]
        th = thresholds['features'][fname]
        
        # Count warnings and dangers
        sigma_th = th['sigma_thresholds']
        
        warning_mask = (
            (res < sigma_th['warning_lower']) | 
            (res > sigma_th['warning_upper'])
        )
        danger_mask = (
            (res < sigma_th['danger_lower']) | 
            (res > sigma_th['danger_upper'])
        )
        
        n_warning = int(warning_mask.sum())
        n_danger = int(danger_mask.sum())
        
        # CUSUM detection using best parameters from calibration
        cusum_params = th['cusum']
        cusum_mu0 = cusum_params.get('mu0', 0)
        cusum_h = cusum_params.get('h', 0)
        cusum_k = cusum_params.get('k', 0)
        
        cusum_pos, cusum_neg, _ = compute_cusum(res, cusum_mu0, cusum_h, cusum_k)
        
        # Warning at 75% of danger threshold
        h_warning = cusum_h * 0.75
        cusum_warning = (cusum_pos > h_warning) | (cusum_neg > h_warning)
        cusum_danger = (cusum_pos > cusum_h) | (cusum_neg > cusum_h)
        
        validation['features'][fname] = {
            'sigma_warning_count': n_warning,
            'sigma_warning_pct': round(n_warning / n_samples * 100, 4),
            'sigma_danger_count': n_danger,
            'sigma_danger_pct': round(n_danger / n_samples * 100, 4),
            'cusum_warning_count': int(cusum_warning.sum()),
            'cusum_warning_pct': round(cusum_warning.sum() / n_samples * 100, 4),
            'cusum_danger_count': int(cusum_danger.sum()),
            'cusum_danger_pct': round(cusum_danger.sum() / n_samples * 100, 4),
            'cusum_best_combo': cusum_params.get('best_combo', 'unknown'),
        }
        
        if verbose:
            print(f"  {fname}: warning={n_warning} ({n_warning/n_samples*100:.2f}%), "
                  f"danger={n_danger} ({n_danger/n_samples*100:.2f}%), "
                  f"cusum_combo={cusum_params.get('best_combo', 'unknown')}")
    
    return validation


def apply_thresholds_to_residuals(
    residuals: np.ndarray,
    feature_names: List[str],
    thresholds: Dict,
    return_cusum: bool = True
) -> Dict:
    """
    Apply calibrated thresholds to REAL MACHINE data (Tampieri) for anomaly detection.
    
    This is the final step: using thresholds learned from lab characterization
    and validated on lab stress test, detect anomalies on production data.
    
    Returns both warning (yellow) and danger (red) flags.
    
    Args:
        residuals: Residuals array [n_samples, n_features]
        feature_names: List of feature names
        thresholds: Calibrated thresholds from characterization
        return_cusum: If True, also return CUSUM values
        
    Returns:
        Dictionary with anomaly flags and CUSUM values for each feature
    """
    # Flatten if 3D
    if residuals.ndim == 3:
        original_shape = residuals.shape
        residuals = residuals.reshape(-1, residuals.shape[-1])
    else:
        original_shape = None
    
    n_samples, n_features = residuals.shape
    
    results = {
        'n_samples': n_samples,
        'features': {}
    }
    
    # Combined flags
    any_warning = np.zeros(n_samples, dtype=bool)
    any_danger = np.zeros(n_samples, dtype=bool)
    
    for i, fname in enumerate(feature_names):
        if i >= n_features or fname not in thresholds.get('features', {}):
            continue
        
        res = residuals[:, i]
        th = thresholds['features'][fname]
        sigma_th = th['sigma_thresholds']
        cusum_params = th['cusum']
        
        # Sigma-based detection
        sigma_warning = (
            (res < sigma_th['warning_lower']) | 
            (res > sigma_th['warning_upper'])
        )
        sigma_danger = (
            (res < sigma_th['danger_lower']) | 
            (res > sigma_th['danger_upper'])
        )
        
        # CUSUM detection using the best combination selected during calibration
        cusum_mu0 = cusum_params.get('mu0', 0)
        cusum_h = cusum_params.get('h', 0)
        cusum_k = cusum_params.get('k', 0)
        
        # Compute CUSUM with the best parameters
        cusum_pos, cusum_neg, _ = compute_cusum(res, cusum_mu0, cusum_h, cusum_k)
        
        # Warning threshold at ~75% of danger threshold
        h_warning = cusum_h * 0.75
        
        cusum_warning = (cusum_pos > h_warning) | (cusum_neg > h_warning)
        cusum_danger = (cusum_pos > cusum_h) | (cusum_neg > cusum_h)
        
        # Combined per-feature
        feature_warning = sigma_warning | cusum_warning
        feature_danger = sigma_danger | cusum_danger
        
        any_warning |= feature_warning
        any_danger |= feature_danger
        
        feature_result = {
            'sigma_warning': sigma_warning,
            'sigma_danger': sigma_danger,
            'cusum_warning': cusum_warning,  # ORANGE: early degradation (CUSUM > 75% h)
            'cusum_danger': cusum_danger,    # RED: confirmed (CUSUM > h)
            'warning': feature_warning,      # Combined warning
            'danger': feature_danger,        # Combined danger (red)
        }
        
        if return_cusum:
            feature_result['cusum_pos'] = cusum_pos
            feature_result['cusum_neg'] = cusum_neg
        
        results['features'][fname] = feature_result
    
    results['any_warning'] = any_warning
    results['any_danger'] = any_danger
    
    # Summary statistics
    results['summary'] = {
        'total_samples': n_samples,
        'warning_samples': int(any_warning.sum()),
        'danger_samples': int(any_danger.sum()),
        'warning_pct': round(any_warning.sum() / n_samples * 100, 4),
        'danger_pct': round(any_danger.sum() / n_samples * 100, 4),
    }
    
    return results


def create_anomaly_dataframe(
    df_original: pd.DataFrame,
    anomaly_results: Dict,
    feature_names: List[str],
    include_cusum: bool = True
) -> pd.DataFrame:
    """
    Create a DataFrame with anomaly flags added to original data.
    
    Args:
        df_original: Original DataFrame (must have same length as anomaly results)
        anomaly_results: Results from apply_thresholds_to_residuals
        feature_names: List of feature names
        include_cusum: If True, include CUSUM values
        
    Returns:
        DataFrame with anomaly columns added
    """
    df = df_original.copy()
    n_results = anomaly_results['n_samples']
    
    # Handle length mismatch (due to windowing)
    if len(df) != n_results:
        # Assume anomaly results correspond to the end of the dataframe
        df = df.iloc[-n_results:].copy()
        df = df.reset_index(drop=True)
    
    for fname in feature_names:
        if fname not in anomaly_results.get('features', {}):
            continue
        
        feat_res = anomaly_results['features'][fname]
        
        # Warning (yellow) and Danger (red) flags
        df[f'{fname}_warning'] = feat_res['warning'].astype(int)
        df[f'{fname}_danger'] = feat_res['danger'].astype(int)
        
        # Detailed flags
        df[f'{fname}_sigma_warning'] = feat_res['sigma_warning'].astype(int)
        df[f'{fname}_sigma_danger'] = feat_res['sigma_danger'].astype(int)
        df[f'{fname}_cusum_warning'] = feat_res['cusum_warning'].astype(int)
        df[f'{fname}_cusum_danger'] = feat_res['cusum_danger'].astype(int)
        
        if include_cusum and 'cusum_pos' in feat_res:
            df[f'{fname}_cusum_pos'] = feat_res['cusum_pos']
            df[f'{fname}_cusum_neg'] = feat_res['cusum_neg']
    
    # Global flags
    df['any_warning'] = anomaly_results['any_warning'].astype(int)
    df['any_danger'] = anomaly_results['any_danger'].astype(int)
    
    # Health status
    df['health_status'] = np.where(
        anomaly_results['any_danger'], 'RED',
        np.where(anomaly_results['any_warning'], 'YELLOW', 'GREEN')
    )
    
    return df


def combine_thresholds_weighted(
    thresholds_list: List[Dict],
    weights: List[float]
) -> Dict:
    """
    Combine thresholds from multiple calibrations with weights.
    
    Useful when calibrating on characterization and validating on fatigue,
    then combining with appropriate weights.
    
    Args:
        thresholds_list: List of threshold dictionaries
        weights: Weights for each threshold set (will be normalized)
        
    Returns:
        Combined threshold dictionary
    """
    if not thresholds_list:
        raise ValueError("No thresholds to combine")
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Start with first threshold set
    combined = {
        'config': thresholds_list[0].get('config', {}),
        'features': {}
    }
    
    # Get all feature names
    all_features = set()
    for th in thresholds_list:
        all_features.update(th.get('features', {}).keys())
    
    # Combine statistics for each feature
    for fname in all_features:
        combined['features'][fname] = {
            'statistics': {},
            'sigma_thresholds': {},
            'cusum': {}
        }
        
        # Weighted average of statistics
        for stat_key in ['mean', 'std', 'median', 'mad', 'iqr']:
            values = []
            w_list = []
            for th, w in zip(thresholds_list, weights):
                if fname in th.get('features', {}) and stat_key in th['features'][fname].get('statistics', {}):
                    values.append(th['features'][fname]['statistics'][stat_key])
                    w_list.append(w)
            
            if values:
                # Normalize weights for available values
                w_sum = sum(w_list)
                weighted_avg = sum(v * w / w_sum for v, w in zip(values, w_list))
                combined['features'][fname]['statistics'][stat_key] = weighted_avg
        
        # Weighted average of thresholds
        for th_key in ['warning_lower', 'warning_upper', 'danger_lower', 'danger_upper']:
            values = []
            w_list = []
            for th, w in zip(thresholds_list, weights):
                if fname in th.get('features', {}) and th_key in th['features'][fname].get('sigma_thresholds', {}):
                    values.append(th['features'][fname]['sigma_thresholds'][th_key])
                    w_list.append(w)
            
            if values:
                w_sum = sum(w_list)
                weighted_avg = sum(v * w / w_sum for v, w in zip(values, w_list))
                combined['features'][fname]['sigma_thresholds'][th_key] = weighted_avg
        
        # Weighted average of CUSUM parameters
        for cusum_key in ['mu0', 'k', 'h_warning', 'h_danger']:
            values = []
            w_list = []
            for th, w in zip(thresholds_list, weights):
                if fname in th.get('features', {}) and cusum_key in th['features'][fname].get('cusum', {}):
                    values.append(th['features'][fname]['cusum'][cusum_key])
                    w_list.append(w)
            
            if values:
                w_sum = sum(w_list)
                weighted_avg = sum(v * w / w_sum for v, w in zip(values, w_list))
                combined['features'][fname]['cusum'][cusum_key] = weighted_avg
    
    return combined


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def classify_anomaly_type(
    row: pd.Series,
    feature: str,
    current_col: str,
    startup_threshold: float
) -> str:
    """
    Classify anomaly type based on conditions.
    
    Args:
        row: DataFrame row
        feature: Feature name
        current_col: Name of current column for startup detection
        startup_threshold: Threshold for startup condition
        
    Returns:
        Color string: 'yellow' (startup), 'orange' (CUSUM), 'red' (other)
    """
    # Check if CUSUM anomaly
    cusum_col = f'{feature}_cusum_warning'
    if cusum_col in row.index and row[cusum_col] == 1:
        return 'orange'
    
    # Check if startup (low current)
    if current_col in row.index and pd.notna(row[current_col]):
        if row[current_col] < startup_threshold:
            return 'yellow'
    
    # Default: other anomaly
    return 'red'


def plot_anomalies_with_colors(
    df: pd.DataFrame,
    feature: str,
    datetime_col: str = 'datetime',
    current_col: Optional[str] = None,
    startup_quantile: float = 0.05,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
    show_legend: bool = True,
    title: Optional[str] = None,
    show_metrics_box: bool = True
) -> Optional[object]:
    """
    Plot time series with color-coded anomalies.
    
    Color coding:
    - YELLOW: Anomalies during startup (low current condition)
    - ORANGE: CUSUM early warning (drift detection, CUSUM > 75% of h threshold)
    - RED: Confirmed anomalies (sigma threshold or CUSUM > h threshold)
    
    Detection methods:
    - Sigma-based: Warning at ±2σ, Danger at ±3σ from baseline mean
    - CUSUM: Cumulative sum control chart for drift detection
      - Warning (orange): CUSUM exceeds 75% of threshold h
      - Danger (red): CUSUM exceeds threshold h
    
    Args:
        df: DataFrame with anomaly columns (from create_anomaly_dataframe)
        feature: Feature name to plot
        datetime_col: Name of datetime column
        current_col: Name of current column for startup detection.
                    If None, will look for 'current' or similar
        startup_quantile: Quantile for startup threshold (default 0.05)
        figsize: Figure size
        save_path: If provided, save figure to this path
        show_legend: Whether to show legend
        title: Custom title (if None, auto-generated)
        show_metrics_box: If True, show info box with detection metrics
        
    Returns:
        matplotlib figure or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available for plotting")
        return None
    
    # Find current column if not specified
    if current_col is None:
        possible_current_cols = ['current', 'Current', 'corrente', 'Corrente', 'I', 'i']
        for col in possible_current_cols:
            if col in df.columns:
                current_col = col
                break
    
    # Calculate startup threshold from current column
    startup_threshold = None
    if current_col and current_col in df.columns:
        startup_threshold = df[current_col].quantile(startup_quantile)
    
    # Prepare datetime index
    if datetime_col in df.columns:
        x_values = pd.to_datetime(df[datetime_col])
    else:
        x_values = df.index
    
    # Get feature values
    y_values = df[feature].values
    
    # Identify anomaly columns
    warning_col = f'{feature}_warning'
    danger_col = f'{feature}_danger'
    cusum_warning_col = f'{feature}_cusum_warning'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot main time series
    ax.plot(x_values, y_values, 'b-', alpha=0.7, linewidth=0.5, label='Signal')
    
    # Collect anomaly points by color
    yellow_idx = []  # Startup anomalies
    orange_idx = []  # CUSUM anomalies
    red_idx = []     # Other anomalies
    
    # Check for warning/danger flags
    if warning_col in df.columns or danger_col in df.columns:
        # Get indices where there's any anomaly
        if warning_col in df.columns and danger_col in df.columns:
            anomaly_mask = (df[warning_col] == 1) | (df[danger_col] == 1)
        elif warning_col in df.columns:
            anomaly_mask = df[warning_col] == 1
        else:
            anomaly_mask = df[danger_col] == 1
        
        anomaly_indices = df.index[anomaly_mask].tolist()
        
        # Classify each anomaly
        for idx in anomaly_indices:
            row = df.loc[idx]
            
            # Check CUSUM first
            is_cusum = False
            if cusum_warning_col in df.columns:
                is_cusum = row[cusum_warning_col] == 1
            
            if is_cusum:
                orange_idx.append(idx)
            elif startup_threshold is not None and current_col in df.columns:
                if pd.notna(row[current_col]) and row[current_col] < startup_threshold:
                    yellow_idx.append(idx)
                else:
                    red_idx.append(idx)
            else:
                red_idx.append(idx)
    
    # Plot anomalies with colors
    if yellow_idx:
        ax.scatter(
            [x_values.iloc[i] if hasattr(x_values, 'iloc') else x_values[i] for i in yellow_idx],
            [y_values[i] for i in yellow_idx],
            c='yellow', s=30, alpha=0.8, edgecolors='gold', linewidths=0.5,
            label=f'Startup (n={len(yellow_idx)})', zorder=5
        )
    
    if orange_idx:
        ax.scatter(
            [x_values.iloc[i] if hasattr(x_values, 'iloc') else x_values[i] for i in orange_idx],
            [y_values[i] for i in orange_idx],
            c='orange', s=35, alpha=0.8, edgecolors='darkorange', linewidths=0.5,
            label=f'CUSUM early warning (n={len(orange_idx)})', zorder=6
        )
    
    if red_idx:
        ax.scatter(
            [x_values.iloc[i] if hasattr(x_values, 'iloc') else x_values[i] for i in red_idx],
            [y_values[i] for i in red_idx],
            c='red', s=30, alpha=0.8, edgecolors='darkred', linewidths=0.5,
            label=f'Confirmed anomaly (n={len(red_idx)})', zorder=7
        )
    
    # Formatting
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Anomaly Detection - {feature}', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel(feature, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    if show_legend:
        ax.legend(loc='upper right', fontsize=10)
    
    # Add metrics info box
    if show_metrics_box:
        total_anomalies = len(yellow_idx) + len(orange_idx) + len(red_idx)
        metrics_text = (
            f"Detection Methods:\n"
            f"• Sigma: ±2σ (warning), ±3σ (danger)\n"
            f"• CUSUM: 75%h (orange), h (red)\n"
            f"─────────────────────\n"
            f"Yellow: Startup ({len(yellow_idx)})\n"
            f"Orange: CUSUM drift ({len(orange_idx)})\n"
            f"Red: Confirmed ({len(red_idx)})\n"
            f"Total: {total_anomalies}"
        )
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace', bbox=props)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_all_features_anomalies(
    df: pd.DataFrame,
    feature_names: List[str],
    datetime_col: str = 'datetime',
    current_col: Optional[str] = None,
    startup_quantile: float = 0.05,
    figsize_per_plot: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None,
    title_prefix: str = ''
) -> Optional[object]:
    """
    Plot multiple features with color-coded anomalies in subplots.
    
    Args:
        df: DataFrame with anomaly columns
        feature_names: List of feature names to plot
        datetime_col: Name of datetime column
        current_col: Name of current column for startup detection
        startup_quantile: Quantile for startup threshold
        figsize_per_plot: Size per subplot
        save_path: If provided, save figure to this path
        title_prefix: Prefix for title
        
    Returns:
        matplotlib figure or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available for plotting")
        return None
    
    n_features = len(feature_names)
    fig, axes = plt.subplots(n_features, 1, 
                              figsize=(figsize_per_plot[0], figsize_per_plot[1] * n_features),
                              sharex=True)
    
    if n_features == 1:
        axes = [axes]
    
    # Find current column if not specified
    if current_col is None:
        possible_current_cols = ['current', 'Current', 'corrente', 'Corrente', 'I', 'i']
        for col in possible_current_cols:
            if col in df.columns:
                current_col = col
                break
    
    # Calculate startup threshold
    startup_threshold = None
    if current_col and current_col in df.columns:
        startup_threshold = df[current_col].quantile(startup_quantile)
    
    # Prepare datetime index
    if datetime_col in df.columns:
        x_values = pd.to_datetime(df[datetime_col])
    else:
        x_values = df.index
    
    for ax, feature in zip(axes, feature_names):
        y_values = df[feature].values
        
        # Plot main signal
        ax.plot(x_values, y_values, 'b-', alpha=0.7, linewidth=0.5)
        
        # Get anomaly indices
        warning_col = f'{feature}_warning'
        danger_col = f'{feature}_danger'
        cusum_warning_col = f'{feature}_cusum_warning'
        
        yellow_idx = []
        orange_idx = []
        red_idx = []
        
        if warning_col in df.columns or danger_col in df.columns:
            if warning_col in df.columns and danger_col in df.columns:
                anomaly_mask = (df[warning_col] == 1) | (df[danger_col] == 1)
            elif warning_col in df.columns:
                anomaly_mask = df[warning_col] == 1
            else:
                anomaly_mask = df[danger_col] == 1
            
            anomaly_indices = df.index[anomaly_mask].tolist()
            
            for idx in anomaly_indices:
                row = df.loc[idx]
                
                is_cusum = False
                if cusum_warning_col in df.columns:
                    is_cusum = row[cusum_warning_col] == 1
                
                if is_cusum:
                    orange_idx.append(idx)
                elif startup_threshold is not None and current_col in df.columns:
                    if pd.notna(row[current_col]) and row[current_col] < startup_threshold:
                        yellow_idx.append(idx)
                    else:
                        red_idx.append(idx)
                else:
                    red_idx.append(idx)
        
        # Plot anomalies
        if yellow_idx:
            ax.scatter(
                [x_values.iloc[i] if hasattr(x_values, 'iloc') else x_values[i] for i in yellow_idx],
                [y_values[i] for i in yellow_idx],
                c='yellow', s=20, alpha=0.8, edgecolors='gold', linewidths=0.3,
                label=f'Startup ({len(yellow_idx)})', zorder=5
            )
        
        if orange_idx:
            ax.scatter(
                [x_values.iloc[i] if hasattr(x_values, 'iloc') else x_values[i] for i in orange_idx],
                [y_values[i] for i in orange_idx],
                c='orange', s=25, alpha=0.8, edgecolors='darkorange', linewidths=0.3,
                label=f'CUSUM ({len(orange_idx)})', zorder=6
            )
        
        if red_idx:
            ax.scatter(
                [x_values.iloc[i] if hasattr(x_values, 'iloc') else x_values[i] for i in red_idx],
                [y_values[i] for i in red_idx],
                c='red', s=20, alpha=0.8, edgecolors='darkred', linewidths=0.3,
                label=f'Other ({len(red_idx)})', zorder=7
            )
        
        ax.set_ylabel(feature, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Time', fontsize=12)
    
    # Main title
    suptitle = f'{title_prefix}Anomaly Detection - Color Coded' if title_prefix else 'Anomaly Detection - Color Coded'
    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_anomaly_summary(
    df: pd.DataFrame,
    feature_names: List[str],
    current_col: Optional[str] = None,
    startup_quantile: float = 0.05,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> Optional[object]:
    """
    Create a summary bar chart of anomaly counts by type and feature.
    
    Args:
        df: DataFrame with anomaly columns
        feature_names: List of feature names
        current_col: Name of current column for startup detection
        startup_quantile: Quantile for startup threshold
        figsize: Figure size
        save_path: If provided, save figure to this path
        
    Returns:
        matplotlib figure or None if matplotlib not available
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available for plotting")
        return None
    
    # Find current column if not specified
    if current_col is None:
        possible_current_cols = ['current', 'Current', 'corrente', 'Corrente', 'I', 'i']
        for col in possible_current_cols:
            if col in df.columns:
                current_col = col
                break
    
    # Calculate startup threshold
    startup_threshold = None
    if current_col and current_col in df.columns:
        startup_threshold = df[current_col].quantile(startup_quantile)
    
    # Count anomalies by type for each feature
    summary_data = []
    
    for feature in feature_names:
        warning_col = f'{feature}_warning'
        danger_col = f'{feature}_danger'
        cusum_warning_col = f'{feature}_cusum_warning'
        
        yellow_count = 0
        orange_count = 0
        red_count = 0
        
        if warning_col in df.columns or danger_col in df.columns:
            if warning_col in df.columns and danger_col in df.columns:
                anomaly_mask = (df[warning_col] == 1) | (df[danger_col] == 1)
            elif warning_col in df.columns:
                anomaly_mask = df[warning_col] == 1
            else:
                anomaly_mask = df[danger_col] == 1
            
            anomaly_indices = df.index[anomaly_mask].tolist()
            
            for idx in anomaly_indices:
                row = df.loc[idx]
                
                is_cusum = False
                if cusum_warning_col in df.columns:
                    is_cusum = row[cusum_warning_col] == 1
                
                if is_cusum:
                    orange_count += 1
                elif startup_threshold is not None and current_col in df.columns:
                    if pd.notna(row[current_col]) and row[current_col] < startup_threshold:
                        yellow_count += 1
                    else:
                        red_count += 1
                else:
                    red_count += 1
        
        summary_data.append({
            'feature': feature,
            'startup': yellow_count,
            'cusum': orange_count,
            'other': red_count
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(feature_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, summary_df['startup'], width, label='Startup (Yellow)', 
                   color='yellow', edgecolor='gold', linewidth=1)
    bars2 = ax.bar(x, summary_df['cusum'], width, label='CUSUM (Orange)',
                   color='orange', edgecolor='darkorange', linewidth=1)
    bars3 = ax.bar(x + width, summary_df['other'], width, label='Other (Red)',
                   color='red', edgecolor='darkred', linewidth=1)
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Anomaly Count', fontsize=12)
    ax.set_title('Anomaly Count by Type and Feature', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


if __name__ == "__main__":
    print("Anomaly Detection Module - Test")
    print("=" * 50)
    
    # Test with characterization dataset (real data)
    import os
    test_data_path = "../../data/final/bonfiglioli_caratterizzazione_continuous.csv"
    
    if os.path.exists(test_data_path):
        print(f"\nLoading real data: {test_data_path}")
        df_test = pd.read_csv(test_data_path)
        print(f"Dataset shape: {df_test.shape}")
        print(f"Columns: {list(df_test.columns)}")
        
        # Get numeric columns (exclude datetime)
        numeric_cols = [c for c in df_test.columns if c not in ['datetime', 'timestamp', 'time']]
        print(f"Numeric columns: {numeric_cols}")
        
        # Use first 10000 samples for quick test
        df_sample = df_test.head(10000)
        
        config = AnomalyConfig(z_window=100, trend_window=500, cusum_baseline_samples=1000)
        
        # Analyze first sensor
        if numeric_cols:
            signal = df_sample[numeric_cols[0]].values.astype(float)
            results = analyze_signal_anomalies(signal, numeric_cols[0], config)
            
            print(f"\nResults for {numeric_cols[0]}:")
            print(f"  Z-score anomalies: {results['z_anomalies'].sum()}")
            print(f"  MAD anomalies: {results['mad_anomalies'].sum()}")
            print(f"  Peaks: {results['peaks'].sum()}")
            print(f"  Trend anomalies: {results['trend_anomalies'].sum()}")
            print(f"  CUSUM alarms: {results['cusum_alarm'].sum()}")
            print(f"  Any anomaly: {results['any_anomaly'].sum()}")
    else:
        print(f"\nTest data not found: {test_data_path}")
        print("Creating synthetic data for test...")
        
        np.random.seed(42)
        n = 10000
        signal = np.sin(np.linspace(0, 20 * np.pi, n)) + np.random.randn(n) * 0.1
        signal[5000:5010] = 5  # Anomalies
        
        config = AnomalyConfig(z_window=100, trend_window=500, cusum_baseline_samples=1000)
        
        results = analyze_signal_anomalies(signal, 'test_signal', config)
        
        print(f"\nResults:")
        print(f"  Z-score anomalies: {results['z_anomalies'].sum()}")
        print(f"  MAD anomalies: {results['mad_anomalies'].sum()}")
        print(f"  Peaks: {results['peaks'].sum()}")
        print(f"  Trend anomalies: {results['trend_anomalies'].sum()}")
        print(f"  CUSUM alarms: {results['cusum_alarm'].sum()}")
        print(f"  Any anomaly: {results['any_anomaly'].sum()}")