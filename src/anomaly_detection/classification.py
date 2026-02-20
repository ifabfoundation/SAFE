"""
Forecasting-based Anomaly Detection con due approcci:

APPROCCIO A (Independent Windows):
- Ogni predizione usa i dati REALI della finestra precedente
- Le finestre sono indipendenti tra loro

APPROCCIO B (Rolling Forecast):  
- La prima predizione usa dati reali
- Le predizioni successive usano i valori PREDETTI come input
- Simula deployment reale dove non hai i dati futuri

Pipeline:
1. Resample a 0.5 Hz (frequenza Campo)
2. Applica Domain Adaptation sui dati grezzi
3. Train modelli di forecasting
4. Valuta con entrambi gli approcci
5. Genera plot dell'intera serie temporale
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURAZIONE
# =============================================================================

@dataclass 
class Config:
    target_rate_hz: float = 0.5
    window_hours: int = 2
    
    lab_features: List[str] = None
    campo_features: List[str] = None
    unified_features: List[str] = None
    
    lab_caratterizzazione: str = "/home/projects/safe/data/bonfiglioli/processed/ETR_946 - Prova 30_0 - 310 - Caratterizzazione 24h_1.csv"
    lab_fatica_dir: str = "/home/projects/safe/data/bonfiglioli/processed"
    campo_tampieri: str = "/home/projects/safe/data/tampieri/processed_streaming_row_continuous.csv"
    output_dir: str = "/home/projects/safe/results/forecasting"
    
    def __post_init__(self):
        self.samples_per_hour = int(self.target_rate_hz * 3600)
        self.window_samples = self.samples_per_hour * self.window_hours
        
        if self.lab_features is None:
            self.lab_features = [
                'vib_rid_marc2_rms', 'vib_rid_marcia_rms',
                'temp_mot_marcia', 'temp_cassa_riduttore'
            ]
        if self.campo_features is None:
            self.campo_features = [
                'bonfi/gb1_p4_acc_rms', 'bonfi/gb1_p3_acc_rms',
                'bonfi/gb1_p4_temp', 'bonfi/gb1_p3_temp'
            ]
        if self.unified_features is None:
            self.unified_features = ['vib_1', 'vib_2', 'temp_1', 'temp_2']


# =============================================================================
# DOMAIN ADAPTATION METHODS
# =============================================================================

class DomainAdapter(ABC):
    name: str = "base"
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'DomainAdapter':
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class NoAdapter(DomainAdapter):
    name = "None"
    
    def fit(self, X: np.ndarray) -> 'NoAdapter':
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X.copy()


class StandardAdapter(DomainAdapter):
    name = "StandardScaler"
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, X: np.ndarray) -> 'StandardAdapter':
        self.scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)


class RobustAdapter(DomainAdapter):
    name = "RobustScaler"
    
    def __init__(self):
        self.scaler = RobustScaler()
    
    def fit(self, X: np.ndarray) -> 'RobustAdapter':
        self.scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)


class MinMaxAdapter(DomainAdapter):
    name = "MinMaxScaler"
    
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def fit(self, X: np.ndarray) -> 'MinMaxAdapter':
        self.scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)


# =============================================================================
# DATA LOADING
# =============================================================================

class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.lab_to_unified = dict(zip(config.lab_features, config.unified_features))
        self.campo_to_unified = dict(zip(config.campo_features, config.unified_features))
    
    def resample_to_target_rate(self, df: pd.DataFrame, time_col: str, feature_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col).reset_index(drop=True)
        
        interval_seconds = 1.0 / self.config.target_rate_hz
        start_time = df[time_col].iloc[0]
        
        df['time_delta'] = (df[time_col] - start_time).dt.total_seconds()
        df['bin'] = (df['time_delta'] / interval_seconds).astype(int)
        
        return df.groupby('bin')[feature_cols].mean().reset_index(drop=True)
    
    def load_lab(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        feature_cols = [c for c in self.config.lab_features if c in df.columns]
        df_resampled = self.resample_to_target_rate(df, 'datetime', feature_cols)
        rename_map = {old: self.lab_to_unified[old] for old in feature_cols}
        df_resampled = df_resampled.rename(columns=rename_map)
        return df_resampled[self.config.unified_features]
    
    def load_campo(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        feature_cols = [c for c in self.config.campo_features if c in df.columns]
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_diffs = df['timestamp'].diff().dt.total_seconds().dropna()
        median_interval = time_diffs.median()
        
        if abs(median_interval - 2.0) > 0.1:
            df_resampled = self.resample_to_target_rate(df, 'timestamp', feature_cols)
        else:
            df_resampled = df[feature_cols].copy()
        
        rename_map = {old: self.campo_to_unified[old] for old in feature_cols}
        df_resampled = df_resampled.rename(columns=rename_map)
        return df_resampled[self.config.unified_features].dropna()
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        print("Caricamento e resampling dati...")
        print(f"   Target rate: {self.config.target_rate_hz} Hz")
        print(f"   Finestra: {self.config.window_hours}h = {self.config.window_samples} campioni\n")
        
        df_lab = self.load_lab(self.config.lab_caratterizzazione)
        print(f"  Lab Caratterizzazione: {len(df_lab):,} campioni")
        
        fatica_files = sorted(Path(self.config.lab_fatica_dir).glob("*Fatica*.csv"))
        dfs_fatica = [self.load_lab(str(f)) for f in fatica_files]
        df_fatica = pd.concat(dfs_fatica, ignore_index=True)
        print(f"  Lab Fatica: {len(df_fatica):,} campioni")
        
        df_campo = self.load_campo(self.config.campo_tampieri)
        print(f"  Campo Tampieri: {len(df_campo):,} campioni")
        
        return {
            'lab_normal': df_lab,
            'lab_fatica': df_fatica,
            'campo': df_campo
        }


# =============================================================================
# WINDOWING
# =============================================================================

def create_windows(data: np.ndarray, input_size: int, output_size: int, stride: int = None) -> Tuple[np.ndarray, np.ndarray]:
    if stride is None:
        stride = output_size
    
    X_list, Y_list = [], []
    i = 0
    while i + input_size + output_size <= len(data):
        X_list.append(data[i:i+input_size])
        Y_list.append(data[i+input_size:i+input_size+output_size])
        i += stride
    
    return np.array(X_list), np.array(Y_list)


# =============================================================================
# FORECASTING MODELS
# =============================================================================

class PersistenceModel:
    name = "Persistence"
    
    def fit(self, X, Y):
        self.output_size = Y.shape[1]
        return self
    
    def predict(self, X):
        last = X[:, -1, :]
        return np.repeat(last[:, np.newaxis, :], self.output_size, axis=1)
    
    def predict_single(self, x_window):
        return np.repeat(x_window[-1:, :], self.output_size, axis=0)


class MovingAverageModel:
    name = "MovingAverage"
    
    def __init__(self, window: int = 100):
        self.window = window
    
    def fit(self, X, Y):
        self.output_size = Y.shape[1]
        return self
    
    def predict(self, X):
        mean = X[:, -self.window:, :].mean(axis=1)
        return np.repeat(mean[:, np.newaxis, :], self.output_size, axis=1)
    
    def predict_single(self, x_window):
        mean = x_window[-self.window:, :].mean(axis=0)
        return np.repeat(mean[np.newaxis, :], self.output_size, axis=0)


class LinearTrendModel:
    name = "LinearTrend"
    
    def fit(self, X, Y):
        self.input_size = X.shape[1]
        self.output_size = Y.shape[1]
        return self
    
    def predict(self, X):
        n_windows, input_size, n_features = X.shape
        Y_pred = np.zeros((n_windows, self.output_size, n_features))
        
        t_input = np.arange(input_size)
        t_output = np.arange(input_size, input_size + self.output_size)
        
        for i in range(n_windows):
            for f in range(n_features):
                y = X[i, :, f]
                slope, intercept = np.polyfit(t_input, y, 1)
                Y_pred[i, :, f] = slope * t_output + intercept
        
        return Y_pred
    
    def predict_single(self, x_window):
        input_size, n_features = x_window.shape
        Y_pred = np.zeros((self.output_size, n_features))
        
        t_input = np.arange(input_size)
        t_output = np.arange(input_size, input_size + self.output_size)
        
        for f in range(n_features):
            slope, intercept = np.polyfit(t_input, x_window[:, f], 1)
            Y_pred[:, f] = slope * t_output + intercept
        
        return Y_pred


class VARModel:
    name = "VAR"
    
    def __init__(self, lag: int = 50):
        self.lag = lag
        self.models = []
        self.internal_scaler = StandardScaler()
    
    def fit(self, X, Y):
        self.output_size = Y.shape[1]
        self.n_features = X.shape[2]
        
        X_flat = X[:, -self.lag:, :].reshape(X.shape[0], -1)
        X_scaled = self.internal_scaler.fit_transform(X_flat)
        
        self.models = []
        for t in range(self.output_size):
            models_t = []
            for f in range(self.n_features):
                model = Ridge(alpha=1.0)
                model.fit(X_scaled, Y[:, t, f])
                models_t.append(model)
            self.models.append(models_t)
        
        return self
    
    def predict(self, X):
        n_windows = X.shape[0]
        Y_pred = np.zeros((n_windows, self.output_size, self.n_features))
        
        X_flat = X[:, -self.lag:, :].reshape(n_windows, -1)
        X_scaled = self.internal_scaler.transform(X_flat)
        
        for t in range(self.output_size):
            for f in range(self.n_features):
                Y_pred[:, t, f] = self.models[t][f].predict(X_scaled)
        
        return Y_pred
    
    def predict_single(self, x_window):
        Y_pred = np.zeros((self.output_size, self.n_features))
        
        X_flat = x_window[-self.lag:, :].reshape(1, -1)
        X_scaled = self.internal_scaler.transform(X_flat)
        
        for t in range(self.output_size):
            for f in range(self.n_features):
                Y_pred[t, f] = self.models[t][f].predict(X_scaled)[0]
        
        return Y_pred


class ExponentialSmoothingModel:
    name = "ExpSmoothing"
    
    def __init__(self, trend: str = 'add', seasonal: str = None):
        self.trend = trend
        self.seasonal = seasonal
        self.alpha_avg = []
    
    def fit(self, X, Y):
        self.output_size = Y.shape[1]
        self.n_features = X.shape[2]
        
        n_samples = min(20, X.shape[0])
        indices = np.linspace(0, X.shape[0]-1, n_samples, dtype=int)
        
        self.alpha_avg = []
        for f in range(self.n_features):
            alphas = []
            for idx in indices:
                try:
                    series = X[idx, :, f]
                    model = ExponentialSmoothing(series, trend=self.trend, seasonal=self.seasonal, initialization_method='estimated')
                    fitted = model.fit(optimized=True)
                    alphas.append(fitted.params.get('smoothing_level', 0.3))
                except:
                    alphas.append(0.3)
            self.alpha_avg.append(np.mean(alphas))
        
        return self
    
    def predict(self, X):
        n_windows = X.shape[0]
        Y_pred = np.zeros((n_windows, self.output_size, self.n_features))
        
        for i in range(n_windows):
            for f in range(self.n_features):
                try:
                    series = X[i, :, f]
                    model = ExponentialSmoothing(series, trend=self.trend, seasonal=self.seasonal, initialization_method='estimated')
                    fitted = model.fit(smoothing_level=self.alpha_avg[f], optimized=False)
                    Y_pred[i, :, f] = fitted.forecast(self.output_size)
                except:
                    Y_pred[i, :, f] = X[i, -1, f]
        
        return Y_pred
    
    def predict_single(self, x_window):
        Y_pred = np.zeros((self.output_size, self.n_features))
        
        for f in range(self.n_features):
            try:
                model = ExponentialSmoothing(x_window[:, f], trend=self.trend, seasonal=self.seasonal, initialization_method='estimated')
                fitted = model.fit(smoothing_level=self.alpha_avg[f], optimized=False)
                Y_pred[:, f] = fitted.forecast(self.output_size)
            except:
                Y_pred[:, f] = x_window[-1, f]
        
        return Y_pred


# =============================================================================
# APPROCCIO A: Independent Windows
# =============================================================================

def evaluate_approach_A(model, X_test: np.ndarray, Y_test: np.ndarray, threshold: float) -> Dict:
    """
    Approccio A: Ogni finestra e' indipendente.
    Input = dati reali, Output = predizione.
    """
    Y_pred = model.predict(X_test)
    errors = ((Y_test - Y_pred) ** 2).mean(axis=(1, 2))
    
    return {
        'Y_pred': Y_pred,
        'Y_true': Y_test,
        'errors': errors,
        'anomalies': errors > threshold,
        'threshold': threshold,
    }


# =============================================================================
# APPROCCIO B: Rolling Forecast
# =============================================================================

def evaluate_approach_B(model, data: np.ndarray, window_size: int, threshold: float) -> Dict:
    """
    Approccio B: Rolling forecast.
    - Prima finestra: usa dati reali
    - Finestre successive: usa dati PREDETTI come input
    """
    n_samples, n_features = data.shape
    n_windows = (n_samples - window_size) // window_size
    
    Y_pred_all = []
    Y_true_all = []
    errors = []
    
    # Buffer iniziale: prima finestra reale
    current_input = data[:window_size].copy()
    
    for w in range(n_windows):
        start_true = (w + 1) * window_size
        end_true = (w + 2) * window_size
        
        if end_true > n_samples:
            break
        
        y_true = data[start_true:end_true]
        Y_true_all.append(y_true)
        
        y_pred = model.predict_single(current_input)
        Y_pred_all.append(y_pred)
        
        error = ((y_true - y_pred) ** 2).mean()
        errors.append(error)
        
        # Usa PREDIZIONI come prossimo input
        current_input = y_pred.copy()
    
    return {
        'Y_pred': np.array(Y_pred_all),
        'Y_true': np.array(Y_true_all),
        'errors': np.array(errors),
        'anomalies': np.array(errors) > threshold,
        'threshold': threshold,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def create_output_structure(base_dir: str) -> Dict[str, Path]:
    dirs = {
        'base': Path(base_dir),
        'approach_A': Path(base_dir) / 'approach_A',
        'approach_B': Path(base_dir) / 'approach_B',
        'comparison': Path(base_dir) / 'comparison',
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    
    for approach in ['approach_A', 'approach_B']:
        (dirs[approach] / 'errors').mkdir(exist_ok=True)
        (dirs[approach] / 'timeseries_normal').mkdir(exist_ok=True)
        (dirs[approach] / 'timeseries_anomaly').mkdir(exist_ok=True)
    
    return dirs


def plot_full_timeseries(result: Dict, dataset_name: str, output_path: Path, 
                         model_name: str, adapter_name: str, approach: str,
                         rate_hz: float = 0.5):
    """Plot intera serie temporale predetta vs reale."""
    Y_true = result['Y_true']
    Y_pred = result['Y_pred']
    errors = result['errors']
    anomalies = result['anomalies']
    threshold = result['threshold']
    
    n_windows, window_size, n_features = Y_true.shape
    feature_names = ['vib_1', 'vib_2', 'temp_1', 'temp_2']
    
    total_samples = n_windows * window_size
    time_hours = np.arange(total_samples) / (rate_hz * 3600)
    
    Y_true_flat = Y_true.reshape(-1, n_features)
    Y_pred_flat = Y_pred.reshape(-1, n_features)
    
    # Plot grande
    fig, axes = plt.subplots(n_features + 1, 1, figsize=(30, 4 * (n_features + 1)), sharex=True)
    
    for f in range(n_features):
        ax = axes[f]
        
        if 'vib' in feature_names[f]:
            c_true, c_pred = 'tab:blue', 'tab:cyan'
        else:
            c_true, c_pred = 'tab:red', 'tab:orange'
        
        ax.plot(time_hours, Y_true_flat[:, f], c_true, linewidth=0.5, alpha=0.8, label='Reale')
        ax.plot(time_hours, Y_pred_flat[:, f], c_pred, linewidth=0.5, alpha=0.6, label='Predetto')
        
        ax.set_ylabel(feature_names[f], fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # Errori
    ax = axes[-1]
    window_times = np.arange(n_windows) * window_size / (rate_hz * 3600)
    colors_bar = ['red' if a else 'green' for a in anomalies]
    ax.bar(window_times, errors, width=window_size / (rate_hz * 3600) * 0.9, color=colors_bar, alpha=0.7)
    ax.axhline(threshold, color='black', linestyle='--', linewidth=2, label='Soglia')
    ax.set_ylabel('MSE Error', fontsize=12)
    ax.set_xlabel('Tempo (ore)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    title = f'{model_name} + {adapter_name} - {approach} - {dataset_name}\n'
    title += f'Durata: {time_hours[-1]:.1f} ore | Finestre: {n_windows} | Anomalie: {anomalies.sum()}/{n_windows}'
    plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_error_distribution(result_normal: Dict, result_anomaly: Dict, 
                            output_path: Path, model_name: str, adapter_name: str, approach: str):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    threshold = result_normal['threshold']
    
    ax = axes[0]
    ax.hist(result_normal['errors'], bins=50, alpha=0.7, label='Normale (Tampieri)', color='green')
    ax.hist(result_anomaly['errors'], bins=50, alpha=0.7, label='Anomalo (Fatica)', color='red')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Soglia (95%)')
    ax.set_xlabel('MSE Error', fontsize=12)
    ax.set_ylabel('Frequenza', fontsize=12)
    ax.set_title(f'{model_name} + {adapter_name} - {approach}', fontsize=14)
    ax.legend()
    ax.set_yscale('log')
    
    ax = axes[1]
    bp = ax.boxplot([result_normal['errors'], result_anomaly['errors']], 
                     labels=['Normale', 'Anomalo'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    ax.axhline(threshold, color='black', linestyle='--', linewidth=2, label='Soglia')
    ax.set_ylabel('MSE Error', fontsize=12)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_comparison_AB(results_A: Dict, results_B: Dict, output_path: Path, 
                       model_name: str, adapter_name: str, dataset_name: str):
    fig, axes = plt.subplots(2, 1, figsize=(24, 12), sharex=True)
    
    # A
    ax = axes[0]
    n_windows_A = len(results_A['errors'])
    ax.bar(range(n_windows_A), results_A['errors'], 
           color=['red' if a else 'green' for a in results_A['anomalies']], alpha=0.7)
    ax.axhline(results_A['threshold'], color='black', linestyle='--', linewidth=2, label='Soglia')
    ax.set_ylabel('MSE Error', fontsize=12)
    ax.set_title(f'Approccio A (Independent) - Anomalie: {results_A["anomalies"].sum()}/{n_windows_A}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # B
    ax = axes[1]
    n_windows_B = len(results_B['errors'])
    ax.bar(range(n_windows_B), results_B['errors'],
           color=['red' if a else 'green' for a in results_B['anomalies']], alpha=0.7)
    ax.axhline(results_B['threshold'], color='black', linestyle='--', linewidth=2, label='Soglia')
    ax.set_ylabel('MSE Error', fontsize=12)
    ax.set_xlabel('Finestra (2h ciascuna)', fontsize=12)
    ax.set_title(f'Approccio B (Rolling) - Anomalie: {results_B["anomalies"].sum()}/{n_windows_B}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} + {adapter_name} - {dataset_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("FORECASTING PIPELINE: Approccio A vs B")
    print("=" * 70)
    print("\nApproccio A: Finestre indipendenti (input = dati reali)")
    print("Approccio B: Rolling forecast (input = predizioni precedenti)\n")
    
    config = Config()
    loader = DataLoader(config)
    dirs = create_output_structure(config.output_dir)
    
    data = loader.load_all()
    
    print("\nPreparazione split...")
    campo_split = int(len(data['campo']) * 0.8)
    
    train_lab = data['lab_normal'].values
    train_campo = data['campo'].values[:campo_split]
    test_normal = data['campo'].values[campo_split:]
    test_anomaly = data['lab_fatica'].values
    
    print(f"  Train Lab: {len(train_lab):,}")
    print(f"  Train Campo: {len(train_campo):,}")
    print(f"  Test Normale: {len(test_normal):,} ({len(test_normal) / config.target_rate_hz / 3600:.1f} ore)")
    print(f"  Test Anomalo: {len(test_anomaly):,} ({len(test_anomaly) / config.target_rate_hz / 3600:.1f} ore)")
    
    train_data_raw = np.vstack([train_lab, train_campo])
    
    adapters = [NoAdapter(), RobustAdapter()]
    model_classes = [PersistenceModel, MovingAverageModel, LinearTrendModel, VARModel]
    
    window_size = config.window_samples
    stride = window_size
    
    all_results = []
    
    for adapter in adapters:
        print(f"\n{'='*60}")
        print(f"Domain Adaptation: {adapter.name}")
        print(f"{'='*60}")
        
        adapter_instance = type(adapter)()
        train_adapted = adapter_instance.fit_transform(train_data_raw)
        test_normal_adapted = adapter_instance.transform(test_normal)
        test_anomaly_adapted = adapter_instance.transform(test_anomaly)
        
        X_train, Y_train = create_windows(train_adapted, window_size, window_size, stride)
        X_test_normal, Y_test_normal = create_windows(test_normal_adapted, window_size, window_size, stride)
        X_test_anomaly, Y_test_anomaly = create_windows(test_anomaly_adapted, window_size, window_size, stride)
        
        for model_class in model_classes:
            if model_class == MovingAverageModel:
                model = model_class(window=100)
            elif model_class == VARModel:
                model = model_class(lag=50)
            elif model_class == ExponentialSmoothingModel:
                model = model_class(trend='add')
            else:
                model = model_class()
            
            try:
                model.fit(X_train, Y_train)
                
                # Soglia da training
                Y_pred_train = model.predict(X_train)
                train_errors = ((Y_train - Y_pred_train) ** 2).mean(axis=(1, 2))
                threshold = np.percentile(train_errors, 95)
                
                print(f"\n  {model.name}")
                print(f"  {'-'*40}")
                
                # APPROCCIO A
                result_A_normal = evaluate_approach_A(model, X_test_normal, Y_test_normal, threshold)
                result_A_anomaly = evaluate_approach_A(model, X_test_anomaly, Y_test_anomaly, threshold)
                
                tp_A = result_A_anomaly['anomalies'].sum()
                fn_A = (~result_A_anomaly['anomalies']).sum()
                fp_A = result_A_normal['anomalies'].sum()
                tn_A = (~result_A_normal['anomalies']).sum()
                
                recall_A = tp_A / (tp_A + fn_A) if (tp_A + fn_A) > 0 else 0
                precision_A = tp_A / (tp_A + fp_A) if (tp_A + fp_A) > 0 else 0
                f1_A = 2 * precision_A * recall_A / (precision_A + recall_A) if (precision_A + recall_A) > 0 else 0
                fpr_A = fp_A / (fp_A + tn_A) if (fp_A + tn_A) > 0 else 0
                
                print(f"    Approccio A: Recall={recall_A:.3f} | Prec={precision_A:.3f} | F1={f1_A:.3f} | FPR={fpr_A:.3f}")
                
                # APPROCCIO B
                result_B_normal = evaluate_approach_B(model, test_normal_adapted, window_size, threshold)
                result_B_anomaly = evaluate_approach_B(model, test_anomaly_adapted, window_size, threshold)
                
                tp_B = result_B_anomaly['anomalies'].sum()
                fn_B = (~result_B_anomaly['anomalies']).sum()
                fp_B = result_B_normal['anomalies'].sum()
                tn_B = (~result_B_normal['anomalies']).sum()
                
                recall_B = tp_B / (tp_B + fn_B) if (tp_B + fn_B) > 0 else 0
                precision_B = tp_B / (tp_B + fp_B) if (tp_B + fp_B) > 0 else 0
                f1_B = 2 * precision_B * recall_B / (precision_B + recall_B) if (precision_B + recall_B) > 0 else 0
                fpr_B = fp_B / (fp_B + tn_B) if (fp_B + tn_B) > 0 else 0
                
                print(f"    Approccio B: Recall={recall_B:.3f} | Prec={precision_B:.3f} | F1={f1_B:.3f} | FPR={fpr_B:.3f}")
                
                # Salva
                all_results.append({
                    'Model': model.name, 'DA': adapter.name, 'Approach': 'A',
                    'Recall': recall_A, 'Precision': precision_A, 'F1': f1_A, 'FPR': fpr_A,
                    'TP': tp_A, 'TN': tn_A, 'FP': fp_A, 'FN': fn_A,
                })
                all_results.append({
                    'Model': model.name, 'DA': adapter.name, 'Approach': 'B',
                    'Recall': recall_B, 'Precision': precision_B, 'F1': f1_B, 'FPR': fpr_B,
                    'TP': tp_B, 'TN': tn_B, 'FP': fp_B, 'FN': fn_B,
                })
                
                # PLOT
                plot_full_timeseries(result_A_normal, 'Normal', 
                    dirs['approach_A'] / 'timeseries_normal' / f'{model.name}_{adapter.name}.png',
                    model.name, adapter.name, 'Approach_A', config.target_rate_hz)
                plot_full_timeseries(result_A_anomaly, 'Anomaly',
                    dirs['approach_A'] / 'timeseries_anomaly' / f'{model.name}_{adapter.name}.png',
                    model.name, adapter.name, 'Approach_A', config.target_rate_hz)
                plot_error_distribution(result_A_normal, result_A_anomaly,
                    dirs['approach_A'] / 'errors' / f'{model.name}_{adapter.name}.png',
                    model.name, adapter.name, 'Approach_A')
                
                plot_full_timeseries(result_B_normal, 'Normal',
                    dirs['approach_B'] / 'timeseries_normal' / f'{model.name}_{adapter.name}.png',
                    model.name, adapter.name, 'Approach_B', config.target_rate_hz)
                plot_full_timeseries(result_B_anomaly, 'Anomaly',
                    dirs['approach_B'] / 'timeseries_anomaly' / f'{model.name}_{adapter.name}.png',
                    model.name, adapter.name, 'Approach_B', config.target_rate_hz)
                plot_error_distribution(result_B_normal, result_B_anomaly,
                    dirs['approach_B'] / 'errors' / f'{model.name}_{adapter.name}.png',
                    model.name, adapter.name, 'Approach_B')
                
                plot_comparison_AB(result_A_anomaly, result_B_anomaly,
                    dirs['comparison'] / f'{model.name}_{adapter.name}_anomaly.png',
                    model.name, adapter.name, 'Anomaly')
                plot_comparison_AB(result_A_normal, result_B_normal,
                    dirs['comparison'] / f'{model.name}_{adapter.name}_normal.png',
                    model.name, adapter.name, 'Normal')
                
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    df = pd.DataFrame(all_results)
    df.to_csv(dirs['base'] / 'results_comparison.csv', index=False)
    
    print("\n" + "=" * 70)
    print("SUMMARY: Approccio A vs B (F1 Score)")
    print("=" * 70)
    
    pivot = df.pivot_table(index=['Model', 'DA'], columns='Approach', values='F1')
    print(pivot.to_string())
    
    print(f"\nOutput salvati in: {config.output_dir}/")


if __name__ == "__main__":
    main()
