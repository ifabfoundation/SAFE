"""
Exploratory Data Analysis (EDA) per Bonfiglioli Anomaly Detection Pipeline

Modulo per analisi esplorativa dei dati time series.
Crea una sottocartella per ogni segnale con tutti i plot e test.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict, Union
import logging
import warnings

# Statistical tests
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import STL

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

logger = logging.getLogger(__name__)


class DataExplorer:
    """
    Classe per esplorazione dati time series.
    
    Per ogni segnale crea una sottocartella con:
    - distribution.png
    - time_series.png
    - acf_pacf.png
    - stl_decomposition.png
    - boxplot.png
    - statistics.csv (include stationarity, normality tests)
    
    Globale:
    - correlation_matrix.png
    - strong_correlations.csv
    - summary_all_signals.csv
    """
    
    def __init__(self, config: dict):
        """
        Inizializza l'explorer.
        
        Args:
            config: Dizionario di configurazione con paths
        """
        self.config = config
        
        # Setup paths
        base_dir = Path(config['paths'].get('base_directory', '.'))
        self.plots_dir = base_dir / config['paths'].get('plots_directory', 'plots')
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Colonne temporali da escludere dall'analisi
        self.temporal_cols = ['datetime', 'time', 'time_relative']
        
        # Downsample factor
        self.downsample = config.get('exploration', {}).get('downsample', 10)
        
        logger.info(f"DataExplorer inizializzato, output: {self.plots_dir}")
    
    def _get_signal_cols(self, df: pd.DataFrame) -> List[str]:
        """Identifica colonne segnali (escluse temporali)."""
        return [c for c in df.columns 
                if c not in self.temporal_cols 
                and pd.api.types.is_numeric_dtype(df[c])]
    
    # =========================================================================
    # ANALISI PER SINGOLO SEGNALE
    # =========================================================================
    
    def analyze_signal(self, df: pd.DataFrame, signal: str, output_dir: Path, downsample: int = 10):
        """
        Esegue analisi completa per un singolo segnale.
        Crea sottocartella con tutti i plot e statistiche.
        
        Args:
            df: DataFrame con i dati
            signal: Nome del segnale
            output_dir: Cartella output per questo dataset
            downsample: Fattore di downsampling per velocità
        """
        # Crea sottocartella
        signal_dir = output_dir / signal
        signal_dir.mkdir(parents=True, exist_ok=True)
        
        data = df[signal].dropna()
        
        # Downsampling per velocità
        data_ds = data.iloc[::downsample]
        
        logger.info(f"  Analisi: {signal}")
        
        # 1. Statistiche descrittive e test
        stats_dict = self._compute_statistics(data, data_ds)
        pd.DataFrame([stats_dict]).to_csv(signal_dir / "statistics.csv", index=False)
        
        # 2. Plot distribuzione
        self._plot_distribution(data, signal, signal_dir)
        
        # 3. Plot time series
        self._plot_time_series(df, data_ds, signal, signal_dir)
        
        # 4. Plot ACF/PACF
        self._plot_acf_pacf(data_ds, signal, signal_dir)
        
        # 5. Decomposizione STL
        self._plot_stl(data_ds, signal, signal_dir, downsample)
        
        # 6. Boxplot
        self._plot_boxplot(data, signal, signal_dir)
    
    def _compute_statistics(self, data: pd.Series, data_ds: pd.Series) -> dict:
        """Calcola tutte le statistiche e test per un segnale."""
        stats_dict = {
            'n_samples': len(data),
            'missing': data.isna().sum(),
            'mean': round(data.mean(), 6),
            'std': round(data.std(), 6),
            'min': round(data.min(), 6),
            'max': round(data.max(), 6),
            'median': round(data.median(), 6),
            'q25': round(data.quantile(0.25), 6),
            'q75': round(data.quantile(0.75), 6),
            'skewness': round(data.skew(), 4),
            'kurtosis': round(data.kurtosis(), 4),
        }
        
        # Test stazionarietà (ADF)
        try:
            adf_stat, adf_p, _, _, _, _ = adfuller(data_ds, maxlag=30)
            stats_dict['adf_statistic'] = round(adf_stat, 4)
            stats_dict['adf_pvalue'] = round(adf_p, 4)
            stats_dict['adf_stationary'] = adf_p < 0.05
        except:
            stats_dict['adf_statistic'] = None
            stats_dict['adf_pvalue'] = None
            stats_dict['adf_stationary'] = None
        
        # Test stazionarietà (KPSS)
        try:
            kpss_stat, kpss_p, _, _ = kpss(data_ds, regression='c', nlags='auto')
            stats_dict['kpss_statistic'] = round(kpss_stat, 4)
            stats_dict['kpss_pvalue'] = round(kpss_p, 4)
            stats_dict['kpss_stationary'] = kpss_p > 0.05
        except:
            stats_dict['kpss_statistic'] = None
            stats_dict['kpss_pvalue'] = None
            stats_dict['kpss_stationary'] = None
        
        # Conclusione stazionarietà
        if stats_dict.get('adf_stationary') and stats_dict.get('kpss_stationary'):
            stats_dict['stationarity'] = 'Stationary'
        elif not stats_dict.get('adf_stationary') and not stats_dict.get('kpss_stationary'):
            stats_dict['stationarity'] = 'Non-stationary'
        else:
            stats_dict['stationarity'] = 'Inconclusive'
        
        # Test normalità (KS)
        try:
            data_norm = (data_ds - data_ds.mean()) / data_ds.std()
            ks_stat, ks_p = stats.kstest(data_norm, 'norm')
            stats_dict['ks_statistic'] = round(ks_stat, 4)
            stats_dict['ks_pvalue'] = round(ks_p, 4)
            stats_dict['ks_normal'] = ks_p > 0.05
        except:
            stats_dict['ks_statistic'] = None
            stats_dict['ks_pvalue'] = None
            stats_dict['ks_normal'] = None
        
        # Test normalità (Shapiro-Wilk, max 5000)
        try:
            sample = data.sample(min(5000, len(data)), random_state=42)
            shapiro_stat, shapiro_p = stats.shapiro(sample)
            stats_dict['shapiro_statistic'] = round(shapiro_stat, 4)
            stats_dict['shapiro_pvalue'] = round(shapiro_p, 4)
            stats_dict['shapiro_normal'] = shapiro_p > 0.05
        except:
            stats_dict['shapiro_statistic'] = None
            stats_dict['shapiro_pvalue'] = None
            stats_dict['shapiro_normal'] = None
        
        return stats_dict
    
    def _plot_distribution(self, data: pd.Series, signal: str, out_dir: Path):
        """Plot distribuzione con istogramma e KDE."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Istogramma
        axes[0].hist(data, bins=50, edgecolor='black', alpha=0.7, density=True)
        axes[0].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.2f}')
        axes[0].axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {data.median():.2f}')
        axes[0].set_xlabel(signal)
        axes[0].set_ylabel('Density')
        axes[0].set_title('Histogram')
        axes[0].legend()
        
        # QQ plot
        stats.probplot(data.sample(min(5000, len(data)), random_state=42), dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normal)')
        
        plt.suptitle(f'Distribution: {signal}', fontsize=12)
        plt.tight_layout()
        plt.savefig(out_dir / 'distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_time_series(self, df: pd.DataFrame, data: pd.Series, signal: str, out_dir: Path):
        """Plot serie temporale."""
        fig, ax = plt.subplots(figsize=(14, 4))
        
        # Usa datetime se disponibile
        if 'datetime' in df.columns:
            x = df['datetime'].iloc[data.index]
        else:
            x = data.index
        
        ax.plot(x, data.values, linewidth=0.5, alpha=0.8)
        ax.set_xlabel('Time')
        ax.set_ylabel(signal)
        ax.set_title(f'Time Series: {signal}')
        ax.grid(True, alpha=0.3)
        
        # Aggiungi media mobile per trend
        if len(data) > 100:
            window = min(len(data) // 20, 500)
            ma = data.rolling(window=window, center=True).mean()
            ax.plot(x, ma.values, color='red', linewidth=1.5, alpha=0.7, label=f'MA({window})')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(out_dir / 'time_series.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_acf_pacf(self, data: pd.Series, signal: str, out_dir: Path, lags: int = 50):
        """Plot ACF e PACF."""
        if len(data) < lags + 10:
            return
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # ACF
            acf_vals = acf(data, nlags=lags, fft=True)
            conf = 1.96 / np.sqrt(len(data))
            
            axes[0].bar(range(len(acf_vals)), acf_vals, width=0.3, color='steelblue')
            axes[0].axhline(y=conf, color='r', linestyle='--', alpha=0.5)
            axes[0].axhline(y=-conf, color='r', linestyle='--', alpha=0.5)
            axes[0].axhline(y=0, color='black', linewidth=0.5)
            axes[0].set_xlabel('Lag')
            axes[0].set_ylabel('ACF')
            axes[0].set_title('Autocorrelation Function')
            axes[0].set_ylim(-0.5, 1.1)
            
            # PACF
            pacf_vals = pacf(data, nlags=lags)
            axes[1].bar(range(len(pacf_vals)), pacf_vals, width=0.3, color='darkorange')
            axes[1].axhline(y=conf, color='r', linestyle='--', alpha=0.5)
            axes[1].axhline(y=-conf, color='r', linestyle='--', alpha=0.5)
            axes[1].axhline(y=0, color='black', linewidth=0.5)
            axes[1].set_xlabel('Lag')
            axes[1].set_ylabel('PACF')
            axes[1].set_title('Partial Autocorrelation Function')
            axes[1].set_ylim(-0.5, 1.1)
            
            plt.suptitle(f'ACF/PACF: {signal}', fontsize=12)
            plt.tight_layout()
            plt.savefig(out_dir / 'acf_pacf.png', dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"    ACF/PACF error: {e}")
    
    def _plot_stl(self, data: pd.Series, signal: str, out_dir: Path, downsample: int):
        """Decomposizione STL."""
        # Periodo: assumiamo ~1 ora se campionamento ~1s
        period = max(3600 // downsample, 10)
        
        if len(data) < period * 3:
            return
        
        try:
            stl = STL(data.values, period=period, robust=True)
            result = stl.fit()
            
            fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
            
            axes[0].plot(data.values, linewidth=0.5)
            axes[0].set_ylabel('Original')
            axes[0].set_title(f'STL Decomposition: {signal}')
            
            axes[1].plot(result.trend, linewidth=0.8, color='red')
            axes[1].set_ylabel('Trend')
            
            axes[2].plot(result.seasonal, linewidth=0.5, color='green')
            axes[2].set_ylabel('Seasonal')
            
            axes[3].plot(result.resid, linewidth=0.3, color='gray', alpha=0.7)
            axes[3].set_ylabel('Residual')
            axes[3].set_xlabel('Samples')
            
            plt.tight_layout()
            plt.savefig(out_dir / 'stl_decomposition.png', dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"    STL error: {e}")
    
    def _plot_boxplot(self, data: pd.Series, signal: str, out_dir: Path):
        """Boxplot per outliers."""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.boxplot(data.dropna(), vert=False, patch_artist=True,
                  boxprops=dict(facecolor='lightblue'))
        ax.set_xlabel(signal)
        ax.set_title(f'Boxplot: {signal}')
        
        # Calcola e mostra outliers
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((data < lower) | (data > upper)).sum()
        pct_outliers = n_outliers / len(data) * 100
        
        ax.text(0.02, 0.95, f'Outliers: {n_outliers:,} ({pct_outliers:.2f}%)',
               transform=ax.transAxes, fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(out_dir / 'boxplot.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # =========================================================================
    # ANALISI GLOBALE
    # =========================================================================
    
    def _correlation_matrix(self, df: pd.DataFrame, signal_cols: List[str], output_dir: Path):
        """Matrice correlazione tra tutti i segnali."""
        logger.info("  Matrice correlazione globale...")
        
        corr = df[signal_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, vmin=-1, vmax=1, ax=ax, annot_kws={'size': 8})
        plt.title('Correlation Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Salva correlazioni forti
        strong = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                r = corr.iloc[i, j]
                if abs(r) > 0.7:
                    strong.append({
                        'signal_1': corr.columns[i],
                        'signal_2': corr.columns[j],
                        'correlation': round(r, 4)
                    })
        
        if strong:
            pd.DataFrame(strong).sort_values('correlation', key=abs, ascending=False).to_csv(
                output_dir / 'strong_correlations.csv', index=False
            )
    
    def _summary_statistics(self, signal_cols: List[str], output_dir: Path):
        """Tabella riassuntiva di tutte le statistiche."""
        logger.info("  Sommario statistiche...")
        
        summary = []
        for signal in signal_cols:
            signal_dir = output_dir / signal
            stats_file = signal_dir / 'statistics.csv'
            if stats_file.exists():
                stats_df = pd.read_csv(stats_file)
                stats_df.insert(0, 'signal', signal)
                summary.append(stats_df)
        
        if summary:
            pd.concat(summary, ignore_index=True).to_csv(
                output_dir / 'summary_all_signals.csv', index=False
            )
    
    # =========================================================================
    # RUN EXPLORATION
    # =========================================================================
    
    def run_exploration(self, df: pd.DataFrame, dataset_name: str, downsample: int = None):
        """
        Esegue analisi completa per tutti i segnali.
        
        Args:
            df: DataFrame con dati processati (raw, no features derivate)
            dataset_name: Nome del dataset per la cartella output
            downsample: Fattore downsampling per velocizzare (default: from config)
        """
        if downsample is None:
            downsample = self.downsample
        
        # Cartella output per questo dataset
        output_dir = self.plots_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Identifica segnali
        signal_cols = self._get_signal_cols(df)
        
        logger.info("="*60)
        logger.info("EXPLORATORY DATA ANALYSIS")
        logger.info("="*60)
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Righe: {len(df):,}")
        logger.info(f"Segnali: {len(signal_cols)}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Downsample: {downsample}x")
        logger.info("="*60)
        
        # Converti datetime se necessario
        if 'datetime' in df.columns:
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Analisi per ogni segnale
        for i, signal in enumerate(signal_cols, 1):
            logger.info(f"[{i}/{len(signal_cols)}] {signal}")
            self.analyze_signal(df, signal, output_dir, downsample=downsample)
        
        # Analisi globale
        logger.info("Analisi globale...")
        self._correlation_matrix(df, signal_cols, output_dir)
        self._summary_statistics(signal_cols, output_dir)
        
        logger.info("="*60)
        logger.info(f"EDA completata! Output: {output_dir}")
        logger.info("="*60)


if __name__ == "__main__":
    import sys
    import yaml
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Usage: python exploration.py <input.csv> [dataset_name] [downsample]")
        print("\nEsempio:")
        print("  python exploration.py data/processed/file.csv my_dataset 10")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else Path(input_csv).stem
    downsample = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    # Carica dati
    print(f"Caricamento: {input_csv}")
    df = pd.read_csv(input_csv)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"Shape: {df.shape}")
    
    # Config minimale
    config = {
        'paths': {
            'base_directory': '.',
            'plots_directory': 'plots'
        },
        'exploration': {
            'downsample': downsample
        }
    }
    
    # EDA
    explorer = DataExplorer(config)
    explorer.run_exploration(df, dataset_name, downsample=downsample)
