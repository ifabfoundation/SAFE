"""
Feature Matching: Trova la migliore corrispondenza tra segnali Lab e Campo

Confronta tutte le combinazioni di:
- Temperature Lab vs Temperature Campo
- Vibrazioni Lab vs Accelerazioni Campo

Metriche di similarità:
1. Correlazione di Pearson (su dati normalizzati)
2. Distanza DTW (Dynamic Time Warping)
3. Similarità distribuzioni (KS test)
4. Similarità pattern (cross-correlation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
import warnings
import logging

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureMatcher:
    """Trova il miglior matching tra feature Lab e Campo."""
    
    # Feature Lab (esclusi fieldbus)
    LAB_VIB_FEATURES = [
        'vib_rid_marcia_rms',
        'vib_rid_marc2_rms', 
        'vib_mot_marcia_rms',
    ]
    
    LAB_TEMP_FEATURES = [
        'temp_cassa_riduttore',
        'temp_oil_marcia',
        'temp_mot_marcia',
        'temp_pt100_oil',
    ]
    
    # Feature Campo
    CAMPO_ACC_FEATURES = [
        'bonfi/gb1_p3_acc_rms',
        'bonfi/gb1_p4_acc_rms',
    ]
    
    CAMPO_TEMP_FEATURES = [
        'bonfi/gb1_p3_temp',
        'bonfi/gb1_p4_temp',
    ]
    
    def __init__(self, lab_csv_path: str, campo_csv_path: str, output_dir: str = None):
        """
        Args:
            lab_csv_path: Path a un CSV Lab processato (Caratterizzazione 24h)
            campo_csv_path: Path al CSV Tampieri
            output_dir: Directory output per risultati
        """
        self.lab_path = Path(lab_csv_path)
        self.campo_path = Path(campo_csv_path)
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.campo_path.parent.parent / 'plots' / 'feature_matching'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output: {self.output_dir}")
    
    def load_data(self, n_samples: int = 50000):
        """Carica e campiona dati da Lab e Campo."""
        logger.info("Caricamento dati...")
        
        # Lab
        self.df_lab = pd.read_csv(self.lab_path)
        if 'datetime' in self.df_lab.columns:
            self.df_lab['datetime'] = pd.to_datetime(self.df_lab['datetime'])
        logger.info(f"  Lab: {self.df_lab.shape}")
        
        # Campo
        self.df_campo = pd.read_csv(self.campo_path)
        if 'timestamp' in self.df_campo.columns:
            self.df_campo['timestamp'] = pd.to_datetime(self.df_campo['timestamp'])
        logger.info(f"  Campo: {self.df_campo.shape}")
        
        # Campiona per uniformare lunghezza
        n = min(n_samples, len(self.df_lab), len(self.df_campo))
        
        self.lab_sample = self.df_lab.sample(n, random_state=42).reset_index(drop=True)
        self.campo_sample = self.df_campo.sample(n, random_state=42).reset_index(drop=True)
        
        logger.info(f"  Campionati: {n} righe ciascuno")
        
        return self.lab_sample, self.campo_sample
    
    def normalize_features(self):
        """Normalizza le feature per confronto fair (z-score)."""
        self.scaler = StandardScaler()
        
        # Normalizza Lab
        self.lab_vib_norm = {}
        for col in self.LAB_VIB_FEATURES:
            if col in self.lab_sample.columns:
                data = self.lab_sample[col].dropna().values.reshape(-1, 1)
                self.lab_vib_norm[col] = self.scaler.fit_transform(data).flatten()
        
        self.lab_temp_norm = {}
        for col in self.LAB_TEMP_FEATURES:
            if col in self.lab_sample.columns:
                data = self.lab_sample[col].dropna().values.reshape(-1, 1)
                self.lab_temp_norm[col] = self.scaler.fit_transform(data).flatten()
        
        # Normalizza Campo
        self.campo_acc_norm = {}
        for col in self.CAMPO_ACC_FEATURES:
            if col in self.campo_sample.columns:
                data = self.campo_sample[col].dropna().values.reshape(-1, 1)
                self.campo_acc_norm[col] = self.scaler.fit_transform(data).flatten()
        
        self.campo_temp_norm = {}
        for col in self.CAMPO_TEMP_FEATURES:
            if col in self.campo_sample.columns:
                data = self.campo_sample[col].dropna().values.reshape(-1, 1)
                self.campo_temp_norm[col] = self.scaler.fit_transform(data).flatten()
        
        logger.info(f"  Normalizzati: {len(self.lab_vib_norm)} vib Lab, {len(self.lab_temp_norm)} temp Lab")
        logger.info(f"               {len(self.campo_acc_norm)} acc Campo, {len(self.campo_temp_norm)} temp Campo")
    
    def compute_similarity_metrics(self, data1: np.ndarray, data2: np.ndarray) -> dict:
        """
        Calcola metriche di similarità tra due serie.
        
        Returns:
            dict con varie metriche (più alto = più simile, tranne ks_statistic)
        """
        # Allinea lunghezze
        n = min(len(data1), len(data2))
        d1, d2 = data1[:n], data2[:n]
        
        metrics = {}
        
        # 1. Correlazione di Pearson
        try:
            corr, p_val = stats.pearsonr(d1, d2)
            metrics['pearson_corr'] = abs(corr)  # Abs perché anche correlazione negativa è info
            metrics['pearson_pval'] = p_val
        except:
            metrics['pearson_corr'] = 0
            metrics['pearson_pval'] = 1
        
        # 2. Correlazione di Spearman (rank-based, robusta a outlier)
        try:
            corr, p_val = stats.spearmanr(d1, d2)
            metrics['spearman_corr'] = abs(corr)
        except:
            metrics['spearman_corr'] = 0
        
        # 3. KS test - similarità distribuzioni (statistica bassa = simili)
        try:
            ks_stat, ks_pval = stats.ks_2samp(d1, d2)
            metrics['ks_statistic'] = ks_stat  # Più basso = più simili
            metrics['ks_similarity'] = 1 - ks_stat  # Convertito in similarità
        except:
            metrics['ks_statistic'] = 1
            metrics['ks_similarity'] = 0
        
        # 4. Cosine similarity (su istogrammi)
        try:
            hist1, _ = np.histogram(d1, bins=50, density=True)
            hist2, _ = np.histogram(d2, bins=50, density=True)
            cos_sim = 1 - cosine(hist1, hist2)
            metrics['histogram_cosine'] = cos_sim
        except:
            metrics['histogram_cosine'] = 0
        
        # 5. Differenza momenti statistici
        try:
            mean_diff = abs(np.mean(d1) - np.mean(d2))
            std_diff = abs(np.std(d1) - np.std(d2))
            skew_diff = abs(stats.skew(d1) - stats.skew(d2))
            kurt_diff = abs(stats.kurtosis(d1) - stats.kurtosis(d2))
            
            # Normalizzati sono già ~0 mean, ~1 std
            metrics['mean_diff'] = mean_diff
            metrics['std_diff'] = std_diff
            metrics['skew_diff'] = skew_diff
            metrics['kurt_diff'] = kurt_diff
            
            # Score complessivo momenti (inverso, più basso = più simile)
            metrics['moments_distance'] = mean_diff + std_diff + skew_diff/10 + kurt_diff/100
            metrics['moments_similarity'] = 1 / (1 + metrics['moments_distance'])
        except:
            metrics['moments_similarity'] = 0
        
        # 6. Score complessivo (media pesata)
        metrics['overall_score'] = (
            0.3 * metrics.get('pearson_corr', 0) +
            0.2 * metrics.get('spearman_corr', 0) +
            0.2 * metrics.get('ks_similarity', 0) +
            0.15 * metrics.get('histogram_cosine', 0) +
            0.15 * metrics.get('moments_similarity', 0)
        )
        
        return metrics
    
    def compare_all_combinations(self):
        """Confronta tutte le combinazioni Lab-Campo."""
        logger.info("\n" + "="*60)
        logger.info("CONFRONTO VIBRAZIONI (Lab) vs ACCELERAZIONI (Campo)")
        logger.info("="*60)
        
        vib_results = []
        for lab_col in self.lab_vib_norm:
            for campo_col in self.campo_acc_norm:
                metrics = self.compute_similarity_metrics(
                    self.lab_vib_norm[lab_col],
                    self.campo_acc_norm[campo_col]
                )
                metrics['lab_feature'] = lab_col
                metrics['campo_feature'] = campo_col
                vib_results.append(metrics)
        
        self.vib_comparison = pd.DataFrame(vib_results)
        self.vib_comparison = self.vib_comparison.sort_values('overall_score', ascending=False)
        
        print("\nVIBRAZIONI - Ranking per Overall Score:")
        print(self.vib_comparison[['lab_feature', 'campo_feature', 'overall_score', 
                                    'pearson_corr', 'ks_similarity', 'histogram_cosine']].to_string())
        
        logger.info("\n" + "="*60)
        logger.info("CONFRONTO TEMPERATURE (Lab) vs TEMPERATURE (Campo)")
        logger.info("="*60)
        
        temp_results = []
        for lab_col in self.lab_temp_norm:
            for campo_col in self.campo_temp_norm:
                metrics = self.compute_similarity_metrics(
                    self.lab_temp_norm[lab_col],
                    self.campo_temp_norm[campo_col]
                )
                metrics['lab_feature'] = lab_col
                metrics['campo_feature'] = campo_col
                temp_results.append(metrics)
        
        self.temp_comparison = pd.DataFrame(temp_results)
        self.temp_comparison = self.temp_comparison.sort_values('overall_score', ascending=False)
        
        print("\nTEMPERATURE - Ranking per Overall Score:")
        print(self.temp_comparison[['lab_feature', 'campo_feature', 'overall_score',
                                     'pearson_corr', 'ks_similarity', 'histogram_cosine']].to_string())
        
        # Salva risultati
        self.vib_comparison.to_csv(self.output_dir / 'vibration_matching.csv', index=False)
        self.temp_comparison.to_csv(self.output_dir / 'temperature_matching.csv', index=False)
        
        return self.vib_comparison, self.temp_comparison
    
    def plot_best_matches(self):
        """Visualizza i migliori match."""
        logger.info("\nGenerazione plot migliori match...")
        
        # Best vibration match
        best_vib = self.vib_comparison.iloc[0]
        lab_vib = best_vib['lab_feature']
        campo_acc = best_vib['campo_feature']
        
        # Best temperature match
        best_temp = self.temp_comparison.iloc[0]
        lab_temp = best_temp['lab_feature']
        campo_temp = best_temp['campo_feature']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # === ROW 1: Vibrazioni ===
        # Distribuzioni normalizzate
        axes[0, 0].hist(self.lab_vib_norm[lab_vib], bins=50, alpha=0.5, label=f'Lab: {lab_vib}', density=True)
        axes[0, 0].hist(self.campo_acc_norm[campo_acc], bins=50, alpha=0.5, label=f'Campo: {campo_acc.split("/")[-1]}', density=True)
        axes[0, 0].set_title(f'Best Vibration Match\nScore: {best_vib["overall_score"]:.3f}')
        axes[0, 0].legend()
        axes[0, 0].set_xlabel('Normalized Value')
        
        # Q-Q plot
        lab_sorted = np.sort(self.lab_vib_norm[lab_vib])
        campo_sorted = np.sort(self.campo_acc_norm[campo_acc])
        n = min(len(lab_sorted), len(campo_sorted))
        axes[0, 1].scatter(lab_sorted[:n:100], campo_sorted[:n:100], alpha=0.5, s=10)
        axes[0, 1].plot([lab_sorted.min(), lab_sorted.max()], [lab_sorted.min(), lab_sorted.max()], 'r--')
        axes[0, 1].set_xlabel(f'Lab: {lab_vib}')
        axes[0, 1].set_ylabel(f'Campo: {campo_acc.split("/")[-1]}')
        axes[0, 1].set_title('Q-Q Plot (Vibrazioni)')
        
        # Heatmap vibrazioni
        vib_pivot = self.vib_comparison.pivot(index='lab_feature', columns='campo_feature', values='overall_score')
        vib_pivot.columns = [c.split('/')[-1] for c in vib_pivot.columns]
        sns.heatmap(vib_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0, 2], 
                   vmin=0, vmax=1, cbar_kws={'label': 'Score'})
        axes[0, 2].set_title('Vibration Matching Matrix')
        
        # === ROW 2: Temperature ===
        # Distribuzioni normalizzate
        axes[1, 0].hist(self.lab_temp_norm[lab_temp], bins=50, alpha=0.5, label=f'Lab: {lab_temp}', density=True, color='red')
        axes[1, 0].hist(self.campo_temp_norm[campo_temp], bins=50, alpha=0.5, label=f'Campo: {campo_temp.split("/")[-1]}', density=True, color='darkred')
        axes[1, 0].set_title(f'Best Temperature Match\nScore: {best_temp["overall_score"]:.3f}')
        axes[1, 0].legend()
        axes[1, 0].set_xlabel('Normalized Value')
        
        # Q-Q plot
        lab_sorted = np.sort(self.lab_temp_norm[lab_temp])
        campo_sorted = np.sort(self.campo_temp_norm[campo_temp])
        n = min(len(lab_sorted), len(campo_sorted))
        axes[1, 1].scatter(lab_sorted[:n:100], campo_sorted[:n:100], alpha=0.5, s=10, color='red')
        axes[1, 1].plot([lab_sorted.min(), lab_sorted.max()], [lab_sorted.min(), lab_sorted.max()], 'b--')
        axes[1, 1].set_xlabel(f'Lab: {lab_temp}')
        axes[1, 1].set_ylabel(f'Campo: {campo_temp.split("/")[-1]}')
        axes[1, 1].set_title('Q-Q Plot (Temperature)')
        
        # Heatmap temperature
        temp_pivot = self.temp_comparison.pivot(index='lab_feature', columns='campo_feature', values='overall_score')
        temp_pivot.columns = [c.split('/')[-1] for c in temp_pivot.columns]
        sns.heatmap(temp_pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1, 2],
                   vmin=0, vmax=1, cbar_kws={'label': 'Score'})
        axes[1, 2].set_title('Temperature Matching Matrix')
        
        plt.suptitle('Feature Matching: Lab vs Campo', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_matching_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Salvato: feature_matching_summary.png")
    
    def plot_all_distributions(self):
        """Plot tutte le distribuzioni sovrapposte."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Vibrazioni
        colors_lab = plt.cm.Blues(np.linspace(0.4, 0.8, len(self.lab_vib_norm)))
        colors_campo = plt.cm.Oranges(np.linspace(0.4, 0.8, len(self.campo_acc_norm)))
        
        for i, (col, data) in enumerate(self.lab_vib_norm.items()):
            axes[0].hist(data, bins=50, alpha=0.4, label=f'Lab: {col}', density=True, color=colors_lab[i])
        for i, (col, data) in enumerate(self.campo_acc_norm.items()):
            axes[0].hist(data, bins=50, alpha=0.4, label=f'Campo: {col.split("/")[-1]}', density=True, 
                        color=colors_campo[i], histtype='step', linewidth=2)
        
        axes[0].set_title('Vibrazioni/Accelerazioni - Distribuzioni Normalizzate')
        axes[0].set_xlabel('Z-score')
        axes[0].legend()
        
        # Temperature
        colors_lab = plt.cm.Reds(np.linspace(0.3, 0.7, len(self.lab_temp_norm)))
        colors_campo = plt.cm.Purples(np.linspace(0.4, 0.8, len(self.campo_temp_norm)))
        
        for i, (col, data) in enumerate(self.lab_temp_norm.items()):
            axes[1].hist(data, bins=50, alpha=0.4, label=f'Lab: {col}', density=True, color=colors_lab[i])
        for i, (col, data) in enumerate(self.campo_temp_norm.items()):
            axes[1].hist(data, bins=50, alpha=0.4, label=f'Campo: {col.split("/")[-1]}', density=True,
                        color=colors_campo[i], histtype='step', linewidth=2)
        
        axes[1].set_title('Temperature - Distribuzioni Normalizzate')
        axes[1].set_xlabel('Z-score')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'all_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Salvato: all_distributions.png")
    
    def get_best_mapping(self) -> dict:
        """Restituisce il miglior mapping Lab -> Campo."""
        best_vib = self.vib_comparison.iloc[0]
        best_temp = self.temp_comparison.iloc[0]
        
        # Secondo miglior match per avere 2 coppie
        second_vib = self.vib_comparison[
            self.vib_comparison['campo_feature'] != best_vib['campo_feature']
        ].iloc[0]
        second_temp = self.temp_comparison[
            self.temp_comparison['campo_feature'] != best_temp['campo_feature']
        ].iloc[0]
        
        mapping = {
            'vibration': {
                'primary': {
                    'lab': best_vib['lab_feature'],
                    'campo': best_vib['campo_feature'],
                    'score': best_vib['overall_score']
                },
                'secondary': {
                    'lab': second_vib['lab_feature'],
                    'campo': second_vib['campo_feature'],
                    'score': second_vib['overall_score']
                }
            },
            'temperature': {
                'primary': {
                    'lab': best_temp['lab_feature'],
                    'campo': best_temp['campo_feature'],
                    'score': best_temp['overall_score']
                },
                'secondary': {
                    'lab': second_temp['lab_feature'],
                    'campo': second_temp['campo_feature'],
                    'score': second_temp['overall_score']
                }
            }
        }
        
        return mapping
    
    def run_matching(self):
        """Esegue l'analisi completa di matching."""
        logger.info("="*60)
        logger.info("FEATURE MATCHING: Lab vs Campo")
        logger.info("="*60)
        
        # Carica e normalizza
        self.load_data()
        self.normalize_features()
        
        # Confronta
        self.compare_all_combinations()
        
        # Plot
        self.plot_best_matches()
        self.plot_all_distributions()
        
        # Best mapping
        mapping = self.get_best_mapping()
        
        print("\n" + "="*60)
        print("MIGLIOR MAPPING CONSIGLIATO:")
        print("="*60)
        print("\nVIBRAZIONI:")
        print(f"   Primary:   {mapping['vibration']['primary']['lab']}")
        print(f"              ↔ {mapping['vibration']['primary']['campo']}")
        print(f"              Score: {mapping['vibration']['primary']['score']:.3f}")
        print(f"   Secondary: {mapping['vibration']['secondary']['lab']}")
        print(f"              ↔ {mapping['vibration']['secondary']['campo']}")
        print(f"              Score: {mapping['vibration']['secondary']['score']:.3f}")
        
        print("\nTEMPERATURE:")
        print(f"   Primary:   {mapping['temperature']['primary']['lab']}")
        print(f"              ↔ {mapping['temperature']['primary']['campo']}")
        print(f"              Score: {mapping['temperature']['primary']['score']:.3f}")
        print(f"   Secondary: {mapping['temperature']['secondary']['lab']}")
        print(f"              ↔ {mapping['temperature']['secondary']['campo']}")
        print(f"              Score: {mapping['temperature']['secondary']['score']:.3f}")
        
        logger.info("="*60)
        logger.info(f"Risultati salvati in: {self.output_dir}")
        logger.info("="*60)
        
        return mapping


if __name__ == "__main__":
    # Paths
    lab_path = "/home/projects/safe/data/bonfiglioli/processed/ETR_946 - Prova 30_0 - 310 - Caratterizzazione 24h_1.csv"
    campo_path = "/home/projects/safe/data/tampieri/processed_streaming_row_continuous.csv"
    
    # Run matching
    matcher = FeatureMatcher(lab_path, campo_path)
    best_mapping = matcher.run_matching()
